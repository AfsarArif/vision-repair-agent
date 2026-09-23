"""Retriever setup using FAISS for local vector storage."""

from __future__ import annotations

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from repair_agent.config import settings
from repair_agent.rag.metadata import matches_defect_filter
from repair_agent.rag.prompts import RETRIEVAL_K

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: FAISS | None = None
_rerankers: dict[str, object] = {}

FAISS_INDEX_DIR = Path(settings.CORPUS_DIR).resolve().parent / ".faiss_index"
_FILTER_OVERSAMPLE = 4


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        if FAISS_INDEX_DIR.exists():
            _vectorstore = FAISS.load_local(
                str(FAISS_INDEX_DIR),
                _get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        else:
            _vectorstore = FAISS.from_texts(["placeholder"], _get_embeddings())
    return _vectorstore


def _get_reranker(model_name: str):
    model = _rerankers.get(model_name)
    if model is None:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name, device="cpu")
        _rerankers[model_name] = model
    return model


def rerank(query: str, docs: list, model_name: str) -> list:
    """Order candidate chunks by cross-encoder relevance to the query."""
    if len(docs) < 2:
        return list(docs)
    scores = _get_reranker(model_name).predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(scores, range(len(docs))), key=lambda t: -float(t[0]))
    return [docs[i] for _, i in ranked]


def reset_vectorstore_cache() -> None:
    global _vectorstore
    _vectorstore = None


def get_retriever(k: int = RETRIEVAL_K):
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def _filter_docs(docs: list, defect_class: str | None, k: int) -> list:
    if not defect_class:
        return docs[:k]
    filtered = [doc for doc in docs if matches_defect_filter(doc.metadata, defect_class)]
    if filtered:
        return filtered[:k]
    return docs[:k]


async def aretrieve(
    query: str,
    k: int = RETRIEVAL_K,
    defect_class: str | None = None,
    reranker: str | None = None,
) -> list[dict]:
    """Retrieve top-k chunks, optionally preferring defect-tagged metadata.

    FAISS proposes `RAG_FETCH_K` candidates, the defect-class filter drops
    chunks tagged for other classes, and a cross-encoder reorders the rest.
    Pass `reranker=""` (or set `RAG_RERANKER=""`) for dense-only ranking.
    """
    vectorstore = get_vectorstore()
    model_name = settings.RAG_RERANKER if reranker is None else reranker
    if model_name:
        fetch_k = max(k, settings.RAG_FETCH_K)
    else:
        fetch_k = k if not defect_class else min(k * _FILTER_OVERSAMPLE, 50)
    docs = vectorstore.similarity_search(query, k=fetch_k)
    docs = _filter_docs(docs, defect_class, len(docs))
    if model_name:
        docs = rerank(query, docs, model_name)
    docs = docs[:k]
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": getattr(doc, "score", None),
        }
        for doc in docs
    ]
