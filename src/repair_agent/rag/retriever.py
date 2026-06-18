"""Retriever setup using FAISS for local vector storage."""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from repair_agent.config import settings
from repair_agent.rag.prompts import RETRIEVAL_K

# Module-level singletons
_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: FAISS | None = None

FAISS_INDEX_DIR = Path(settings.CORPUS_DIR).resolve().parent / ".faiss_index"


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-load the local embedding model (cached after first call)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore() -> FAISS:
    """Load or create a FAISS vector store from the persisted index."""
    global _vectorstore
    if _vectorstore is None:
        if FAISS_INDEX_DIR.exists():
            _vectorstore = FAISS.load_local(
                str(FAISS_INDEX_DIR),
                _get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        else:
            # Create empty — caller should run ingest first
            _vectorstore = FAISS.from_texts(
                ["placeholder"], _get_embeddings()
            )
    return _vectorstore


def get_retriever(k: int = RETRIEVAL_K):
    """Get a retriever configured with top-k retrieval."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


async def aretrieve(query: str, k: int = RETRIEVAL_K) -> list[dict]:
    """Async retrieval: query the FAISS vector store and return documents with metadata.

    Args:
        query: The search query string.
        k: Number of documents to retrieve.

    Returns:
        List of dicts with keys: content, metadata, score.
    """
    retriever = get_retriever(k=k)
    docs = await retriever.ainvoke(query)
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": getattr(doc, "score", None),
        }
        for doc in docs
    ]
