"""Document ingestion pipeline: load, chunk, embed, and store in FAISS."""

import asyncio
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from repair_agent.config import settings
from repair_agent.rag.prompts import CHUNK_SIZE, CHUNK_OVERLAP

# Track ingested files to support incremental ingestion
INGESTION_LOG = Path(settings.CORPUS_DIR) / ".ingestion_log.json"
FAISS_INDEX_DIR = Path(settings.CORPUS_DIR).resolve().parent / ".faiss_index"


def load_ingestion_log() -> set[str]:
    """Load set of previously ingested file paths."""
    if INGESTION_LOG.exists():
        return set(json.loads(INGESTION_LOG.read_text()))
    return set()


def save_ingestion_log(files: set[str]) -> None:
    """Save set of ingested file paths."""
    INGESTION_LOG.write_text(json.dumps(sorted(files), indent=2))


async def ingest_corpus(incremental: bool = True) -> dict:
    """Walk CORPUS_DIR for documents, chunk, embed, and store in FAISS.

    Args:
        incremental: If True, skip previously ingested files.

    Returns:
        dict with ingestion statistics.
    """
    corpus_dir = Path(settings.CORPUS_DIR).resolve()
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    previously_ingested = load_ingestion_log() if incremental else set()

    # Load PDFs
    pdf_loader = DirectoryLoader(
        str(corpus_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    pdf_docs = pdf_loader.load()

    # Load text files
    txt_loader = DirectoryLoader(
        str(corpus_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    txt_docs = txt_loader.load()

    all_docs = pdf_docs + txt_docs

    # Filter out already-ingested files
    if incremental:
        all_docs = [d for d in all_docs if d.metadata.get("source") not in previously_ingested]

    if not all_docs:
        print("No new documents to ingest.")
        return {"new_documents": 0, "new_chunks": 0, "total_files_tracked": len(previously_ingested)}

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    print(f"Split {len(all_docs)} documents into {len(chunks)} chunks.")

    # Embed with local model
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.LOCAL_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Merge with existing index if present
    if FAISS_INDEX_DIR.exists() and incremental:
        print("Merging with existing FAISS index...")
        existing = FAISS.load_local(
            str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        existing.add_documents(chunks)
        existing.save_local(str(FAISS_INDEX_DIR))
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(FAISS_INDEX_DIR))

    # Update ingestion log
    new_files = {d.metadata.get("source", "") for d in all_docs}
    all_files = previously_ingested | new_files
    save_ingestion_log(all_files)

    stats = {
        "new_documents": len(all_docs),
        "new_chunks": len(chunks),
        "total_files_tracked": len(all_files),
    }
    print(f"Ingestion complete: {stats}")
    return stats


async def delete_index() -> None:
    """Delete the FAISS index (for full re-ingestion)."""
    import shutil

    if FAISS_INDEX_DIR.exists():
        shutil.rmtree(FAISS_INDEX_DIR)
    if INGESTION_LOG.exists():
        INGESTION_LOG.unlink()
    print("FAISS index deleted.")


if __name__ == "__main__":
    asyncio.run(ingest_corpus())
