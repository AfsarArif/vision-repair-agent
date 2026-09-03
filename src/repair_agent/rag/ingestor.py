"""Document ingestion pipeline: load, chunk, embed, and store in FAISS."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from repair_agent.config import settings
from repair_agent.rag.metadata import (
    defect_classes_to_meta,
    load_manifest,
    manifest_entry_for_path,
    parse_defect_classes,
    parse_frontmatter,
    validate_manifest_classes,
)
from repair_agent.rag.prompts import CHUNK_OVERLAP, CHUNK_SIZE

INGESTION_LOG = Path(settings.CORPUS_DIR) / ".ingestion_log.json"
FAISS_INDEX_DIR = Path(settings.CORPUS_DIR).resolve().parent / ".faiss_index"


def load_ingestion_log() -> set[str]:
    if INGESTION_LOG.exists():
        return set(json.loads(INGESTION_LOG.read_text(encoding="utf-8")))
    return set()


def save_ingestion_log(files: set[str]) -> None:
    INGESTION_LOG.write_text(json.dumps(sorted(files), indent=2), encoding="utf-8")


def _apply_metadata(doc: Document, path: Path, manifest: dict[str, dict]) -> None:
    entry = manifest_entry_for_path(path, manifest)
    frontmatter, _ = parse_frontmatter(doc.page_content)
    source_id = frontmatter.get("source_id") or entry.get("source_id") or path.stem.lower()
    license_tag = frontmatter.get("license") or entry.get("license") or ""
    classes = parse_defect_classes(
        frontmatter.get("defect_classes") or entry.get("defect_classes") or []
    )
    doc.metadata["source"] = str(path.resolve())
    doc.metadata["source_id"] = source_id
    if license_tag:
        doc.metadata["license"] = license_tag
    doc.metadata["defect_classes"] = defect_classes_to_meta(classes)


def _load_markdown(path: Path, manifest: dict[str, dict]) -> Document:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    frontmatter, body = parse_frontmatter(raw)
    doc = Document(page_content=body or raw, metadata={"source": str(path.resolve())})
    if frontmatter:
        doc.page_content = body or raw
    _apply_metadata(doc, path, manifest)
    return doc


def _load_pdf(path: Path, manifest: dict[str, dict]) -> list[Document]:
    docs = PyPDFLoader(str(path)).load()
    for doc in docs:
        doc.metadata["source"] = str(path.resolve())
        _apply_metadata(doc, path, manifest)
    return docs


def collect_documents(corpus_dir: Path) -> list[Document]:
    manifest = load_manifest()
    validate_manifest_classes(manifest)
    docs: list[Document] = []
    for pattern, kind in (
        ("adapters/**/*.md", "md"),
        ("wikipedia/**/*.md", "md"),
        ("pdfs/**/*.pdf", "pdf"),
    ):
        for path in sorted(corpus_dir.glob(pattern)):
            if not path.is_file():
                continue
            if kind == "md":
                docs.append(_load_markdown(path, manifest))
            else:
                docs.extend(_load_pdf(path, manifest))
    return docs


async def ingest_corpus(incremental: bool = True, rebuild: bool = False) -> dict:
    """Walk scoped corpus folders, chunk, embed, and store in FAISS."""
    corpus_dir = Path(settings.CORPUS_DIR).resolve()
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    if rebuild:
        await delete_index()

    previously_ingested = load_ingestion_log() if incremental and not rebuild else set()
    all_docs = collect_documents(corpus_dir)

    if incremental and not rebuild:
        all_docs = [d for d in all_docs if d.metadata.get("source") not in previously_ingested]

    if not all_docs:
        print("No new documents to ingest.")
        return {
            "new_documents": 0,
            "new_chunks": 0,
            "total_files_tracked": len(previously_ingested),
        }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Split {len(all_docs)} documents into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.LOCAL_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if FAISS_INDEX_DIR.exists() and incremental and not rebuild:
        print("Merging with existing FAISS index...")
        existing = FAISS.load_local(
            str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        existing.add_documents(chunks)
        existing.save_local(str(FAISS_INDEX_DIR))
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(FAISS_INDEX_DIR))

    new_files = {str(d.metadata.get("source", "")) for d in all_docs}
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
    import shutil

    if FAISS_INDEX_DIR.exists():
        shutil.rmtree(FAISS_INDEX_DIR)
    if INGESTION_LOG.exists():
        INGESTION_LOG.unlink()
    print("FAISS index deleted.")


if __name__ == "__main__":
    asyncio.run(ingest_corpus())
