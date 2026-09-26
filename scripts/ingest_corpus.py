#!/usr/bin/env python3
"""One-shot script: load public corpus → chunk → embed → store in FAISS."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repair_agent.rag.ingestor import ingest_corpus
from repair_agent.rag.retriever import reset_vectorstore_cache


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the FAISS index and re-ingest all scoped corpus files",
    )
    args = parser.parse_args()
    try:
        stats = await ingest_corpus(incremental=not args.rebuild, rebuild=args.rebuild)
        reset_vectorstore_cache()
        print(f"Ingestion successful: {stats}")
        return 0
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Ingestion failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
