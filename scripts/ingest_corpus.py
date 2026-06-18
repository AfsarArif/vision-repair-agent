#!/usr/bin/env python3
"""One-shot script: load documents from CORPUS_DIR → chunk → embed → store in pgvector."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repair_agent.rag.ingestor import ingest_corpus


async def main():
    try:
        stats = await ingest_corpus(incremental=True)
        print(f"✅ Ingestion successful: {stats}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
