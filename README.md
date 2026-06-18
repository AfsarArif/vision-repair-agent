# Vision-Driven Self-Correcting Repair Agent

A production-ready agentic AI system that autonomously diagnoses hardware defects by chaining Computer Vision (CV), OCR, and Retrieval-Augmented Generation (RAG) in a self-correcting feedback loop — with stateful memory across multi-step tool calls.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent Graph                    │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐ │
│  │ CV Node  │───▶│RAG Query │───▶│ OCR Node │───▶│  RAG  │ │
│  │(defect   │    │ (initial │    │(serial # │    │(re-   │ │
│  │ detect)  │    │ lookup)  │    │ extract) │    │query) │ │
│  └──────────┘    └──────────┘    └──────────┘    └───────┘ │
│        │                │               │              │    │
│        └────────────────┴───────────────┴──────────────┘   │
│                     Shared Agent State                      │
│              (PostgreSQL-backed checkpointing)              │
└─────────────────────────────────────────────────────────────┘
```

**Flow:**
1. Image input → CV node detects defect region and type
2. Initial RAG query with defect classification → retrieves candidate docs
3. If confidence < threshold → self-correction triggers
4. OCR node crops image region → extracts serial number
5. Re-queries RAG with serial number + defect type for precision results
6. Final diagnosis output with evidence citations from corpus

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph |
| LLM | DeepSeek (`deepseek-chat`, OpenAI-compatible) |
| Embeddings | Local `all-MiniLM-L6-v2` (sentence-transformers) |
| RAG Framework | LangChain + pgvector |
| Computer Vision | OpenCV + Pillow |
| OCR | Tesseract (pytesseract) |
| Vector Store | PostgreSQL + pgvector |
| State Persistence | PostgreSQL (LangGraph checkpointer) |
| API Layer | FastAPI |

## Prerequisites

- **Python 3.11+** with Poetry
- **Docker** (for PostgreSQL + pgvector)
- **Tesseract OCR** installed locally:
  ```bash
  # macOS
  brew install tesseract

  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr

  # Windows
  # Download from: https://github.com/UB-Mannheim/tesseract/wiki
  ```
- **DeepSeek API key** (set in `.env`) — get one at [platform.deepseek.com](https://platform.deepseek.com)
- **~2GB disk space** for the embedding model (auto-downloaded on first run)

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url> && cd vision-repair-agent

# 2. Install dependencies
poetry install

# 3. Configure environment
cp .env.example .env
# Edit .env with your DEEPSEEK_API_KEY

# 4. Start PostgreSQL with pgvector
make up

# 5. Run database migrations
make migrate

# 6. Seed test corpus (for development)
poetry run python scripts/seed_test_docs.py
make ingest

# 7. Start the API server
make run
```

The API is now available at `http://localhost:8000`.
API docs: `http://localhost:8000/docs`

## API Usage

### Submit an image for diagnosis

```bash
curl -X POST http://localhost:8000/api/v1/diagnose \
  -F "file=@path/to/hardware_image.png"
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "diagnosis": "## Defect Classification\nBurn mark detected on PCB...",
  "defect_type": "burn_mark",
  "defect_confidence": 0.87,
  "serial_number": "SN-XR9821A",
  "self_correction_triggered": true,
  "correction_attempts": 1,
  "rag_documents_used": 5
}
```

### Health check

```bash
curl http://localhost:8000/api/v1/health
```

## Adding Documents to Corpus

Drop PDFs or `.txt` files into `docs/corpus/` and run:

```bash
make ingest
```

The ingestion is incremental — previously ingested files are skipped.

## Running Tests

```bash
# All tests
make test

# Unit tests only
poetry run pytest tests/unit/ -v

# Integration tests only
poetry run pytest tests/integration/ -v
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | (required) | DeepSeek API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | DeepSeek API base URL |
| `DEEPSEEK_LLM_MODEL` | `deepseek-chat` | DeepSeek model for diagnosis |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model (downloaded on first run) |
| `DATABASE_URL` | — | Async PostgreSQL connection string |
| `SYNC_DATABASE_URL` | — | Sync PostgreSQL connection string |
| `CORPUS_DIR` | `./docs/corpus` | Directory for RAG documents |
| `CONFIDENCE_THRESHOLD` | `0.75` | Below this, self-correction triggers |
| `MAX_CORRECTION_RETRIES` | `3` | Max OCR + re-query retries |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Project Structure

```
vision-repair-agent/
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── Makefile
├── docs/corpus/           # RAG document corpus
├── scripts/
│   ├── ingest_corpus.py
│   └── seed_test_docs.py
├── src/repair_agent/
│   ├── config.py
│   ├── agent/
│   │   ├── graph.py       # LangGraph StateGraph
│   │   ├── state.py       # AgentState TypedDict
│   │   ├── edges.py       # Conditional edge logic
│   │   └── nodes/         # CV, OCR, RAG, Diagnosis nodes
│   ├── tools/             # CV, OCR, RAG utilities
│   ├── rag/               # Ingestion, retrieval, prompts
│   ├── db/                # ORM models, connection
│   └── api/               # FastAPI app + routes
└── tests/
    ├── unit/               # Unit tests
    └── integration/        # Integration tests
```
