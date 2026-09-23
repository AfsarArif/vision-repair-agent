# Vision-Driven Self-Correcting Repair Agent

LangGraph agent that inspects a PCB image, classifies **fabrication defects**, retrieves **public workmanship text**, and self-corrects when confidence is low.

The repo today is a **Phase C** AOI agent: canonical DeepPCB classes, a YOLOv8n detector trained on the official split, template-diff self-correction, a public RAG corpus (NASA / ECSS / arXiv / Wikipedia + adapter pages) with defect-class metadata filtering, and stage-wise eval scripts. Read [docs/BUILD.md](docs/BUILD.md) for directory map, learning strategy, and eval protocol. Datasets: [docs/DATASETS.md](docs/DATASETS.md). Requirements: [vision_repair_agent_plan.md](vision_repair_agent_plan.md).

### Current results

| Stage | Gate | Measured |
|---|---|---|
| CV — YOLOv8n, DeepPCB official test (500) | mAP@0.5 ≥ 0.85 | **0.967** (val 0.987) |
| RAG — `evals/rag_queries.jsonl` | Recall@5 ≥ 0.80 | **0.933** (MRR 0.469, 15 queries) |

Numbers come from gitignored `evals/results/*.json`; rerun the commands under [Eval and data](#eval-and-data) to reproduce.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     LangGraph Agent Graph                        │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌───────┐      │
│  │ CV Node  │──▶│RAG Query │──▶│ Self-correct│──▶│  RAG  │      │
│  │(DeepPCB  │   │ (class   │   │ template    │   │(re-   │      │
│  │ classes) │   │  lookup) │   │ diff or OCR │   │query) │      │
│  └──────────┘   └──────────┘   └────────────┘   └───────┘      │
│        │               │              │              │          │
│        └───────────────┴──────────────┴──────────────┘          │
│                      Shared Agent State                         │
│              (MemorySaver today; Postgres optional)             │
└──────────────────────────────────────────────────────────────────┘
```

**Flow (target):**
1. Image (+ optional DeepPCB-style template) → detector proposes boxes in `{open, short, mousebite, spur, spurious_copper, pin_hole, missing_hole, normal}`
2. RAG retrieves NASA / ESA / Wikipedia / arXiv passages for that class
3. If confidence is low: **template absdiff** (DeepPCB) or **silkscreen designator OCR** (FPIC/VisA color boards) — not fictional serial numbers
4. Re-query RAG and synthesize a diagnosis that cites retrieved sources only

**Flow (what the code does today):** YOLO (`CV_BACKEND=yolo`) when `data/processed/deeppcb/weights/best.pt` exists, otherwise heuristic or template-diff → FAISS over adapter markdown → self-correct (template absdiff and/or designator OCR) → DeepSeek. Tests stay on heuristic. Accuracy is the DeepPCB test mAP logged after `train_detector.py --run`, not a single blended “90%.”

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph |
| LLM | DeepSeek (`deepseek-chat`, OpenAI-compatible) |
| Embeddings | Local `all-MiniLM-L6-v2` (sentence-transformers) |
| RAG Framework | LangChain + FAISS (pgvector optional) |
| Computer Vision | OpenCV heuristic or template-diff; YOLO when weights exist |
| OCR | Tesseract — designators (`R12`) plus legacy serial fixtures |
| Vector Store | FAISS on disk (Postgres optional) |
| State Persistence | In-memory MemorySaver (Postgres checkpointer optional) |
| API Layer | FastAPI |

## Prerequisites

- **Python 3.11+** with Poetry
- **Docker** (optional — only if you want Postgres/pgvector; the API runs with in-memory checkpointing and FAISS)
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
      "diagnosis": "## Defect Classification\nOpen circuit on copper trace...",
      "defect_type": "open",
      "defect_confidence": 0.87,
      "serial_number": "R12",
  "self_correction_triggered": true,
  "correction_attempts": 1,
  "rag_documents_used": 5
}
```

### Health check

```bash
curl http://localhost:8000/api/v1/health
```

## Data and corpus

Public datasets, licenses, and download URLs: **[docs/DATASETS.md](docs/DATASETS.md)**.

- **Vision:** [DeepPCB](https://github.com/tangsanli5201/DeepPCB) (1,500 template/test pairs, 6 classes). Optional: PKU-Market-PCB, VisA PCB1–4.
- **RAG:** NASA-STD-8739.6B / 8739.1B, ECSS-Q-ST-70-61C, cancelled-but-public NASA-STD-8739.3, arXiv dataset papers, Wikipedia (CC BY-SA). Not IPC-A-610 (paid).
- **Do not commit** downloads. They belong in gitignored `data/raw/` and `docs/corpus/pdfs/`.

The five files already in `docs/corpus/` are **CI stubs** (burn/corrosion/serial). Replace them with adapter pages that map DeepPCB class names onto the public standards.

```bash
make ingest
```

Ingestion is incremental — previously ingested files are skipped.

## Running Tests

```bash
# All tests
make test

# Unit tests only
poetry run pytest tests/unit/ -v

# Integration tests only
poetry run pytest tests/integration/ -v
```

## Eval and data

```bash
# Template-diff IoU on generated pairs (no download)
make eval

# Phase C: public PDFs + Wikipedia extracts (gitignored), rebuilt index, Recall@5
poetry run python scripts/download_public_data.py --corpus --wikipedia
poetry run python scripts/ingest_corpus.py --rebuild
poetry run python evals/run_eval.py --stage rag

# Phase B detector
poetry install --extras train
poetry run python scripts/download_public_data.py --deeppcb
poetry run python scripts/prepare_deeppcb.py
poetry run python scripts/train_detector.py --run
poetry run python evals/run_eval.py --stage cv --backend yolo --ultralytics-val
```

Phases, learning strategy, and metrics: [docs/BUILD.md](docs/BUILD.md).

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
| `CV_BACKEND` | `heuristic` | `heuristic`, `template_diff`, or `yolo` (use `yolo` once `best.pt` exists; tests force `heuristic`) |
| `YOLO_WEIGHTS` | `data/processed/deeppcb/weights/best.pt` | Path to Phase B detector weights |
| `CONFIDENCE_THRESHOLD` | `0.75` | Below this, self-correction triggers |
| `MAX_CORRECTION_RETRIES` | `3` | Max template-diff / OCR re-query retries |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Project Structure

```
vision-repair-agent/
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── Makefile
├── docs/
│   ├── BUILD.md           # Structure, phases, learning, eval
│   ├── DATASETS.md        # Public data registry
│   └── corpus/            # Adapters + CI stubs; PDFs gitignored
├── evals/                 # Gold queries + run_eval.py
├── scripts/
│   ├── download_public_data.py
│   ├── prepare_deeppcb.py
│   ├── train_detector.py
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
