# Vision-Driven Self-Correcting Repair Agent

LangGraph agent that inspects a PCB image, classifies **fabrication defects**, retrieves **public workmanship text**, and self-corrects when confidence is low.

The repo today is a **Phase C + E** AOI agent (Phases A, B, C and E pass their gates; the Phase D OCR pipeline is built but its gate is blocked on FPIC data): canonical DeepPCB classes, a YOLOv8n detector trained on the official split, template-diff self-correction, a public RAG corpus (NASA / ECSS / arXiv / Wikipedia + adapter pages) with defect-class metadata filtering, and stage-wise eval scripts. Read [docs/BUILD.md](docs/BUILD.md) for directory map, learning strategy, and eval protocol. Datasets: [docs/DATASETS.md](docs/DATASETS.md). Requirements: [vision_repair_agent_plan.md](vision_repair_agent_plan.md).

### Current results

| Stage | Gate | Measured |
|---|---|---|
| CV — YOLOv8n, DeepPCB official test (500) | mAP@0.5 ≥ 0.85 | **0.967** (val 0.987) |
| RAG — `evals/rag_queries.jsonl` (34 queries, cross-encoder rerank) | Recall@5 ≥ 0.80 | **0.941** (MRR 0.865; dense-only 0.882 / 0.740) |
| Self-correction A/B — DeepPCB test (500), template verification vs off | measure | mAP 0.940 → **0.950**, micro-F1 0.940 → 0.936, fully-correct images 68.4% → 70.8% |
| Transfer — PKU-Market-PCB (693), no fine-tuning | measure | macro-F1 **0.024** (best of 5 preprocessings; does not transfer) |
| OCR — FPIC designators | exact-match ≥ 0.70 | not measured (FPIC needs PhysicalDB registration) |
| Diagnosis groundedness — 50 test images | ≥ 0.80 | not measured (needs `DEEPSEEK_API_KEY`) |
| Persistence — sessions survive restart | pass | **pass** (real Postgres) |

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
│   (MemorySaver default; Postgres via PERSISTENCE_BACKEND)       │
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
| State Persistence | MemorySaver by default; `PERSISTENCE_BACKEND=postgres` → LangGraph `AsyncPostgresSaver` + `diagnostic_sessions` rows |
| API Layer | FastAPI |

## Prerequisites

- **Python 3.11+** with Poetry
- **Docker** (optional — only for Postgres persistence; the API runs with in-memory checkpointing and FAISS by default)
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

# 4. (Optional) Start PostgreSQL and run migrations — only for PERSISTENCE_BACKEND=postgres
make up          # POSTGRES_PORT=5433 make up if 5432 is taken; match DATABASE_URL
make migrate     # alembic 001 → 002

# 6. Seed test corpus (for development)
poetry run python scripts/seed_test_docs.py
make ingest

# 7. Start the API server
make run            # in-memory sessions (lost on restart)
make run-postgres   # sessions + checkpoints persisted in Postgres
```

The API is now available at `http://localhost:8000`.
API docs: `http://localhost:8000/docs`

## API Usage

### Submit an image for diagnosis

```bash
curl -X POST http://localhost:8000/api/v1/diagnose \
  -F "file=@path/to/test_image.jpg" \
  -F "template=@path/to/template_image.jpg"   # optional; enables template verification
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "diagnosis": "## Defect Classification\nOpen circuit on copper trace...",
  "defect_type": "open",
  "defect_confidence": 0.87,
  "serial_number": null,
  "designator": null,
  "correction_mode": "template_diff",
  "detections": [{"cls": "open", "bbox": [412, 118, 441, 150], "score": 0.87}],
  "self_correction_triggered": true,
  "correction_attempts": 1,
  "rag_documents_used": 5,
  "cv_backend": "yolo",
  "template_provided": true,
  "persistence_backend": "memory"
}
```

Every call writes a session row (image/template SHA-256, detections, status). Failures are stored with `status=failed` and return the id in the `X-Session-Id` header.

### Fetch a session

```bash
curl http://localhost:8000/api/v1/sessions/550e8400-e29b-41d4-a716-446655440000
```

Returns the stored row plus a `checkpoint` summary of the final LangGraph state (thread id = session id), or 404. With `PERSISTENCE_BACKEND=memory`, sessions disappear on restart; with `postgres`, they survive it.

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

# Persistence against real Postgres (skips if unreachable; needs make up + make migrate)
make test-persistence
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

# Threshold sweep (val only) and self-correction A/B (template verification on vs off)
poetry run python scripts/prepare_deeppcb.py --gold-only
poetry run python evals/run_eval.py --stage sweep
poetry run python evals/run_eval.py --stage ab --limit 0        # 0 = all 500 test images

# Transfer: PKU-Market-PCB holdout, no fine-tuning (academic-use data, never trained on)
poetry run python -c "from huggingface_hub import snapshot_download; snapshot_download('RobotHuman/PCB_defect', repo_type='dataset', local_dir='data/raw/pku_pcb')"
poetry run python scripts/prepare_pku.py
poetry run python evals/run_eval.py --stage transfer --preprocess all

# Diagnosis groundedness (needs DEEPSEEK_API_KEY; --no-judge skips the LLM judge)
poetry run python evals/run_eval.py --stage diagnosis --limit 50

# Phase D — designator OCR (blocked until FPIC is downloaded; see docs/BUILD.md)
# Register at PhysicalDB, unzip pcb_image.zip + ocr_annotation.zip into data/raw/fpic/
poetry run python scripts/prepare_fpic.py
poetry run python evals/run_eval.py --stage ocr --split dev     # tune on dev only
poetry run python evals/run_eval.py --stage ocr --split test    # report once
```

Phases, learning strategy, and metrics: [docs/BUILD.md](docs/BUILD.md).

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | (required) | DeepSeek API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | DeepSeek API base URL |
| `DEEPSEEK_LLM_MODEL` | `deepseek-chat` | DeepSeek model for diagnosis |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model (downloaded on first run) |
| `RAG_RERANKER` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder that reranks FAISS candidates; `""` disables |
| `RAG_FETCH_K` | `20` | FAISS candidates before class filter + rerank |
| `DATABASE_URL` | — | Async PostgreSQL connection string |
| `SYNC_DATABASE_URL` | — | Sync PostgreSQL connection string |
| `PERSISTENCE_BACKEND` | `memory` | `memory` or `postgres` (checkpoints + session rows survive restart; needs `DATABASE_URL`) |
| `CORPUS_DIR` | `./docs/corpus` | Directory for RAG documents |
| `CV_BACKEND` | `heuristic` | `heuristic`, `template_diff`, or `yolo` (use `yolo` once `best.pt` exists; tests force `heuristic`) |
| `YOLO_WEIGHTS` | `data/processed/deeppcb/weights/best.pt` | Path to Phase B detector weights |
| `YOLO_CONF` | `0.55` | YOLO box operating point (val F1 sweep) |
| `YOLO_CANDIDATE_CONF` | `0.4` | Lowest YOLO box kept as a candidate for template verification |
| `CONFIDENCE_THRESHOLD` | `0.8` | Self-correct when the least confident candidate box is below this |
| `TEMPLATE_MIN_COVERAGE` | `0.05` | Fraction of a box an absdiff blob must cover to verify it |
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
