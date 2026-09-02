# Vision Repair Agent — Requirements

A LangGraph agent that inspects a PCB image, classifies **publicly labeled fabrication defects**, retrieves **public workmanship documents**, and (when unsure) self-corrects using **template comparison** or **silkscreen OCR** — whichever the image source actually supports.

Dataset registry, licenses, and download URLs: [docs/DATASETS.md](docs/DATASETS.md).

---

## 1. What this project is (and is not)

**Is:** An automated optical inspection (AOI) assistant for **bare-board copper-pattern defects**, with RAG over NASA / ESA / Wikipedia / arXiv text that a real inspector would be allowed to copy.

**Is not:** A field-repair bot for burn marks, corrosion, and delamination against 500 proprietary service manuals. That corpus does not exist in public form, and those class names do not appear on DeepPCB or PKU-Market-PCB.

**Is not yet:** A production system. The repo today is a working FastAPI + LangGraph **scaffold** (see §2). These requirements define the realistic product to build on top of that scaffold.

---

## 2. What is already built (scan of `main`)

Single commit (`68f574a`, 2026-06-18). Runnable skeleton, not a measured diagnostic system.

| Piece | Status |
|-------|--------|
| LangGraph: CV → RAG → (optional OCR) → RAG → diagnosis | Implemented |
| FastAPI `POST /api/v1/diagnose` | Implemented; in-memory `MemorySaver` (sessions not persisted) |
| CV | OpenCV **heuristics** (HSV / aspect ratio) for `burn_mark`, `crack`, `corrosion`, `delamination` |
| OCR | Tesseract looking for fictional `SN-…` patterns on the **defect crop** |
| RAG | Local FAISS + `all-MiniLM-L6-v2`; **5 short `.txt` stubs** in `docs/corpus/` |
| LLM | DeepSeek (`deepseek-chat`), not GPT-4o |
| Postgres + pgvector + LangGraph checkpointer | Docker Compose + Alembic exist; **API path does not use them** |
| Tests | Unit + integration on **synthetic** images; LLM/RAG mocked. No eval harness, no real photos |
| Claimed “90%+ accuracy / 500+ documents / 70% faster resolution” | **Not implemented and not measurable** with current data |

Self-correction as coded (`should_self_correct`) gates on **CV contour-area confidence**, not retrieval quality, then OCRs the defect bounding box. On a real DeepPCB crop there is no serial number to read.

---

## 3. Why the original requirements were not realistic

1. **Label mismatch.** Public PCB sets use `open`, `short`, `mousebite`, `spur`, `spurious_copper`, `pin_hole` / `missing_hole`. The code uses field-failure names that those sets never annotate.
2. **No public 500-document hardware corpus.** IPC-A-610 and IPC-7711/7721 (the actual industry manuals) are **paid**. We cannot require them.
3. **OCR cannot be the DeepPCB self-correction path.** DeepPCB is aligned, binarized copper with a **template image per sample**. That pair *is* the public self-correction signal.
4. **One end-to-end “90%” is undefined.** CV mAP, OCR exact-match, RAG Recall@k, and diagnosis groundedness are four different numbers.

---

## 4. Product requirements (revised)

### 4.1 Inputs

- A PCB **test** image (PNG/JPEG).
- Optional aligned **template** image (DeepPCB-style). If omitted, skip template-diff self-correction.
- Optional thread id for checkpointed runs (when Postgres is wired).

### 4.2 Outputs

Structured diagnosis including:

- One or more detections: `{cls, bbox, score}` using **canonical classes** in [docs/DATASETS.md](docs/DATASETS.md).
- Retrieved passages with `source_id`, filename/URL, and license.
- Whether self-correction ran (`template_diff` | `designator_ocr` | none) and how many attempts.
- Repair / disposition **grounded in retrieved text** (scrap, isolate-and-etch, jumper, send to electrical test). No invented part numbers or IPC clause numbers that were not retrieved.

### 4.3 Agent graph (target)

```
test image [+ optional template]
        │
        ▼
 [CV detect]  YOLO (DeepPCB-trained)  or  template-absdiff proposal
        │
        ▼
 [RAG initial]  query = defect class (+ optional designator)
        │
        ├── score OK ──────────────────────────────► [diagnosis]
        │
        ▼
 [self-correct]
    A. template present  → crop + absdiff vs template → re-score / new boxes
    B. color / silkscreen → OCR reference designator (R12, C3) → richer RAG query
        │
        ▼
 [RAG corrected] → [diagnosis] → END
```

Keep LangGraph, FastAPI, DeepSeek, local embeddings, and FAISS (Postgres optional). Replace the heuristic classifier and the serial-number story.

### 4.4 Defect taxonomy

Canonical ids: `open`, `short`, `mousebite`, `spur`, `spurious_copper`, `pin_hole`, `missing_hole`, `normal`.

Mapping from DeepPCB / PKU names is in [docs/DATASETS.md](docs/DATASETS.md). Do not add `burn_mark` / `corrosion` unless we later collect and license a separate labeled set.

### 4.5 Self-correction (must match the image domain)

| Image source | Self-correction | Why |
|--------------|-----------------|-----|
| DeepPCB | Template absdiff on low-score boxes | Every sample ships a registered template; no text on the board |
| VisA PCB / FPIC | Tesseract (or EasyOCR) on silkscreen ROI → designator | Public OCR labels exist; DeepPCB detector may not apply |
| CI synthetics | Rendered `R12`-style text | Regression only |

Maximum 3 correction attempts. Stop when a designator is found, template-diff raises score above threshold, or attempts are exhausted.

### 4.6 RAG corpus

Ingest **only** the public sources listed in [docs/DATASETS.md](docs/DATASETS.md):

- NASA-STD-8739.6B, NASA-STD-8739.1B (current, public).
- NASA-STD-8739.3 (cancelled, still public) for solder open/short language — cite as historical.
- ECSS-Q-ST-70-61C (free ESA assembly/workmanship standard; large).
- arXiv papers that **define** the six defect classes.
- Wikipedia articles (CC BY-SA; store URL + license on every chunk).
- Short in-repo **adapter** markdown that maps each canonical class to those sources.

**Out of corpus:** IPC J-STD-001, IPC-A-610, IPC-7711/7721, scraped vendor blogs without a clear license, the five original serial/burn/corrosion stubs (keep only as test fixtures until adapters replace them).

Success looks like **~15–25 documents → hundreds of chunks**, not 500 PDFs.

### 4.7 Engineering constraints (keep)

- Python 3.11+, Poetry, FastAPI, LangGraph.
- LLM: DeepSeek OpenAI-compatible API.
- Embeddings: local `all-MiniLM-L6-v2` (no OpenAI embedding spend).
- Vector store: FAISS on disk for local/dev; pgvector remains optional for persistence.
- Downloaded datasets and PDFs are gitignored. Commit scripts, manifests, and citations.

---

## 5. Accuracy requirements (measurable)

Drop the single “90%+ diagnostic accuracy” line. Report **per stage** on the gold splits in [docs/DATASETS.md](docs/DATASETS.md).

| Stage | Split | Metric | First-pass target | Notes |
|-------|-------|--------|-------------------|--------|
| CV detection | DeepPCB official 500 test | mAP@0.5 | ≥ 0.85 | Paper model reports 98.6% mAP — that is an **upper bound**, not our claim |
| CV classification (if using max-box only) | Same | macro-F1 | ≥ 0.80 | |
| Template-diff baseline | Same | mAP@0.5 | measure, no gate | Must beat “always `open`” |
| Transfer | PKU holdout | macro-F1 | measure | Expect a drop vs DeepPCB |
| Anomaly gate (optional) | VisA PCB1–4 | image AUROC | ≥ 0.80 | Not a 6-class score |
| Designator OCR | FPIC crops | exact-match | ≥ 0.70 | Not evaluated on DeepPCB |
| RAG | ~30 labeled queries | Recall@5 | ≥ 0.80 | Query → expected `source_id` |
| Diagnosis | 50-image subset | groundedness | ≥ 0.80 | Defect name from CV; no clause/page that was not retrieved (LLM-as-judge + spot human check) |

Self-correction A/B: same 50 images with template-diff on vs off. Ship the table; do not require a specific lift until it is measured.

`CONFIDENCE_THRESHOLD` is tuned from a precision-recall sweep on DeepPCB val, not left at 0.75 by folklore.

---

## 6. Implementation phases

### Phase A — Honesty and harness (shipped)

Implemented in this tree:

- Canonical class enum in `src/repair_agent/taxonomy.py`; heuristic CV emits these ids.
- `CV_BACKEND=heuristic|template_diff|yolo`.
- Template-diff localizer + `self_correct` node.
- Adapter markdown in `docs/corpus/adapters/`.
- `scripts/download_public_data.py`, `prepare_deeppcb.py`, `train_detector.py`.
- `evals/run_eval.py` + `evals/rag_queries.jsonl`.
- [docs/BUILD.md](docs/BUILD.md) — structure, later phases, learning strategy, eval.

### Phase B — Train CV on DeepPCB (this build)

- Convert DeepPCB boxes to YOLO using the official `trainval.txt` / `test.txt` lists; templates stay out of `images/train`.
- Train YOLOv8n (COCO init) with freeze-then-unfreeze; copy `best.pt` to `data/processed/deeppcb/weights/best.pt`.
- Template-diff remains the localization baseline and self-correction path.
- Report mAP on the official test set in `evals/results/` (gitignored).

### Phase C — Public RAG

- Ingest NASA / ECSS / arXiv / Wikipedia + adapter pages.
- Metadata filter by `defect_classes` before vector search.
- Gold queries for Recall@5.

### Phase D — OCR path (optional, color boards)

- Designator regex (`[RCULJQDW]\d{1,4}`) instead of `SN-`.
- Evaluate on FPIC; do not claim OCR accuracy on DeepPCB.

### Phase E — Persistence (optional)

- Wire `DiagnosticSession` + Postgres checkpointer in the API if we need resumable threads.

Do not add a paid reranker until Recall@5 on the gold queries is the bottleneck.

---

## 7. Tech stack (current → keep)

| Layer | Technology |
|-------|------------|
| Agent | LangGraph `StateGraph` |
| LLM | DeepSeek `deepseek-chat` |
| Embeddings | `all-MiniLM-L6-v2` |
| RAG | LangChain + FAISS (pgvector optional) |
| Detection | Ultralytics YOLO (Phase B); OpenCV template-diff |
| OCR | Tesseract (Phase D only) |
| API | FastAPI |
| Tests | pytest |

---

## 8. Directory additions (planned)

```
data/raw/                 # gitignored downloads (DeepPCB, PKU, VisA, PDFs)
data/processed/           # YOLO layout, eval images
docs/corpus/pdfs/         # gitignored
docs/corpus/adapters/     # class → source index (in git)
docs/DATASETS.md          # this registry
evals/gold.jsonl
evals/run_eval.py
scripts/download_public_data.py
scripts/prepare_deeppcb.py
```

Existing layout (`src/repair_agent/`, `tests/`, `scripts/ingest_corpus.py`) stays.

---

## 9. License and citation obligations

- DeepPCB: research/demo; cite Tang et al., arXiv:1902.06197.
- PKU-Market-PCB: academic only; do not republish the zip; cite Huang & Wei, arXiv:1901.08204.
- VisA: CC BY 4.0 (upstream README); cite Zou et al., ECCV 2022.
- NASA standards: public technical standards; cite document number and revision.
- ECSS-Q-ST-70-61C: ESA copyright; free to download, keep the PDF intact, cite the standard.
- Wikipedia: CC BY-SA; attribution URL on every chunk.

This project is a **research / portfolio demo**, not a certified NASA or IPC inspection tool.

---

*Phase B detector training is in the tree. Weights stay gitignored under `data/processed/`. See [docs/BUILD.md](docs/BUILD.md).*
