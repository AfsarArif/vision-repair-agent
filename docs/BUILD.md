# Build guide — structure, phases, learning, eval

This is the high-level map of the repo: what each directory is for, what **Phase A** (this build) shipped, and how later phases train and score models. Requirements and dataset licenses live in [vision_repair_agent_plan.md](../vision_repair_agent_plan.md) and [DATASETS.md](DATASETS.md).

The product is an AOI assistant: **find copper-pattern defects on a PCB image**, **retrieve public workmanship text**, **self-correct** when unsure. It is not a certified NASA/IPC inspector.

---

## 1. Runtime architecture

```
test image  (+ optional aligned template)
        │
        ▼
   [cv]  heuristic | template_diff | yolo
        │  detections[{cls, bbox, score}]
        ▼
   [rag_initial]  query = class [+ designator]
        │
        ├── confidence ≥ threshold ─────────────► [diagnosis] → END
        │
        ▼
   [self_correct]
        template present → absdiff vs template → new boxes
        always           → OCR for R12/C3-style designators
        │
        ▼
   [rag_corrected] → [diagnosis] → END
```

Shared state is a LangGraph `AgentState`. The API (`POST /api/v1/diagnose`) runs this graph with in-memory checkpointing. Postgres is optional and unused by the default path.

**Self-correction is domain-specific.** DeepPCB images are binarized copper with a template pair — OCR cannot help. Color boards (FPIC / VisA) can yield a silkscreen designator. The self-correct node does **both**; whichever signal exists is used.

---

## 2. Repository structure

```
vision-repair-agent/
├── docs/
│   ├── BUILD.md                 ← this file
│   ├── DATASETS.md              ← licenses, URLs, class mapping
│   └── corpus/
│       ├── adapters/            ← class → public-source index (in git)
│       ├── *.txt                ← CI stubs only
│       └── pdfs/                ← gitignored NASA/ESA/arXiv downloads
├── src/repair_agent/
│   ├── taxonomy.py              ← canonical classes + DeepPCB/PKU maps
│   ├── config.py                ← env (DeepSeek, CV_BACKEND, paths)
│   ├── agent/                   ← LangGraph: state, nodes, edges, graph
│   ├── tools/                   ← OpenCV, template-diff, OCR, YOLO loader
│   ├── rag/                     ← ingest, FAISS retrieve, prompts
│   ├── eval/                    ← IoU / mAP@0.5 / Recall@k (no I/O)
│   ├── api/                     ← FastAPI
│   └── db/                      ← optional SQLAlchemy + Alembic
├── scripts/
│   ├── download_public_data.py  ← DeepPCB clone, PDFs, Wikipedia extracts
│   ├── prepare_deeppcb.py       ← raw DeepPCB → YOLO layout + gold JSONL
│   ├── train_detector.py        ← Phase B entry (Ultralytics)
│   ├── ingest_corpus.py
│   └── seed_test_docs.py
├── evals/
│   ├── rag_queries.jsonl        ← retrieval gold (in git)
│   ├── gold.example.jsonl       ← schema example
│   ├── run_eval.py              ← stage-wise scoring
│   └── results/                 ← gitignored JSON reports
├── data/raw/  data/processed/   ← gitignored binaries
└── tests/                       ← unit + integration (synthetic images)
```

| Path | Role |
|------|------|
| `taxonomy.py` | Single list of class ids. Training, RAG metadata, and eval all import this. |
| `tools/template_diff.py` | Non-learned localizer: `|test − template|` → boxes. Baseline and self-correct path. |
| `tools/yolo_detector.py` | Loads Phase B weights if `CV_BACKEND=yolo` and the file exists. |
| `eval/metrics.py` | Pure functions so `pytest` and `run_eval.py` share the same IoU/mAP. |
| `docs/corpus/adapters/` | Short pages we wrote that **cite** NASA/ESA/Wikipedia/arXiv. They join a DeepPCB class name to retrievable text. |

Downloaded images and PDFs never go in git.

---

## 3. Build phases

### Phase A — Harness (this build)

**Goal:** Make the project honest and measurable without training a detector.

Shipped:

- Canonical classes (`open`, `short`, `mousebite`, `spur`, `spurious_copper`, `pin_hole`, `missing_hole`, `normal`). Heuristic CV still runs for tests/demo but **emits these ids**, not burn/corrosion.
- `CV_BACKEND=heuristic|template_diff|yolo` (default `heuristic`).
- Template-diff localizer + self-correct node (template refine and/or designator OCR).
- Adapter markdown in `docs/corpus/adapters/`.
- Download / prepare scripts; eval runner; RAG gold queries.
- `docs/BUILD.md` (this document).

**Not in Phase A:** a trained YOLO, ingested NASA PDFs (scripts only), Postgres in the API.

Run:

```bash
poetry install
poetry run pytest tests/ -v
poetry run python evals/run_eval.py --stage synthetic
# optional, needs network:
poetry run python scripts/download_public_data.py --corpus --wikipedia
poetry run python scripts/ingest_corpus.py
poetry run python evals/run_eval.py --stage rag
```

### Phase B — Detector learning

**Goal:** Replace color heuristics with a detector trained on **DeepPCB’s official split**.

**Data.** 1,000 train pairs / 500 test pairs from Tang et al. (arXiv:1902.06197). Convert with `scripts/prepare_deeppcb.py`. **Do not** put PKU-Market-PCB or VisA into the training set. PKU is transfer eval only; VisA is an optional anomaly gate (different label space).

**Carve validation from train only.** Example: 850 train / 150 val from the 1,000, freeze the official 500 as test. Tune on val; report test **once**.

**Model.** YOLOv8n (or equivalent nano detector), **COCO-pretrained backbone**, fine-tune all heads. Six foreground classes; DeepPCB type `1..6` maps to YOLO `0..5` via `taxonomy.DEEPPCB_ID_TO_CLASS`. Input 640×640 (native DeepPCB size).

**Why transfer learning.** 1,000 images is small. A COCO-initialized nano model learns “box + texture” faster than training from scratch. We are not using ImageNet classification weights as the final head — detection heads are trained on DeepPCB boxes.

**Loss.** Ultralytics default: box regression + classification + DFL. No extra re-ID or metric learning.

**Augmentation (binary copper, not photos).** Horizontal/vertical flip, small translate/scale. **No hue/saturation jitter** (images are already thresholded). Mosaic/mixup off or very light — they invent impossible trace geometry.

**Optimization.** ~50–100 epochs, AdamW, cosine LR, early stopping on val **mAP@0.5**, patience ~20. Optional: freeze backbone 3 epochs, then unfreeze.

**Not a class-imbalance trick we rely on.** DeepPCB puts several defects per image (~3–12). If one class lags on val, class weights are allowed; do not oversample PKU to “fix” it.

**Two systems, not one.**

| System | Learns classes? | Role |
|--------|-----------------|------|
| YOLO | Yes | Primary CV when `CV_BACKEND=yolo` |
| Template absdiff | No | Localization baseline; self-correct when a template is present |

Template-diff does not output a class. Eval reports **localization** (IoU / class-agnostic AP) for it, and **mAP@0.5** for YOLO.

**Selection rule.** Keep the checkpoint with best val mAP@0.5. Point `YOLO_WEIGHTS` at it. Never choose weights using the official test set.

```bash
poetry install --extras train
poetry run python scripts/download_public_data.py --deeppcb
poetry run python scripts/prepare_deeppcb.py
poetry run python scripts/train_detector.py --run
poetry run python evals/run_eval.py --stage cv --gold data/processed/deeppcb/gold_test.jsonl
```

First-pass gate: **mAP@0.5 ≥ 0.85** on the official 500. The DeepPCB paper’s 98.6% is a published upper bound from a different architecture, not our claim.

### Phase C — Public RAG

**Goal:** Hundreds of chunks from ~15–25 public documents, filtered by `defect_classes` metadata.

Ingest NASA-STD-8739.6B / 8739.1B, ECSS-Q-ST-70-61C, cancelled NASA-STD-8739.3, arXiv PDFs, Wikipedia extracts, plus adapters. Chunk size 1000 / overlap 150. Embed with local MiniLM. Gold: `evals/rag_queries.jsonl` → **Recall@5 ≥ 0.80**.

Do not ingest IPC-A-610 / J-STD-001 / IPC-7711 (paid). Do not add a paid reranker until this Recall@5 is the bottleneck.

### Phase D — Designator OCR (color boards only)

Regex `[RCULJQDW]\d{1,4}` on silkscreen crops. Evaluate exact-match on FPIC, **not** on DeepPCB. Target ≥ 0.70. Optional second engine (EasyOCR) only if Tesseract misses that bar.

### Phase E — Persistence (optional)

Wire `DiagnosticSession` and `AsyncPostgresSaver` into the diagnose route if we need resumable threads. Not required for accuracy.

---

## 4. Eval protocol

Four stages, four numbers. Never average them into one “90%.”

| Stage | Gold | Metric | When |
|-------|------|--------|------|
| CV detection | DeepPCB official test JSONL | mAP@0.5 (class-aware) | Phase B |
| CV localization | Same, class ignored | mAP@0.5 class-agnostic | Phase A baseline (template-diff) |
| Transfer | PKU holdout | macro-F1 | After B |
| RAG | `evals/rag_queries.jsonl` | Recall@5, MRR | Phase C (adapters work in A) |
| OCR | FPIC crops | exact-match | Phase D |
| Diagnosis | 50-image subset | groundedness (cite-only) | After C, costly |

`evals/run_eval.py`:

- `--stage synthetic` — no downloads: template-diff IoU on generated pairs (CI).
- `--stage rag` — needs a FAISS index (ingest adapters at minimum).
- `--stage cv` — needs `--gold` from `prepare_deeppcb.py`.
- `--stage ocr` — skipped until FPIC gold exists.

**Leakage rules.** Test images never in training. Val is a subset of DeepPCB train, not of test. PKU never in the YOLO loss. Wikipedia/NASA text is not used as CV labels.

**Confidence threshold.** Sweep YOLO (or heuristic) confidence vs F1 on **val**, then freeze `CONFIDENCE_THRESHOLD`. Do not retune on test.

**Self-correction A/B.** Same 50 images with template-diff on vs off; report both. No required lift until measured.

Results write to `evals/results/` (gitignored). Commit the script and the gold queries, not the score JSON.

---

## 5. What “done” looks like per phase

| Phase | Done when |
|-------|-----------|
| A | `pytest` green; `run_eval.py --stage synthetic` prints IoU; adapters exist; download/prepare scripts run |
| B | YOLO weights + test mAP@0.5 logged; heuristic is fallback only |
| C | Recall@5 ≥ 0.80 on `rag_queries.jsonl` |
| D | Designator exact-match ≥ 0.70 on FPIC gold |
| E | Diagnose sessions survive process restart |

Until B, treat API diagnoses on real DeepPCB photos as a **pipeline demo**, not an accuracy claim.
