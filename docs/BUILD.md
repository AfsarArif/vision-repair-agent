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
│   ├── data/deeppcb.py          ← official split + YOLO export layout
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

### Phase A — Harness

**Goal:** Make the project honest and measurable without training a detector.

Shipped in an earlier commit:

- Canonical classes (`open`, `short`, `mousebite`, `spur`, `spurious_copper`, `pin_hole`, `missing_hole`, `normal`). Heuristic CV still runs for tests/demo but **emits these ids**, not burn/corrosion.
- `CV_BACKEND=heuristic|template_diff|yolo` (default `heuristic` so pytest does not need weights).
- Template-diff localizer + self-correct node (template refine and/or designator OCR).
- Adapter markdown in `docs/corpus/adapters/`.
- Download / prepare scripts; eval runner; RAG gold queries.

**Not in Phase A:** a trained YOLO, ingested NASA PDFs (scripts only), Postgres in the API.

```bash
poetry install
poetry run pytest tests/ -v
poetry run python evals/run_eval.py --stage synthetic
```

### Phase B — Detector learning (this build)

**Goal:** Replace color heuristics with a detector trained on **DeepPCB’s official split**.

Shipped:

- `repair_agent.data.deeppcb` reads the real GitHub layout (`*_test.jpg` + sibling `*_not` labels + `trainval.txt` / `test.txt`).
- `prepare_deeppcb.py` writes **test images only** under `images/{split}/`. Templates go to `templates/{split}/` so Ultralytics does not treat unlabeled templates as train images. Label xywh uses each file’s actual size, not a hardcoded 640.
- `train_detector.py --run`: auto `cuda`/`mps`/`cpu`, freeze backbone 3 epochs then unfreeze, AdamW + cosine, no HSV/mosaic, copy `best.pt` to `data/processed/deeppcb/weights/best.pt`, report official **test** mAP once.
- YOLO model cache in `yolo_detector.py`. `CV_BACKEND=yolo` resolves `YOLO_WEIGHTS` or that default path; missing weights fall back to heuristic.
- `evals/run_eval.py --stage cv --backend yolo --weights ... --ultralytics-val`.

**Data.** 1,000 trainval pairs / 500 test pairs from Tang et al. (arXiv:1902.06197). **Do not** put PKU-Market-PCB or VisA into the training set.

**Carve validation from train only.** ~15% of the official 1,000 (about 850/150). Freeze the official 500 as test. Tune on val; report test **once**.

**Model.** YOLOv8n, **COCO-pretrained**, six foreground classes; DeepPCB type `1..6` → YOLO `0..5` via `taxonomy.DEEPPCB_ID_TO_CLASS`. Input 640×640.

**Why transfer learning.** 1,000 images is small. A COCO-initialized nano model learns “box + texture” faster than training from scratch.

**Loss.** Ultralytics default: box regression + classification + DFL.

**Augmentation (binary copper, not photos).** Horizontal/vertical flip, small translate/scale. **No hue/saturation jitter**. Mosaic/mixup off.

**Optimization.** 50 epochs default, AdamW, cosine LR, early stopping on val **mAP@0.5**, patience ~20. Freeze first 10 YOLO layers for 3 epochs, then train the rest unfrozen.

**Two systems, not one.**

| System | Learns classes? | Role |
|--------|-----------------|------|
| YOLO | Yes | Primary CV when `CV_BACKEND=yolo` |
| Template absdiff | No | Localization baseline; self-correct when a template is present |

**Selection rule.** Keep the checkpoint with best val mAP@0.5 (`weights/best.pt`). Never choose weights using the official test set.

```bash
poetry install --extras train
poetry run python scripts/download_public_data.py --deeppcb
poetry run python scripts/prepare_deeppcb.py
poetry run python scripts/train_detector.py --run
poetry run python evals/run_eval.py --stage cv --backend yolo --ultralytics-val
```

Point `.env` at the weights (`CV_BACKEND=yolo`, `YOLO_WEIGHTS=data/processed/deeppcb/weights/best.pt`). Tests stay on heuristic.

First-pass gate: **mAP@0.5 ≥ 0.85** on the official 500. The DeepPCB paper’s 98.6% is a published upper bound from a different architecture, not our claim. Logged numbers live in gitignored `evals/results/train_summary.json`.

### Phase C — Public RAG (this build)

**Goal:** Hundreds of chunks from public documents, filtered by `defect_classes` metadata.

Shipped:

- `docs/corpus/manifest.json` maps PDFs and Wikipedia extracts to `source_id`, `license`, and `defect_classes`.
- `ingest_corpus.py` ingests only `adapters/`, `wikipedia/`, and `pdfs/` (CI stub `.txt` files at corpus root are excluded).
- Adapters keep inline `source_id` / `defect_classes` frontmatter; chunking stays 1000 / 150 with MiniLM embeddings.
- `aretrieve(..., defect_class=...)` oversamples then filters chunks whose metadata matches the detector class (general docs with empty classes still pass through).
- `rag_node` passes the current `defect_type` into retrieval.
- Expanded `evals/rag_queries.jsonl` (adapters + Wikipedia + arXiv + standards).

```bash
poetry run python scripts/download_public_data.py --corpus --wikipedia
poetry run python scripts/ingest_corpus.py --rebuild
poetry run python evals/run_eval.py --stage rag
```

Gate: **Recall@5 ≥ 0.80** on `evals/rag_queries.jsonl`. Do not ingest IPC-A-610 / J-STD-001 / IPC-7711.

**Hardening (after C).**

- *Scoring fix.* `run_eval.py --stage rag` used to rank alias ids (`open`, `adapter-open`, …) instead of retrieved chunks, so a correct first hit scored MRR 0.5 and "top 5" covered ~2–3 documents. It now ranks the k chunks the diagnosis prompt actually sees.
- *Ingest fix.* Adapter frontmatter was stripped before metadata was read, so adapters were indexed with `source_id=<stem>` and **no** `defect_classes` (they passed every class filter). `_load_markdown` now passes the parsed frontmatter through. Rebuild the index after pulling.
- *Gold set.* 34 queries (was 15): paraphrases that avoid adapter vocabulary, plus passage-level questions for each PDF and Wikipedia source.
- *Reranker.* FAISS proposes `RAG_FETCH_K=20` candidates, the class filter drops other-class chunks, and `cross-encoder/ms-marco-MiniLM-L-6-v2` reorders them (`RAG_RERANKER=""` disables; tests force it off).

| 34 queries | Recall@5 | MRR |
|---|---|---|
| Dense only (`--reranker ""`) | 0.882 | 0.740 |
| + cross-encoder (default) | **0.941** | **0.865** |

The original 15 queries score Recall@5 0.933 / MRR 0.756 dense-only under the fixed scoring (0.469 MRR before). Remaining misses: `q-short-wiki` (the PCB-manufacturing article's short-circuit text loses to the adapter) and `q-open-para`.

### Phase D — Designator OCR (color boards only)

Regex `[RCULJQDW]\d{1,4}` on silkscreen crops. Evaluate exact-match on FPIC, **not** on DeepPCB. Target ≥ 0.70. Optional second engine (EasyOCR) only if Tesseract misses that bar.

**Status: pipeline built, gate not yet measured.** FPIC requires a (free) PhysicalDB account, so the data is not downloaded.

1. Register at the [FPIC page](https://physicaldb.ece.ufl.edu/index.php/fics-pcb-image-collection-fpic/) and download `pcb_image.zip` + `ocr_annotation.zip`.
2. Unzip to `data/raw/fpic/` → `pcb_image/*.png`, `ocr_annotation/*.csv`.
3. `python scripts/prepare_fpic.py` → `data/processed/fpic/gold_ocr.jsonl` (`split=fpic_ocr`, `subset` dev ≈20% / test, assigned per source board so no board is in both).
4. Tune on dev only: `python evals/run_eval.py --stage ocr --split dev`; report test once: `--split test`.

`ocr_tools.read_designator`: upscale → Otsu → polarity flip (light-on-dark silkscreen) → rotations 0/90/270/180 → Tesseract `--psm 7` with a designator whitelist → digit-confusion fixes (O→0, I/l→1, S→5, …) → confidence vote. The CSV parser follows the FPIC paper's column description and has not yet run on real files — check the first run. Synthetic PIL crops (CI smoke only, **not** the gate): dev 0.927, test 0.940. EasyOCR (`--ocr-engine easyocr`) only if Tesseract misses 0.70 on FPIC dev.

### Phase E — Persistence (done)

Diagnose sessions survive a process restart. Set `PERSISTENCE_BACKEND=postgres` and `DATABASE_URL=postgresql+asyncpg://…`; the API lifespan opens a psycopg pool for LangGraph's `AsyncPostgresSaver` (conninfo derived from `DATABASE_URL` by dropping `+asyncpg`) and closes it on shutdown. `get_agent(url)` is an async context manager.

- Every `POST /api/v1/diagnose` writes a `diagnostic_sessions` row (status, defect type, confidence, designator, correction mode, CV backend, image/template SHA-256, detections JSON). LangGraph thread id = session id. Failures are stored with `status=failed`.
- `GET /api/v1/sessions/{session_id}` returns the row plus the latest checkpoint summary, or 404.
- `POST /diagnose` accepts an optional `template` file (DeepPCB-style defect-free template), which enables template verification in the self-correct node.
- Default `PERSISTENCE_BACKEND=memory` needs no database; sessions are lost on restart.

```bash
POSTGRES_PORT=5433 make up          # POSTGRES_PORT only if 5432 is taken; match DATABASE_URL
make migrate                        # alembic 001 → 002
make run-postgres
make test-persistence               # TEST_DATABASE_URL; skips if Postgres is unreachable
```

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
- `--stage cv` — needs `--gold` from `prepare_deeppcb.py`. Pass `--backend yolo` after training. `--ultralytics-val` adds Ultralytics’ own mAP.
- `--stage ocr` — skipped until FPIC gold exists.

**Leakage rules.** Test images never in training. Val is a subset of DeepPCB train, not of test. PKU never in the YOLO loss. Wikipedia/NASA text is not used as CV labels.

**Confidence threshold.** Sweep YOLO (or heuristic) confidence vs F1 on **val**, then freeze `CONFIDENCE_THRESHOLD`. Do not retune on test.

`run_eval.py --stage sweep` (val only; it refuses a `*test*` gold file) runs YOLO once at conf 0.01 and sweeps offline. `prepare_deeppcb.py --gold-only` writes `gold_val.jsonl` without wiping images or weights. Val results (150 images):

- Box F1 peaks at **conf 0.55** (micro-F1 0.969 vs 0.952 at the old 0.25) → `YOLO_CONF=0.55`.
- The top detection is correct on **100%** of val images and scores ≥ 0.85, so the old gate (`defect_confidence < 0.75` on the top box) **never fired**. Errors live in low-scoring boxes: only 70% of images have every box right.
- The gate now reads `min_confidence` (lowest candidate box ≥ `YOLO_CANDIDATE_CONF=0.4`) against `CONFIDENCE_THRESHOLD=0.8`, i.e. "some box is in the band template verification can act on". It fires on ~71% of images (template verification is a few ms of OpenCV).

**Self-correction A/B.** Same 50 images with template-diff on vs off; report both. No required lift until measured.

`run_eval.py --stage ab [--limit 50|0]`. Fusion parameters were chosen on val; test was scored once.

- `off` — YOLO at `YOLO_CONF`.
- `replace` — the pre-fusion node: when the gate fired, class-less absdiff blobs **replaced** YOLO's boxes. Raw absdiff on DeepPCB yields 70–100 blobs per image (edge noise from sub-pixel misalignment) and gold boxes are padded, so this was destructive.
- `verify` — the current node (`tools/fusion.py`): boxes ≥ 0.8 stay; boxes in [0.4, 0.8) stay only if an absdiff blob covers ≥ 5% of them; the diff never adds a class-less box.

| DeepPCB test, 500 images | off | replace (old) | verify (new) |
|---|---|---|---|
| mAP@0.5 class-aware | 0.940 | 0.226 | **0.950** |
| micro-F1 | **0.940** | 0.049 | 0.936 |
| recall / precision | 0.953 / 0.927 | 0.264 / 0.027 | 0.963 / 0.910 |
| images with every box right | 68.4% | 26.2% | **70.8%** |

The gate fires on 71% of test images. First 50 test images: mAP 0.927 → 0.938, perfect images 54% → 60%, micro-F1 0.834 → 0.833. Val (tuning split): micro-F1 0.969 → 0.974. Net: verification trades ~2 pt precision for ~1 pt recall; it is roughly neutral on F1 and positive on mAP and per-image correctness. The old node, run behind the same gate, cuts micro-F1 from 0.94 to 0.05. (This mAP is the repo's VOC-style `metrics.py`; Ultralytics reports 0.967 for the same weights.)

**Transfer (PKU-Market-PCB holdout).** `scripts/prepare_pku.py` turns the 693 original (unrotated) PKU images and VOC boxes (Hugging Face mirror `RobotHuman/PCB_defect`, still under PKU's academic-only terms) into `data/processed/pku/gold_holdout.jsonl`. It stays in `data/raw/pku_pcb/` and never enters training. `run_eval.py --stage transfer [--preprocess all|none|tile|binarize|binarize_tile|binarize_gray_tile]` runs the DeepPCB YOLOv8n with no fine-tuning and reports macro-F1 (conf 0.25, IoU 0.5) over the five shared classes. `missing_hole` (PKU only) has recall 0 by construction; `pin_hole` (DeepPCB only) is reported as out-of-vocabulary detections. All variants were fixed before scoring:

| variant | macro-F1 | class-agnostic recall |
|---|---|---|
| none (color, letterboxed) | 0.000 | 0.000 |
| tile (color, native-res 640 tiles) | 0.000 | 0.002 |
| **binarize** (green-channel Otsu) | **0.024** | 0.094 |
| binarize_tile | 0.009 | 0.181 |
| binarize_gray_tile | 0.000 | 0.049 |

The detector is blind on color photos. Binarized, it fires near real defects (binarize_tile catches 52% of opens) but precision is ~0.1–1%, because the via-dot grid in PKU's copper pour looks exactly like DeepPCB pin-holes. A model trained on binarized DeepPCB does not transfer to solder-masked color boards. Treat this as the baseline for any future domain adaptation, never as an accuracy claim.

**Diagnosis groundedness.** `run_eval.py --stage diagnosis [--limit 50] [--no-judge]` runs the full agent (template attached) on the first 50 test images and scores each report with `eval/groundedness.py`:

- deterministic — every `[source_id]` citation was retrieved, every standard / clause reference (`IPC-A-610`, `NASA-STD-8739.3`, `clause 10.2.1`, …) appears in retrieved text, and the detector class is named;
- LLM judge — claims supported by passages (same model family as the writer, so spot-check by hand).

The diagnosis prompt now labels passages `[source_id] (filename)` so citations are checkable. Needs `DEEPSEEK_API_KEY`; skipped otherwise. Per-image rows go to `evals/results/diagnosis_per_image.jsonl`. Note that the report describes the top detection only, while DeepPCB images carry 3–12 defects.

Results write to `evals/results/` (gitignored). Commit the script and the gold queries, not the score JSON.

---

## 5. What “done” looks like per phase

| Phase | Done when |
|-------|-----------|
| A | `pytest` green; `run_eval.py --stage synthetic` prints IoU; adapters exist; download/prepare scripts run |
| B | YOLO weights at `data/processed/deeppcb/weights/best.pt`; test mAP@0.5 logged in `evals/results/`; heuristic is fallback only |
| C | Recall@5 ≥ 0.80 on `rag_queries.jsonl` after `ingest_corpus.py --rebuild` |
| D | Designator exact-match ≥ 0.70 on FPIC gold |
| E | `test_persistence.py` passes against Postgres: a session and its checkpoint read back from a brand-new app instance; memory backend returns 404 after restart |

Until B, treat API diagnoses on real DeepPCB photos as a **pipeline demo**, not an accuracy claim.
