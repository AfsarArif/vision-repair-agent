# Public datasets and corpus

This project trains, evaluates, and retrieves against **publicly downloadable** sources only. Images stay out of git (`data/raw/`, `data/processed/`). PDFs stay out of git (`docs/corpus/pdfs/`). Commit manifests, licenses, and citations — not the binaries.

Do **not** ingest IPC-A-610, IPC-7711/7721, or IPC J-STD-001. Those are paid industry standards.

Full requirements: [vision_repair_agent_plan.md](../vision_repair_agent_plan.md).

---

## Canonical defect classes

The agent classifies **bare-board copper-pattern defects** used by public AOI datasets, plus `normal`.

| Canonical id | DeepPCB id / name | PKU-Market-PCB name | Meaning |
|--------------|-------------------|---------------------|---------|
| `open` | 1 — open | open circuit | Trace or pad path is broken |
| `short` | 2 — short | short | Unintended copper connection |
| `mousebite` | 3 — mousebite | mouse bite | Nick / partial bite in a trace edge |
| `spur` | 4 — spur | spur | Unwanted protrusion from a trace |
| `spurious_copper` | 5 — copper | spurious copper | Isolated extra copper |
| `pin_hole` | 6 — pin-hole | — | Hole in a copper feature |
| `missing_hole` | — | missing hole | Expected drill / hole is absent |
| `normal` | (template / no box) | — | No defect above threshold |

Field-repair labels from the original plan (`burn_mark`, `crack`, `corrosion`, `delamination`) are **retired**. They have no public labeled image set that matches this agent.

---

## Vision

### Primary — DeepPCB (train + official test)

| | |
|--|--|
| **What** | 1,500 aligned pairs: defect-free **template** + defective **test**, 640×640, binarized copper |
| **Labels** | Axis-aligned boxes, 6 classes, ~3–12 defects per test image |
| **Split** | Official: 1,000 train / 500 test |
| **Why it fits** | Same task as the CV node (localize + classify PCB defects). Template images enable the self-correction path **without OCR**. |
| **Download** | [github.com/tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB) |
| **Paper** | Tang et al., [arXiv:1902.06197](https://arxiv.org/abs/1902.06197) |
| **License** | GitHub: MIT. Paper states the set is released for **research**. Treat this repo as a research/demo project; do not ship the images in a commercial product without re-checking. |
| **Format** | `{id}_test.jpg`, `{id}_temp.jpg`, `{id}.txt` with `x1 y1 x2 y2 type` |
| **Local path** | `data/raw/deeppcb/` |

YOLO conversion: class index is `type - 1` (DeepPCB is 1-based). Run `scripts/prepare_deeppcb.py` after cloning.

### Secondary — PKU-Market-PCB (transfer / color photos)

| | |
|--|--|
| **What** | 1,386 synthetic color PCB images, 6 defect types |
| **Why it fits** | Same fabrication-defect vocabulary on **color** boards (DeepPCB is binary). Use as a held-out transfer set, not as the training split. |
| **Download** | [PKU HRI datasets (EN)](https://robotics.pkusz.edu.cn/resources/datasetENG/) |
| **Paper** | Huang & Wei, [arXiv:1901.08204](https://arxiv.org/abs/1901.08204) |
| **License** | **Academic research only** (lab page). Do not redistribute the zip in this repo. |
| **Augmented YOLO/COCO mirror** | [huggingface.co/datasets/YangNexus/pcb](https://huggingface.co/datasets/YangNexus/pcb) — still inherit PKU terms |
| **Local path** | `data/raw/pku_pcb/` (gitignored) |

### Optional — VisA PCB1–PCB4 (anomaly gate)

| | |
|--|--|
| **What** | Color photos of assembled boards. PCB1–4: ~4,016 normal + 400 anomalous, pixel masks |
| **Why it fits** | Binary “is this board anomalous?” before the DeepPCB detector. **Not** labeled with the six copper-pattern classes. |
| **Download** | [amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) |
| **Paper / code** | [arXiv:2207.14315](https://arxiv.org/abs/2207.14315), [github.com/amazon-science/spot-diff](https://github.com/amazon-science/spot-diff) |
| **License** | Upstream README: **CC BY 4.0** |
| **Local path** | `data/raw/visa/` |

Use only the `pcb1`–`pcb4` subsets.

---

## OCR / self-correction text

DeepPCB images are **binarized copper**. They have no serial numbers and no silkscreen. Cropping a defect box and running Tesseract cannot succeed on that set.

| Source | Role | URL | License |
|--------|------|-----|---------|
| DeepPCB templates | Primary self-correction: `absdiff(test_crop, template_crop)` when detector confidence is low | DeepPCB repo above | MIT / research |
| FPIC (FICS PCB Image Collection) | Secondary: silkscreen **reference designators** (`R12`, `C3`, `U7`) on color boards | [trust-hub.org data](https://www.trust-hub.org/#/data/pcb-images), [arXiv:2202.08414](https://arxiv.org/abs/2202.08414) | Check Trust-Hub terms before download |
| Synthetic overlays | CI only: render `R12`-style text with PIL (replace `SN-ABC12345` fixtures) | in-repo tests | ours |

Retired: fictional `SN-XR9821A` lookups against a serial-number guide that does not exist.

---

## RAG corpus (text)

Target: **15–25 public documents** that chunk to several hundred passages — not “500 proprietary PDFs.”

Ingest with `scripts/ingest_corpus.py` after downloading PDFs into `docs/corpus/pdfs/`. Tag each chunk with `defect_classes` and `source_id` in metadata.

### Workmanship standards (public)

| Doc | Why | URL | Notes |
|-----|-----|-----|-------|
| NASA-STD-8739.6B | Agency workmanship / ESD implementation language | [PDF](https://standards.nasa.gov/sites/default/files/standards/NASA/B/0/nasa-std-87396b.pdf) · [record](https://standards.nasa.gov/standard/nasa/nasa-std-87396) | Current. Marked public on NASA NTSS. |
| NASA-STD-8739.1B Chg 2 | Conformal coating, staking, encapsulation, rework of polymeric applications | [PDF](https://standards.nasa.gov/sites/default/files/standards/NASA/B/2/nasa-std-87391B-Change-2.pdf) · [record](https://standards.nasa.gov/standard/nasa/nasa-std-87391) | Current. Public. |
| NASA-STD-8739.3 Chg 4 (cancelled) | Hand-solder accept/reject criteria, opens/shorts language | [EverySpec](https://everyspec.com/NASA/NASA-NASA-STD/NASA-STD-8739x3_CHG-4_33216/) | Cancelled 2011; superseded by **paid** IPC J-STD-001. Still a public technical document. Cite as historical. |
| ECSS-Q-ST-70-61C | High-reliability SMT and through-hole assembly, workmanship, accept/reject | [PDF](https://ecss.nl/wp-content/uploads/2022/04/ECSS-Q-ST-70-61C(8April2022).pdf) · [record](https://ecss.nl/standard/ecss-q-st-70-61c-high-reliability-assembly-for-surface-mount-and-through-hole-connections-8-april-2022/) | Current ESA standard, free download. Large (~hundreds of pages). Primary assembly RAG source. |

NASA-STD-8739.4 (crimp/harness) is public but **out of domain** for copper-pattern AOI — do not ingest unless the product expands to wire harnesses.

### Open literature (describe the six defects)

| Doc | Why | URL | License |
|-----|-----|-----|---------|
| DeepPCB paper | Defines the six classes the detector uses | [arXiv:1902.06197](https://arxiv.org/pdf/1902.06197) | arXiv license |
| PKU PCB dataset paper | Same classes on color boards; missing-hole definition | [arXiv:1901.08204](https://arxiv.org/pdf/1901.08204) | arXiv license |
| VisA / SPot-the-Difference paper | Assembled-board anomaly types | [arXiv:2207.14315](https://arxiv.org/pdf/2207.14315) | arXiv license |
| FPIC paper | Silkscreen / designator OCR | [arXiv:2202.08414](https://arxiv.org/pdf/2202.08414) | arXiv license |

### Wikipedia (CC BY-SA — keep attribution in chunk metadata)

| Article | Maps to |
|---------|---------|
| [Printed circuit board](https://en.wikipedia.org/wiki/Printed_circuit_board) | Board anatomy |
| [Printed circuit board manufacturing](https://en.wikipedia.org/wiki/Printed_circuit_board_manufacturing) | Bare-board opens/shorts, e-test, AOI |
| [Automated optical inspection](https://en.wikipedia.org/wiki/Automated_optical_inspection) | Inspection workflow |
| [Soldering](https://en.wikipedia.org/wiki/Soldering) | Joints vs copper defects |
| [Rework (electronics)](https://en.wikipedia.org/wiki/Rework_(electronics)) | Disposition / rework |
| [Conformal coating](https://en.wikipedia.org/wiki/Conformal_coating) | Coating defects vs copper |
| [Solder mask](https://en.wikipedia.org/wiki/Solder_mask) | Mask vs exposed copper |

### In-repo adapter pages (we write these)

Short markdown in `docs/corpus/` that **cite** the sources above and map each canonical class → inspection notes + typical disposition (scrap vs cut-and-etch vs jumper). These are not fake serial-number manuals; they are an index so RAG can join a DeepPCB class name to NASA/ECSS/Wikipedia passages.

The five existing `*_guide.txt` files are **CI stubs** for unit tests. Replace their taxonomy when the adapter pages land.

---

## Eval gold (not training)

Keep a JSONL manifest in git; keep images in `data/processed/eval/` (gitignored) except a tiny checked-in sample.

```json
{"image": "data/processed/eval/deeppcb/00041000_test.jpg", "template": "data/processed/eval/deeppcb/00041000_temp.jpg", "split": "deeppcb_test", "boxes": [{"bbox": [x1,y1,x2,y2], "cls": "open"}], "expected_doc_ids": ["arxiv-1902.06197", "wikipedia-pcb-manufacturing"], "designator": null}
```

| Split | Source | Use |
|-------|--------|-----|
| `deeppcb_test` | Official 500 DeepPCB test pairs | CV mAP / classification; template-diff A/B |
| `pku_holdout` | PKU images never used in training | Transfer F1 (expect drop) |
| `visa_pcb_gate` | VisA PCB1–4 | Anomaly AUROC only |
| `fpic_ocr` | FPIC crops with designator text | OCR exact-match |
| `rag_queries` | ~30 hand-written queries | Recall@5 / MRR |

---

## Citations (minimum)

When publishing results, cite Tang et al. 2019 (DeepPCB), Huang & Wei 2019 (PKU-Market-PCB), and any other set you actually ran. Wikipedia chunks must retain `license: CC-BY-SA` and `source_url` in metadata.
