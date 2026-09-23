#!/usr/bin/env python3
"""Build FPIC designator OCR gold: crops + gold_ocr.jsonl.

FPIC requires a (free) PhysicalDB / Trust-Hub account; this script does not
download it. Put the unzipped folders under data/raw/fpic, e.g.:

  data/raw/fpic/pcb_image/*.png
  data/raw/fpic/ocr_annotation/*.csv
  (smd_annotation/, metadata/, color_checker/ are optional)

Writes:
  data/processed/fpic/crops/*.png
  data/processed/fpic/gold_ocr.jsonl   # split="fpic_ocr", subset="dev"|"test"

Subsets are per source image (~20% dev for tuning, the rest held-out test).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from repair_agent.data.fpic import export_fpic, ocr_annotation_csvs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "fpic")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "fpic")
    parser.add_argument("--dev-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--board-only", action="store_true", help="Skip text printed on devices")
    args = parser.parse_args()

    if not args.raw.exists() or not ocr_annotation_csvs(args.raw):
        print(
            f"FPIC OCR annotations not found under {args.raw}.\n"
            "Register at https://physicaldb.ece.ufl.edu/index.php/fics-pcb-image-collection-fpic/ "
            "(or https://www.trust-hub.org/#/data/pcb-images), download pcb_image.zip and "
            "ocr_annotation.zip, and unzip them into that folder."
        )
        return 1

    summary = export_fpic(
        args.raw, args.out, dev_frac=args.dev_frac, seed=args.seed, board_only=args.board_only
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
