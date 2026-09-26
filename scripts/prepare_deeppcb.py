#!/usr/bin/env python3
"""Convert DeepPCB raw pairs into YOLO folders and eval gold JSONL.

Expects a clone at data/raw/deeppcb (see scripts/download_public_data.py).
Writes:
  data/processed/deeppcb/images/{train,val,test}/   # test images only (no templates)
  data/processed/deeppcb/labels/{train,val,test}/
  data/processed/deeppcb/templates/{train,val,test}/
  data/processed/deeppcb/gold_test.jsonl
  data/processed/deeppcb/deeppcb.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from repair_agent.data.deeppcb import export_splits, find_pairs, split_pairs, write_gold  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "deeppcb")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "deeppcb")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="Only rewrite gold_val/gold_test.jsonl (keeps images, labels, and trained weights)",
    )
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"DeepPCB not found at {args.raw}. Run scripts/download_public_data.py --deeppcb")
        return 1

    pairs = find_pairs(args.raw)
    if not pairs:
        print(f"No *_test.jpg pairs with templates and annotations under {args.raw}")
        return 1

    splits = split_pairs(pairs, raw_root=args.raw)
    if not splits["val"] and splits["train"]:
        splits["val"] = [splits["train"][0]]

    if args.gold_only:
        paths = write_gold(splits, args.out)
        print(f"val={len(splits['val'])} test={len(splits['test'])} gold={sorted(map(str, paths.values()))}")
        return 0

    gold_path = export_splits(splits, args.out, fallback_size=args.img_size)
    n_train_temps = len(list((args.out / "images" / "train").glob("*_temp.jpg")))
    print(
        f"pairs={len(pairs)} train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])} gold={gold_path}"
    )
    if n_train_temps:
        print(f"error: {n_train_temps} templates leaked into images/train", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
