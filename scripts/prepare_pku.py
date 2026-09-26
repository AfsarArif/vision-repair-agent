#!/usr/bin/env python3
"""Convert raw PKU-Market-PCB (VOC XML) into the transfer-eval gold JSONL.

PKU-Market-PCB is academic-research-only. Keep it under the gitignored
data/raw/pku_pcb/ and never add it to the YOLO training set.

Download (no login), original 693 images + VOC XML:
  python -c "from huggingface_hub import snapshot_download; \
snapshot_download('RobotHuman/PCB_defect', repo_type='dataset', local_dir='data/raw/pku_pcb')"

Writes:
  data/processed/pku/gold_holdout.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from repair_agent.data.pku import build_gold_rows, find_samples, write_gold  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "pku_pcb")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "pku" / "gold_holdout.jsonl",
    )
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"PKU-Market-PCB not found at {args.raw} (see this script's docstring)")
        return 1
    samples = find_samples(args.raw)
    if not samples:
        print(f"No PKU images with VOC XML annotations under {args.raw}")
        return 1
    rows, stats = build_gold_rows(samples)
    path = write_gold(rows, args.out)
    print(json.dumps(stats, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
