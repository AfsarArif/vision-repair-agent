#!/usr/bin/env python3
"""Phase B: train a nano YOLO detector on prepared DeepPCB data.

Learning setup (also documented in docs/BUILD.md):
- Split: DeepPCB train/val carved from the 1000; official-style test held out
- Init: COCO-pretrained YOLOv8n
- No PKU/VisA images in the loss
- Augment: flip/translate only (binary copper)
- Select checkpoint by val mAP@0.5; report test once via evals/run_eval.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = ROOT / "data" / "processed" / "deeppcb" / "deeppcb.yaml"
DEFAULT_WEIGHTS_OUT = ROOT / "data" / "processed" / "deeppcb" / "weights" / "best.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--run", action="store_true", help="actually call Ultralytics (needs extra 'train')")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Missing {args.data}. Run download_public_data.py --deeppcb and prepare_deeppcb.py")
        return 1

    cmd_hint = (
        f"yolo detect train data={args.data} model=yolov8n.pt epochs={args.epochs} "
        f"imgsz={args.imgsz} hsv_h=0 hsv_s=0 hsv_v=0 mosaic=0.0 fliplr=0.5 flipud=0.5"
    )
    print("Phase B training plan")
    print(f"  data yaml : {args.data}")
    print(f"  init      : yolov8n.pt (COCO)")
    print(f"  epochs    : {args.epochs}")
    print(f"  selection : best val mAP@0.5")
    print(f"  then      : poetry run python evals/run_eval.py --stage cv --gold data/processed/deeppcb/gold_test.jsonl")
    print(f"  CLI       : {cmd_hint}")

    if not args.run:
        print("Dry run. Pass --run to start training.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. poetry install --extras train")
        return 1

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.0,
        fliplr=0.5,
        flipud=0.5,
        patience=20,
        project=str(ROOT / "data" / "processed" / "deeppcb"),
        name="train",
        exist_ok=True,
    )
    print(f"Set YOLO_WEIGHTS to the checkpoint under {ROOT / 'data' / 'processed' / 'deeppcb' / 'train' / 'weights'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
