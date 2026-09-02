#!/usr/bin/env python3
"""Phase B: train a nano YOLO detector on prepared DeepPCB data.

Learning setup (also documented in docs/BUILD.md):
- Split: DeepPCB train/val carved from the 1000; official-style test held out
- Init: COCO-pretrained YOLOv8n
- No PKU/VisA images in the loss
- Augment: flip/translate only (binary copper)
- Select checkpoint by val mAP@0.5; report test once at the end
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = ROOT / "data" / "processed" / "deeppcb" / "deeppcb.yaml"
STABLE_WEIGHTS = ROOT / "data" / "processed" / "deeppcb" / "weights" / "best.pt"


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def default_batch(device: str) -> int:
    if device == "cuda":
        return 16
    if device == "mps":
        return 8
    return 4


def _serialize_metrics(raw: object) -> dict:
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
        else:
            out[str(key)] = str(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda, mps, cpu, or omit for auto")
    parser.add_argument(
        "--freeze",
        type=int,
        default=3,
        help="Freeze backbone for this many of --epochs, then unfreeze (0=off)",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--run", action="store_true", help="actually call Ultralytics (needs extra 'train')")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Missing {args.data}. Run download_public_data.py --deeppcb and prepare_deeppcb.py")
        return 1

    device = args.device or detect_device()
    batch = args.batch if args.batch is not None else default_batch(device)
    workers = args.workers if args.workers is not None else (0 if device == "mps" else 4)
    project = ROOT / "data" / "processed" / "deeppcb"
    freeze_epochs = min(max(args.freeze, 0), args.epochs)
    rest_epochs = args.epochs - freeze_epochs

    print("Phase B training plan")
    print(f"  data yaml : {args.data}")
    print("  init      : yolov8n.pt (COCO)")
    print(f"  epochs    : {args.epochs} (freeze {freeze_epochs}, then {rest_epochs})")
    print(f"  device    : {device}")
    print(f"  batch     : {batch} workers={workers}")
    print("  selection : best val mAP@0.5")
    print("  then      : poetry run python evals/run_eval.py --stage cv --backend yolo")

    if not args.run:
        print("Dry run. Pass --run to start training.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. poetry install --extras train")
        return 1

    def train_kwargs(epochs: int, freeze: int | None, name: str) -> dict:
        kwargs = {
            "data": str(args.data),
            "epochs": epochs,
            "imgsz": args.imgsz,
            "batch": batch,
            "device": device,
            "workers": workers,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "fliplr": 0.5,
            "flipud": 0.5,
            "translate": 0.1,
            "scale": 0.2,
            "patience": 20,
            "optimizer": "AdamW",
            "cos_lr": True,
            "project": str(project),
            "name": name,
            "exist_ok": True,
            "pretrained": True,
            "amp": device == "cuda",
        }
        if freeze is not None:
            kwargs["freeze"] = freeze
        return kwargs

    model = YOLO("yolov8n.pt")
    results = None
    if freeze_epochs > 0:
        print(f"stage 1: freeze backbone for {freeze_epochs} epochs")
        model.train(**train_kwargs(freeze_epochs, freeze=10, name="train_frozen"))
        frozen_last = project / "train_frozen" / "weights" / "last.pt"
        if not frozen_last.is_file():
            print(f"missing frozen checkpoint {frozen_last}")
            return 1
        model = YOLO(str(frozen_last))

    if rest_epochs > 0:
        print(f"stage 2: unfreeze all for {rest_epochs} epochs")
        results = model.train(**train_kwargs(rest_epochs, freeze=0, name="train"))
    else:
        results = None

    run_dir = project / "train" if rest_epochs > 0 else project / "train_frozen"
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    src = best if best.is_file() else last
    if not src.is_file():
        print("Training finished but no weights were written.")
        return 1

    STABLE_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, STABLE_WEIGHTS)
    print(f"copied {src} -> {STABLE_WEIGHTS}")

    val_metrics = {}
    if results is not None:
        val_metrics = _serialize_metrics(getattr(results, "results_dict", None))

    test_metrics: dict = {}
    if not args.skip_test_eval:
        print("reporting official test split once (not used for checkpoint selection)")
        tester = YOLO(str(STABLE_WEIGHTS))
        test_res = tester.val(
            data=str(args.data),
            split="test",
            imgsz=args.imgsz,
            device=device,
            workers=workers,
            plots=False,
            verbose=False,
        )
        test_metrics = _serialize_metrics(getattr(test_res, "results_dict", None))

    summary = {
        "weights": str(STABLE_WEIGHTS),
        "device": device,
        "epochs": args.epochs,
        "freeze_epochs": freeze_epochs,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "note": "Set CV_BACKEND=yolo and YOLO_WEIGHTS to this path. Do not retune on the test set.",
    }
    summary_path = ROOT / "evals" / "results" / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Set YOLO_WEIGHTS={STABLE_WEIGHTS}")
    print(f"wrote {summary_path}")
    if test_metrics:
        print(f"test metrics: {json.dumps(test_metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
