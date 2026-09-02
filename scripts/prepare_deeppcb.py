#!/usr/bin/env python3
"""Convert DeepPCB raw pairs into YOLO folders and eval gold JSONL.

Expects a clone at data/raw/deeppcb (see scripts/download_public_data.py).
Writes data/processed/deeppcb/{images,labels}/{train,val,test} and gold_test.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from repair_agent.taxonomy import YOLO_CLASS_NAMES, deeppcb_id_to_class, class_to_yolo_index


def find_pairs(raw_root: Path) -> list[tuple[Path, Path, Path]]:
    """Return (test_img, template_img, annotation_txt) triples."""
    pairs: list[tuple[Path, Path, Path]] = []
    for test in raw_root.rglob("*_test.jpg"):
        stem = test.name[: -len("_test.jpg")]
        template = test.with_name(f"{stem}_temp.jpg")
        if not template.exists():
            template = test.with_name(f"{stem}_template.jpg")
        ann = test.with_name(f"{stem}.txt")
        if not ann.exists():
            ann = test.with_suffix(".txt")
        if template.exists() and ann.exists():
            pairs.append((test, template, ann))
    pairs.sort(key=lambda p: p[0].as_posix())
    return pairs


def parse_annotation(path: Path) -> list[dict]:
    boxes: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().replace(",", " ").split()
        if len(parts) < 5:
            continue
        x1, y1, x2, y2, type_id = (int(float(parts[i])) for i in range(5))
        if type_id == 0 or x2 <= x1 or y2 <= y1:
            continue
        cls = deeppcb_id_to_class(type_id)
        boxes.append({"bbox": [x1, y1, x2, y2], "cls": cls})
    return boxes


def to_yolo_line(box: dict, img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = box["bbox"]
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    idx = class_to_yolo_index(box["cls"])
    return f" {idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}".strip()


def split_pairs(
    pairs: list[tuple[Path, Path, Path]], seed: int = 42
) -> dict[str, list[tuple[Path, Path, Path]]]:
    """Official DeepPCB is 1000 train / 500 test. Val is carved from train only."""
    if len(pairs) < 20:
        n_test = max(1, len(pairs) // 5)
        return {"train": pairs[n_test:], "val": [], "test": pairs[:n_test]}

    rng = random.Random(seed)
    # Paper: 1000 train, remaining test. If count != 1500, keep the same ratio.
    n_test = min(500, max(1, round(len(pairs) * 500 / 1500)))
    # Deterministic: last n_test after sort is brittle; shuffle a copy with seed
    # then take test from the end so re-runs match.
    ordered = list(pairs)
    rng.shuffle(ordered)
    test = ordered[:n_test]
    rest = ordered[n_test:]
    n_val = max(1, round(len(rest) * 0.15))
    val = rest[:n_val]
    train = rest[n_val:]
    return {"train": train, "val": val, "test": test}


def write_yaml(processed: Path) -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(YOLO_CLASS_NAMES))
    text = (
        f"path: {processed.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n{names}\n"
    )
    (processed / "deeppcb.yaml").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "deeppcb")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "deeppcb")
    parser.add_argument("--img-size", type=int, default=640)
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"DeepPCB not found at {args.raw}. Run scripts/download_public_data.py --deeppcb")
        return 1

    pairs = find_pairs(args.raw)
    if not pairs:
        print(f"No *_test.jpg pairs under {args.raw}")
        return 1

    splits = split_pairs(pairs)
    if args.out.exists():
        shutil.rmtree(args.out)

    gold_test: list[dict] = []
    for split, items in splits.items():
        img_dir = args.out / "images" / split
        label_dir = args.out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for test, template, ann in items:
            stem = test.name[: -len("_test.jpg")]
            dest_img = img_dir / f"{stem}.jpg"
            shutil.copy2(test, dest_img)
            shutil.copy2(template, img_dir / f"{stem}_temp.jpg")
            boxes = parse_annotation(ann)
            yolo_lines = [to_yolo_line(b, args.img_size, args.img_size) for b in boxes]
            (label_dir / f"{stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
            if split == "test":
                gold_test.append(
                    {
                        "image_id": stem,
                        "image": str(dest_img),
                        "template": str(img_dir / f"{stem}_temp.jpg"),
                        "split": "deeppcb_test",
                        "boxes": boxes,
                    }
                )

    write_yaml(args.out)
    gold_path = args.out / "gold_test.jsonl"
    with gold_path.open("w", encoding="utf-8") as handle:
        for row in gold_test:
            handle.write(json.dumps(row) + "\n")

    print(
        f"pairs={len(pairs)} train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])} gold={gold_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
