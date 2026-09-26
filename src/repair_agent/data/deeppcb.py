"""DeepPCB layout helpers: pair discovery, official split, YOLO labels."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image

from repair_agent.taxonomy import YOLO_CLASS_NAMES, class_to_yolo_index, deeppcb_id_to_class

Pair = tuple[Path, Path, Path]


def pcbdata_root(raw_root: Path) -> Path:
    nested = raw_root / "PCBData"
    return nested if nested.is_dir() else raw_root


def test_stem(test_path: Path) -> str:
    name = test_path.name
    if name.endswith("_test.jpg"):
        return name[: -len("_test.jpg")]
    return test_path.stem.replace("_test", "").replace("_temp", "")


def normalize_list_stem(token: str) -> str:
    """Map a trainval/test list path to the DeepPCB id stem (e.g. 00041000)."""
    name = Path(token.replace("\\", "/")).name
    stem = Path(name).stem
    if stem.endswith("_test"):
        return stem[: -len("_test")]
    if stem.endswith("_temp"):
        return stem[: -len("_temp")]
    return stem


def find_annotation(test: Path) -> Path | None:
    stem = test_stem(test)
    parent = test.parent
    candidates = [
        parent.parent / f"{parent.name}_not" / f"{stem}.txt",
        parent / f"{stem}.txt",
        test.with_name(f"{stem}.txt"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def find_template(test: Path) -> Path | None:
    stem = test_stem(test)
    for name in (f"{stem}_temp.jpg", f"{stem}_template.jpg"):
        path = test.with_name(name)
        if path.is_file():
            return path
    return None


def find_pairs(raw_root: Path) -> list[Pair]:
    """Return (test_img, template_img, annotation_txt) triples."""
    pairs: list[Pair] = []
    for test in raw_root.rglob("*_test.jpg"):
        template = find_template(test)
        ann = find_annotation(test)
        if template is not None and ann is not None:
            pairs.append((test, template, ann))
    pairs.sort(key=lambda p: p[0].as_posix())
    return pairs


def load_split_stems(raw_root: Path) -> tuple[set[str], set[str]] | None:
    """Return (train_stems, test_stems) from official PCBData lists if present."""
    base = pcbdata_root(raw_root)
    train_file = next((base / n for n in ("trainval.txt", "train.txt") if (base / n).is_file()), None)
    test_file = next((base / n for n in ("test.txt",) if (base / n).is_file()), None)
    if train_file is None or test_file is None:
        return None
    train_stems = stems_from_list(train_file)
    test_stems = stems_from_list(test_file)
    if not train_stems or not test_stems:
        return None
    return train_stems, test_stems


def stems_from_list(path: Path) -> set[str]:
    stems: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token:
            stems.add(normalize_list_stem(token))
    return stems


def parse_annotation(path: Path) -> list[dict]:
    boxes: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().replace(",", " ").split()
        if len(parts) < 5:
            continue
        try:
            x1, y1, x2, y2, type_id = (int(float(parts[i])) for i in range(5))
        except ValueError:
            continue
        if type_id == 0 or x2 <= x1 or y2 <= y1:
            continue
        try:
            cls = deeppcb_id_to_class(type_id)
        except ValueError:
            continue
        boxes.append({"bbox": [x1, y1, x2, y2], "cls": cls})
    return boxes


def to_yolo_line(box: dict, img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = (float(v) for v in box["bbox"])
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 1e-6), 1.0)
    h = min(max(h, 1e-6), 1.0)
    idx = class_to_yolo_index(box["cls"])
    return f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def split_pairs(
    pairs: list[Pair],
    seed: int = 42,
    raw_root: Path | None = None,
    val_fraction: float = 0.15,
) -> dict[str, list[Pair]]:
    """Official DeepPCB is 1000 trainval / 500 test. Val is carved from train only."""
    by_stem = {test_stem(test): (test, temp, ann) for test, temp, ann in pairs}
    official = load_split_stems(raw_root) if raw_root is not None else None

    if official:
        train_stems, test_stems = official
        overlap = train_stems & test_stems
        if overlap:
            train_stems = train_stems - overlap
        train_pool = [by_stem[s] for s in sorted(train_stems) if s in by_stem]
        test = [by_stem[s] for s in sorted(test_stems) if s in by_stem]
        leftover = len(by_stem) - len({test_stem(p[0]) for p in train_pool + test})
        if leftover:
            print(f"warning: {leftover} pairs not in official lists; excluded from all splits")
    elif len(pairs) < 20:
        n_test = max(1, len(pairs) // 5)
        rest = pairs[n_test:]
        n_val = max(1, len(rest) // 5) if len(rest) > 1 else 0
        return {"train": rest[n_val:], "val": rest[:n_val], "test": pairs[:n_test]}
    else:
        rng = random.Random(seed)
        n_test = min(500, max(1, round(len(pairs) * 500 / 1500)))
        ordered = list(pairs)
        rng.shuffle(ordered)
        test = ordered[:n_test]
        train_pool = ordered[n_test:]

    rng = random.Random(seed)
    shuffled_train = list(train_pool)
    rng.shuffle(shuffled_train)
    n_val = 0
    if shuffled_train:
        n_val = max(1, round(len(shuffled_train) * val_fraction))
        n_val = min(n_val, len(shuffled_train) - 1) if len(shuffled_train) > 1 else 0
    val = shuffled_train[:n_val]
    train = shuffled_train[n_val:]
    return {"train": train, "val": val, "test": test}


def yaml_text(processed: Path) -> str:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(YOLO_CLASS_NAMES))
    return (
        f"path: {processed.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(YOLO_CLASS_NAMES)}\n"
        f"names:\n{names}\n"
    )


def write_yaml(processed: Path) -> None:
    (processed / "deeppcb.yaml").write_text(yaml_text(processed), encoding="utf-8")


def image_size(path: Path, fallback: int = 640) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size
    except OSError:
        return fallback, fallback


def gold_row(stem: str, out: Path, split: str, boxes: list[dict]) -> dict:
    return {
        "image_id": stem,
        "image": str(out / "images" / split / f"{stem}.jpg"),
        "template": str(out / "templates" / split / f"{stem}_temp.jpg"),
        "split": f"deeppcb_{split}",
        "boxes": boxes,
    }


def write_gold(splits: dict[str, list[Pair]], out: Path, names: tuple[str, ...] = ("val", "test")) -> dict[str, Path]:
    """Write `gold_{split}.jsonl` for eval. Does not touch images, labels, or weights."""
    paths: dict[str, Path] = {}
    out.mkdir(parents=True, exist_ok=True)
    for split in names:
        path = out / f"gold_{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for test, _template, ann in splits.get(split, []):
                row = gold_row(test_stem(test), out, split, parse_annotation(ann))
                handle.write(json.dumps(row) + "\n")
        paths[split] = path
    return paths


def export_splits(
    splits: dict[str, list[Pair]],
    out: Path,
    fallback_size: int = 640,
) -> Path:
    """Write YOLO image/label folders, templates, yaml, and val/test gold JSONL.

    Wipes `out` first (including any weights copied there). Use `write_gold`
    to refresh gold files only.
    """
    if out.exists():
        shutil.rmtree(out)

    for split, items in splits.items():
        img_dir = out / "images" / split
        label_dir = out / "labels" / split
        temp_dir = out / "templates" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        for test, template, ann in items:
            stem = test_stem(test)
            dest_img = img_dir / f"{stem}.jpg"
            dest_temp = temp_dir / f"{stem}_temp.jpg"
            shutil.copy2(test, dest_img)
            shutil.copy2(template, dest_temp)
            img_w, img_h = image_size(test, fallback_size)
            boxes = parse_annotation(ann)
            yolo_lines = [to_yolo_line(b, img_w, img_h) for b in boxes]
            (label_dir / f"{stem}.txt").write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else "")
            )

    write_yaml(out)
    return write_gold(splits, out)["test"]
