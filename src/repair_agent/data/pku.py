"""PKU-Market-PCB helpers: VOC XML parsing and held-out gold JSONL export.

PKU-Market-PCB (Huang & Wei, arXiv:1901.08204) is **academic research only**.
It is a transfer / held-out set for this project and must never enter the YOLO
training loss. Only the original (non-rotated) images are used.

Two on-disk layouts are supported:

* Mirror layout (e.g. Hugging Face ``RobotHuman/PCB_defect``)::

      <root>/Open_circuit/open_circuit01.jpg
      <root>/Open_circuit/open_circuit01_annotation.xml

* Original lab zip layout::

      <root>/images/Open_circuit/01_open_circuit_01.jpg
      <root>/Annotations/Open_circuit/01_open_circuit_01.xml

Any path containing a ``rotation`` / ``rotated`` / ``augment`` component is
skipped so that augmented copies never count as extra held-out images.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from repair_agent.taxonomy import PKU_NAME_TO_CLASS

SPLIT_NAME = "pku_holdout"
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
_SKIP_PARTS = ("rotation", "rotated", "augment")

Sample = tuple[Path, Path]


def normalize_pku_name(name: str) -> str | None:
    """Map a PKU VOC ``<name>`` to a canonical class id, or None if unknown."""
    key = name.strip().lower()
    if key in PKU_NAME_TO_CLASS:
        return PKU_NAME_TO_CLASS[key]
    key = key.replace("-", "_")
    if key in PKU_NAME_TO_CLASS:
        return PKU_NAME_TO_CLASS[key]
    return PKU_NAME_TO_CLASS.get(key.replace("_", " "))


def parse_voc(xml_path: Path) -> dict:
    """Parse one VOC XML file.

    Returns ``{"width", "height", "boxes": [{"bbox": [x1,y1,x2,y2], "cls"}],
    "unknown": [names]}``. Boxes are integer pixel xyxy in the full image.
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0
    boxes: list[dict] = []
    unknown: list[str] = []
    for obj in root.iter("object"):
        name = obj.findtext("name", "")
        cls = normalize_pku_name(name)
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        if cls is None:
            unknown.append(name)
            continue
        x1, y1, x2, y2 = (
            int(round(float(bnd.findtext(tag, "0"))))
            for tag in ("xmin", "ymin", "xmax", "ymax")
        )
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({"bbox": [x1, y1, x2, y2], "cls": cls})
    return {"width": width, "height": height, "boxes": boxes, "unknown": unknown}


def _is_augmented(path: Path) -> bool:
    return any(s in part.lower() for part in path.parts for s in _SKIP_PARTS)


def find_annotation(image: Path, raw_root: Path) -> Path | None:
    stem = image.stem
    candidates = [
        image.with_name(f"{stem}_annotation.xml"),
        image.with_name(f"{stem}.xml"),
    ]
    # Original zip: images/<Class>/x.jpg -> Annotations/<Class>/x.xml
    try:
        rel = image.relative_to(raw_root)
    except ValueError:
        rel = None
    if rel is not None and rel.parts and rel.parts[0].lower() == "images":
        sub = Path(*rel.parts[1:]).with_suffix(".xml")
        candidates.append(raw_root / "Annotations" / sub)
    for path in candidates:
        if path.is_file():
            return path
    return None


def find_samples(raw_root: Path) -> list[Sample]:
    """Return sorted (image, xml) pairs for original PKU images under raw_root."""
    samples: list[Sample] = []
    for image in raw_root.rglob("*"):
        if image.suffix.lower() not in _IMAGE_SUFFIXES or not image.is_file():
            continue
        rel = image.relative_to(raw_root)
        if _is_augmented(rel) or any(p.startswith(".") for p in rel.parts):
            continue
        ann = find_annotation(image, raw_root)
        if ann is not None:
            samples.append((image, ann))
    samples.sort(key=lambda s: s[0].as_posix())
    return samples


def build_gold_rows(samples: list[Sample]) -> tuple[list[dict], dict]:
    """Turn samples into gold rows. Returns (rows, stats)."""
    rows: list[dict] = []
    seen: set[str] = set()
    stats: dict = {"images": 0, "boxes": 0, "per_class": {}, "unknown_names": {}}
    for image, ann in samples:
        parsed = parse_voc(ann)
        image_id = image.stem
        if image_id in seen:
            image_id = f"{image.parent.name}/{image.stem}"
        seen.add(image_id)
        rows.append(
            {
                "image_id": image_id,
                "image": str(image.resolve()),
                "split": SPLIT_NAME,
                "width": parsed["width"],
                "height": parsed["height"],
                "boxes": parsed["boxes"],
            }
        )
        stats["images"] += 1
        stats["boxes"] += len(parsed["boxes"])
        for box in parsed["boxes"]:
            stats["per_class"][box["cls"]] = stats["per_class"].get(box["cls"], 0) + 1
        for name in parsed["unknown"]:
            stats["unknown_names"][name] = stats["unknown_names"].get(name, 0) + 1
    return rows, stats


def write_gold(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return out_path
