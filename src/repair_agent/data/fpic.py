"""FPIC (FICS PCB Image Collection) helpers: designator crops for OCR gold.

FPIC ships S3A-style CSVs under ``ocr_annotation/`` (one per board image) with
columns such as ``Instance ID``, ``Vertices``, ``Text``, ``Class`` (Board /
Device), ``Logo``, ``Orientation`` (0-359) and ``Notes``; board photos are PNGs
under ``pcb_image/``. See arXiv:2202.08414, Tables 2-3.

Column names are matched case-insensitively and ``Vertices`` is parsed by
pulling every number out of the cell, so both ``[[x, y], ...]`` and nested
polygon lists work.

Gold rows (``gold_ocr.jsonl``)::

    {"image": ".../crops/<stem>_<id>.png", "designator": "R12",
     "source_image": "<board>.png", "split": "fpic_ocr", "subset": "dev"|"test",
     "orientation": 90, "bbox": [x1, y1, x2, y2], "text_class": "Board"}

``split`` is always ``fpic_ocr`` (the eval family); ``subset`` is the
dev/test carve. Subsets are assigned per *source image* so crops from one
photo never straddle dev and test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DESIGNATOR_RE = re.compile(r"^[RCULJQDW]\d{1,4}$")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
SPLIT_NAME = "fpic_ocr"

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class TextBox:
    csv_path: Path
    instance_id: str
    text: str
    vertices: list[tuple[float, float]]
    orientation: int | None = None
    text_class: str = ""
    logo: str = ""
    notes: str = ""
    source_image: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.vertices]
        ys = [p[1] for p in self.vertices]
        return int(min(xs)), int(min(ys)), int(np.ceil(max(xs))), int(np.ceil(max(ys)))


def normalize_label(text: str) -> str:
    return re.sub(r"\s+", "", text or "").upper()


def is_designator(text: str) -> bool:
    return bool(DESIGNATOR_RE.match(normalize_label(text)))


def parse_vertices(cell: str) -> list[tuple[float, float]]:
    """Parse an S3A ``Vertices`` cell into (x, y) points (all polygons merged)."""
    nums = [float(n) for n in _NUM.findall(cell or "")]
    if len(nums) < 4 or len(nums) % 2:
        return []
    return list(zip(nums[0::2], nums[1::2]))


def _col(row: dict[str, str], *names: str) -> str:
    lowered = {k.strip().lower(): v for k, v in row.items() if k is not None}
    for name in names:
        v = lowered.get(name.lower())
        if v is not None:
            return v.strip()
    return ""


def _parse_orientation(value: str) -> int | None:
    m = _NUM.search(value or "")
    return int(float(m.group(0))) % 360 if m else None


def ocr_annotation_csvs(raw_root: Path) -> list[Path]:
    """CSV files under any ``ocr_annotation*`` folder (falls back to ``*ocr*.csv``)."""
    dirs = [p for p in raw_root.rglob("*") if p.is_dir() and p.name.lower().startswith("ocr_annotation")]
    files: list[Path] = []
    for d in dirs:
        files.extend(sorted(d.rglob("*.csv")))
    if not files:
        files = sorted(p for p in raw_root.rglob("*.csv") if "ocr" in str(p).lower())
    return sorted(set(files))


def image_index(raw_root: Path) -> dict[str, Path]:
    """Map lower-case image stem -> path, preferring ``pcb_image`` folders."""
    index: dict[str, Path] = {}
    candidates = sorted(p for p in raw_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    candidates.sort(key=lambda p: 0 if "pcb_image" in str(p).lower() else 1)
    for p in candidates:
        if "color_checker" in str(p).lower() or "crops" in p.parts:
            continue
        index.setdefault(p.stem.lower(), p)
    return index


def iter_text_boxes(csv_path: Path) -> Iterator[TextBox]:
    with csv_path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            verts = parse_vertices(_col(row, "Vertices", "vertices", "polygon", "points"))
            if not verts:
                continue
            yield TextBox(
                csv_path=csv_path,
                instance_id=_col(row, "Instance ID", "instance_id", "id") or str(i),
                text=_col(row, "Text", "label"),
                vertices=verts,
                orientation=_parse_orientation(_col(row, "Orientation")),
                text_class=_col(row, "Class", "Designator Class"),
                logo=_col(row, "Logo"),
                notes=_col(row, "Notes"),
                source_image=_col(row, "Source Image Filename", "Image", "Filename") or None,
            )


def resolve_image(box: TextBox, index: dict[str, Path]) -> Path | None:
    keys = []
    if box.source_image:
        keys.append(Path(box.source_image).stem.lower())
    keys.append(box.csv_path.stem.lower())
    for key in keys:
        if key in index:
            return index[key]
    return None


def designator_boxes(raw_root: Path, *, board_only: bool = False) -> list[tuple[TextBox, Path]]:
    """All text boxes whose label is a designator, paired with their board image."""
    index = image_index(raw_root)
    out: list[tuple[TextBox, Path]] = []
    for csv_path in ocr_annotation_csvs(raw_root):
        for box in iter_text_boxes(csv_path):
            if not is_designator(box.text) or box.logo:
                continue
            if board_only and box.text_class and box.text_class.lower() != "board":
                continue
            img = resolve_image(box, index)
            if img is not None:
                out.append((box, img))
    return out


def assign_subsets(sources: Iterable[str], dev_frac: float = 0.2, seed: int = 0) -> dict[str, str]:
    """Deterministic per-source dev/test assignment (hash order, ~dev_frac dev)."""
    uniq = sorted(set(sources))
    ranked = sorted(uniq, key=lambda s: hashlib.sha1(f"{seed}:{s}".encode()).hexdigest())
    n_dev = max(1, round(len(ranked) * dev_frac)) if len(ranked) > 1 else 0
    return {s: ("dev" if i < n_dev else "test") for i, s in enumerate(ranked)}


def crop_box(img: np.ndarray, bbox: tuple[int, int, int, int], pad_frac: float = 0.15) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = int(round(pad_frac * max(2, min(x2 - x1, y2 - y1))))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return img[y1:y2, x1:x2]


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def export_fpic(
    raw_root: Path,
    out_dir: Path,
    *,
    dev_frac: float = 0.2,
    seed: int = 0,
    min_side: int = 4,
    board_only: bool = False,
) -> dict:
    """Write ``crops/*.png`` and ``gold_ocr.jsonl`` under ``out_dir``."""
    boxes = designator_boxes(raw_root, board_only=board_only)
    subsets = assign_subsets((img.name for _, img in boxes), dev_frac=dev_frac, seed=seed)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    gold_path = out_dir / "gold_ocr.jsonl"

    counts = {"dev": 0, "test": 0, "skipped_small": 0, "skipped_unreadable": 0}
    cache: dict[Path, np.ndarray | None] = {}
    rows: list[dict] = []
    for box, img_path in boxes:
        if img_path not in cache:
            cache.clear()  # boards are large; keep one decoded at a time
            cache[img_path] = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cache[img_path]
        if img is None:
            counts["skipped_unreadable"] += 1
            continue
        x1, y1, x2, y2 = box.bbox
        if min(x2 - x1, y2 - y1) < min_side:
            counts["skipped_small"] += 1
            continue
        crop = crop_box(img, box.bbox)
        if crop.size == 0:
            counts["skipped_small"] += 1
            continue
        name = f"{_safe(img_path.stem)}_{_safe(box.instance_id)}.png"
        crop_path = crops_dir / name
        cv2.imwrite(str(crop_path), crop)
        subset = subsets[img_path.name]
        counts[subset] += 1
        rows.append(
            {
                "image": str(crop_path.resolve()),
                "designator": normalize_label(box.text),
                "source_image": img_path.name,
                "split": SPLIT_NAME,
                "subset": subset,
                "orientation": box.orientation,
                "bbox": [x1, y1, x2, y2],
                "text_class": box.text_class,
                "notes": box.notes,
            }
        )
    with gold_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return {
        "gold": str(gold_path),
        "n": len(rows),
        "n_sources": len(subsets),
        "dev_sources": sum(1 for v in subsets.values() if v == "dev"),
        **counts,
    }
