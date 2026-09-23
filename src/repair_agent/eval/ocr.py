"""Designator OCR eval (Phase D).

Gate: exact-match >= 0.70 on FPIC designator crops (``split == "fpic_ocr"``,
``subset == "test"``). Never score OCR on DeepPCB: it has no silkscreen.

Also provides PIL-rendered synthetic silkscreen crops for unit tests and a CI
smoke eval. Synthetic numbers are **not** the gate.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from repair_agent.eval.metrics import exact_match

GATE = 0.70
SYNTHETIC_SPLIT = "synthetic_ocr"

# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


def _tesseract_reader() -> Callable[[np.ndarray], tuple[str | None, float]]:
    from repair_agent.tools.ocr_tools import read_designator

    return read_designator


def _easyocr_reader() -> Callable[[np.ndarray], tuple[str | None, float]]:
    try:
        import easyocr  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("engine='easyocr' needs `pip install easyocr`") from exc
    from repair_agent.tools.ocr_tools import DEFAULT_ROTATIONS, _rotate, normalize_designator

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(img: np.ndarray) -> tuple[str | None, float]:
        scores: dict[str, float] = {}
        for angle in DEFAULT_ROTATIONS:
            rot = _rotate(img, angle)
            h, w = rot.shape[:2]
            scale = max(1.0, 48 / max(1, min(h, w)))
            if scale > 1:
                rot = cv2.resize(rot, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for _box, text, conf in reader.readtext(rot, allowlist="RCULJQDW0123456789OISZB"):
                cand = normalize_designator(text)
                if cand:
                    scores[cand] = max(scores.get(cand, 0.0), float(conf))
        if not scores:
            return None, 0.0
        best = max(scores, key=scores.get)
        return best, round(scores[best], 4)

    return read


ENGINES: dict[str, Callable[[], Callable[[np.ndarray], tuple[str | None, float]]]] = {
    "tesseract": _tesseract_reader,
    "easyocr": _easyocr_reader,
}

# ---------------------------------------------------------------------------
# Gold IO
# ---------------------------------------------------------------------------


def load_gold(gold_path: Path, split: str | None = "test") -> list[dict]:
    """Rows from a gold JSONL, filtered by ``split``.

    ``split`` matches either the row's ``subset`` (``dev`` / ``test``) or its
    ``split`` family (``fpic_ocr``); ``None`` keeps every row.
    """
    rows = []
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if split is not None and split not in (row.get("subset"), row.get("split")):
            continue
        rows.append(row)
    return rows


def _resolve(image: str, gold_path: Path) -> Path:
    p = Path(image)
    if p.is_absolute() or p.exists():
        return p
    return gold_path.parent / p


def eval_ocr(
    gold_path: Path,
    split: str | None = "test",
    engine: str = "tesseract",
    *,
    limit: int | None = None,
    return_rows: bool = False,
) -> dict:
    """Designator exact-match over a gold JSONL.

    Returns ``{"stage": "ocr", "n", "exact_match", "engine", "split", ...}``.
    """
    gold_path = Path(gold_path)
    if engine not in ENGINES:
        raise ValueError(f"Unknown OCR engine {engine!r}; choose from {sorted(ENGINES)}")
    rows = load_gold(gold_path, split)
    if limit:
        rows = rows[:limit]
    families = {r.get("split") for r in rows}
    if any(f and "deeppcb" in str(f).lower() for f in families):
        raise ValueError("Refusing to score OCR on DeepPCB rows (no silkscreen).")

    read = ENGINES[engine]()
    n = hits = no_read = 0
    per_row = []
    t0 = time.perf_counter()
    for row in rows:
        img = cv2.imread(str(_resolve(row["image"], gold_path)), cv2.IMREAD_COLOR)
        if img is None:
            pred, conf = None, 0.0
        else:
            pred, conf = read(img)
        ok = exact_match(pred, row.get("designator"))
        hits += int(ok)
        no_read += int(pred is None)
        n += 1
        if return_rows:
            per_row.append({"image": row["image"], "gold": row.get("designator"), "pred": pred, "conf": conf, "ok": ok})
    elapsed = time.perf_counter() - t0
    is_fpic = SYNTHETIC_SPLIT not in families
    report = {
        "stage": "ocr",
        "n": n,
        "exact_match": round(hits / n, 4) if n else 0.0,
        "no_read_rate": round(no_read / n, 4) if n else 0.0,
        "engine": engine,
        "split": split,
        "gold": str(gold_path),
        "dataset": "fpic" if is_fpic else "synthetic",
        "gate": GATE,
        "passes_gate": (hits / n >= GATE) if (n and is_fpic) else None,
        "sec_per_crop": round(elapsed / n, 4) if n else 0.0,
        "note": (
            "FPIC designator gold. Do not score DeepPCB."
            if is_fpic
            else "SYNTHETIC smoke eval - NOT the Phase D gate."
        ),
    }
    if return_rows:
        report["rows"] = per_row
    return report


# ---------------------------------------------------------------------------
# Synthetic silkscreen crops (tests / smoke eval only)
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Narrow.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

BOARD_COLORS = [(20, 90, 30), (15, 40, 110), (120, 20, 20), (15, 15, 15), (30, 60, 20)]
INK_COLORS = [(240, 240, 240), (230, 230, 200), (250, 250, 250)]


def _font(size: int, rng: random.Random) -> ImageFont.ImageFont:
    paths = [p for p in _FONT_CANDIDATES if Path(p).exists()]
    if paths:
        try:
            return ImageFont.truetype(rng.choice(paths), size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def random_designator(rng: random.Random) -> str:
    prefix = rng.choice("RCULJQDW")
    digits = rng.choice([1, 1, 2, 2, 2, 3, 3, 4])
    first = str(rng.randint(1, 9))
    return prefix + first + "".join(str(rng.randint(0, 9)) for _ in range(digits - 1))


def render_designator_crop(
    text: str,
    *,
    font_size: int = 16,
    rotation: int = 0,
    board: tuple[int, int, int] = (20, 90, 30),
    ink: tuple[int, int, int] = (240, 240, 240),
    blur: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Render light-on-dark silkscreen text as a BGR crop (like a detector box)."""
    rng = random.Random(seed)
    font = _font(font_size, rng)
    left, top, right, bottom = font.getbbox(text)
    tw, th = right - left, bottom - top
    pad = max(2, font_size // 5)
    img = Image.new("RGB", (tw + 2 * pad, th + 2 * pad), board)
    ImageDraw.Draw(img).text((pad - left, pad - top), text, fill=ink, font=font)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img)[:, :, ::-1].copy()  # RGB -> BGR
    if noise:
        nrng = np.random.default_rng(seed)
        arr = np.clip(arr.astype(np.float32) + nrng.normal(0, noise, arr.shape), 0, 255).astype(np.uint8)
    rotation %= 360
    if rotation == 90:
        arr = cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)  # text reads bottom-to-top
    elif rotation == 180:
        arr = cv2.rotate(arr, cv2.ROTATE_180)
    elif rotation == 270:
        arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
    return arr


def write_synthetic_gold(out_dir: Path, n: int = 100, seed: int = 0, subset: str = "dev") -> Path:
    """Write ``n`` synthetic crops + ``gold_ocr.jsonl`` (split=synthetic_ocr)."""
    rng = random.Random(seed)
    crops = out_dir / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    gold = out_dir / "gold_ocr.jsonl"
    with gold.open("w", encoding="utf-8") as fh:
        for i in range(n):
            text = random_designator(rng)
            crop = render_designator_crop(
                text,
                font_size=rng.randint(10, 24),
                rotation=rng.choice([0, 0, 0, 90, 270, 180]),
                board=rng.choice(BOARD_COLORS),
                ink=rng.choice(INK_COLORS),
                blur=rng.choice([0.0, 0.4, 0.7]),
                noise=rng.choice([0.0, 6.0, 12.0]),
                seed=seed * 100_003 + i,
            )
            path = crops / f"syn_{subset}_{i:04d}.png"
            cv2.imwrite(str(path), crop)
            fh.write(
                json.dumps(
                    {
                        "image": str(path.resolve()),
                        "designator": text,
                        "source_image": f"synthetic_{seed}",
                        "split": SYNTHETIC_SPLIT,
                        "subset": subset,
                    }
                )
                + "\n"
            )
    return gold
