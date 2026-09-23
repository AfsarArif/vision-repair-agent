"""OCR utilities: serial numbers on labels and reference designators on silkscreen."""

import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from repair_agent.config import settings

# Reference designators on PCB silkscreen (Phase D / self-correct).
DESIGNATOR_PATTERN = re.compile(r"\b([RCULJQDW]\d{1,4})\b", re.IGNORECASE)

# Legacy serial patterns kept for CI fixtures.
SERIAL_PATTERNS = [
    re.compile(r"S/?N[:\s]*([A-Z0-9]*\d[A-Z0-9\-]*)", re.IGNORECASE),
    re.compile(r"Serial[:\s]*([A-Z0-9]*\d[A-Z0-9\-]*)", re.IGNORECASE),
    re.compile(r"[A-Z]{2,4}-[A-Z0-9]{4,10}", re.IGNORECASE),
    re.compile(r"[\d]{3,4}-[A-Z0-9]{4,8}", re.IGNORECASE),
]

TESSERACT_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-:"


def _configure_tesseract() -> None:
    cmd = settings.TESSERACT_CMD
    if cmd and Path(cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = cmd
        return
    if not shutil.which(pytesseract.pytesseract.tesseract_cmd):
        for candidate in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"):
            if Path(candidate).exists():
                pytesseract.pytesseract.tesseract_cmd = candidate
                return


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """Preprocess image for optimal OCR performance.

    Steps:
    1. Upscale 2x with cubic interpolation
    2. Convert to grayscale
    3. Sharpen with kernel
    4. Otsu binarization
    """
    # Upscale to improve OCR on small text
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)

    # Binarize
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def extract_text(img: np.ndarray) -> str:
    """Run pytesseract OCR on preprocessed image."""
    _configure_tesseract()
    try:
        pil_img = Image.fromarray(img)
        text = pytesseract.image_to_string(
            pil_img,
            config=f"--psm 6 --oem 3 -c tessedit_char_whitelist={TESSERACT_WHITELIST}",
        )
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        return ""


def find_designator(raw_text: str) -> str | None:
    """Return the first PCB reference designator (R12, C3, U7, …)."""
    match = DESIGNATOR_PATTERN.search(raw_text)
    if not match:
        return None
    return match.group(1).upper()


def find_serial_number(raw_text: str) -> str | None:
    """Search raw OCR text for known serial number patterns."""
    for pattern in SERIAL_PATTERNS:
        match = pattern.search(raw_text)
        if match:
            # Return group(1) if capturing group exists, else group(0)
            try:
                return match.group(1).strip() if match.lastindex else match.group(0).strip()
            except (IndexError, AttributeError):
                return match.group(0).strip()
    return None


def estimate_ocr_confidence(img: np.ndarray) -> float:
    """Estimate OCR confidence from pytesseract word-level confidences."""
    _configure_tesseract()
    try:
        pil_img = Image.fromarray(img)
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        return 0.0
    confidences = [int(c) for c in data["conf"] if int(c) > 0]
    if not confidences:
        return 0.0
    return round(sum(confidences) / len(confidences) / 100, 4)


def decode_and_preprocess(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes and run the full OCR preprocessing pipeline."""
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes for OCR.")
    return preprocess_for_ocr(img)


# ---------------------------------------------------------------------------
# Phase D: designator crops (FPIC silkscreen)
# ---------------------------------------------------------------------------

DESIGNATOR_PREFIXES = "RCULJQDW"
DESIGNATOR_FULL = re.compile(r"^[RCULJQDW]\d{1,4}$")
# Letters Tesseract may emit for digits, plus the valid prefixes and digits.
DESIGNATOR_WHITELIST = "RCULJQDW0123456789OISZBGTl|"

# Common OCR confusions inside the digit part of a designator.
_DIGIT_FIXES = str.maketrans(
    {"O": "0", "o": "0", "D": "0", "Q": "0", "U": "0",
     "I": "1", "l": "1", "|": "1", "T": "1", "L": "1", "J": "1",
     "S": "5", "s": "5", "Z": "2", "z": "2", "B": "8", "G": "6"}
)
# Confusions in the leading prefix letter. Empty by default: on synthetic dev,
# mapping e.g. 0->D mostly turned upside-down reads into confident wrong
# answers (0.920 without vs 0.907 with). Re-tune on FPIC dev.
_PREFIX_FIXES: dict[str, str] = {}

DEFAULT_ROTATIONS: tuple[int, ...] = (0, 90, 270, 180)
DEFAULT_PSMS: tuple[int, ...] = (7,)  # psm 7 alone 0.927 vs 7+8 0.907 on synthetic dev
TARGET_TEXT_HEIGHT = 48  # px after upscaling; Tesseract likes ~30-50 px glyphs


def normalize_designator(raw: str) -> str | None:
    """Map a raw OCR token to a canonical designator (``R12``) or ``None``.

    Strips whitespace/punctuation, upper-cases the prefix, and applies the
    usual confusions in the digit part (O->0, I/l->1, S->5, ...).
    """
    token = re.sub(r"[^A-Za-z0-9|(\[]", "", raw or "")
    if len(token) < 2:
        return None
    head, tail = token[0], token[1:]
    head = _PREFIX_FIXES.get(head, head).upper()
    if head not in DESIGNATOR_PREFIXES:
        return None
    tail = tail.translate(_DIGIT_FIXES)
    cand = head + tail
    return cand if DESIGNATOR_FULL.match(cand) else None


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _rotate(img: np.ndarray, angle: int) -> np.ndarray:
    angle %= 360
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def designator_variants(img: np.ndarray, rotations: tuple[int, ...] = DEFAULT_ROTATIONS):
    """Yield ``(name, binary)`` preprocess variants for a designator crop.

    Each variant is dark text on a white background (what Tesseract expects),
    upscaled so the short side is ~``TARGET_TEXT_HEIGHT`` px, padded with a
    white border. Silkscreen is usually light-on-dark, so the Otsu result is
    flipped whenever the border is mostly dark; the opposite polarity is also
    tried as a fallback.
    """
    if img is None or img.size == 0:
        return
    gray = _to_gray(img)
    h, w = gray.shape[:2]
    scale = max(1.0, TARGET_TEXT_HEIGHT / max(1, min(h, w)))
    scale = min(scale, 8.0)
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
    # Want background (border) white.
    primary = binary if border.mean() >= 127 else cv2.bitwise_not(binary)
    secondary = cv2.bitwise_not(primary)
    pad = max(8, int(0.25 * min(primary.shape[:2])))
    for angle in rotations:
        for pol, b in (("pos", primary), ("neg", secondary)):
            rot = _rotate(b, angle)
            rot = cv2.copyMakeBorder(rot, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
            yield f"r{angle}_{pol}", rot


def _tesseract_tokens(img: np.ndarray, psm: int) -> list[tuple[str, float]]:
    try:
        data = pytesseract.image_to_data(
            Image.fromarray(img),
            config=f"--psm {psm} --oem 3 -c tessedit_char_whitelist={DESIGNATOR_WHITELIST}",
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError:
        raise
    except Exception:  # noqa: BLE001 - TesseractError, lost temp file, ENOSPC, ...
        return []  # one bad variant (or a lost temp file) should not sink the crop
    out: list[tuple[str, float]] = []
    for text, conf in zip(data["text"], data["conf"]):
        text = (text or "").strip()
        try:
            c = float(conf)
        except (TypeError, ValueError):
            c = -1.0
        if text and c >= 0:
            out.append((text, c / 100.0))
    # Also offer the whole line joined (psm 7 can split "R 12").
    if len(out) > 1:
        joined = "".join(t for t, _ in out)
        out.append((joined, min(c for _, c in out)))
    return out


def read_designator(
    img: np.ndarray,
    *,
    rotations: tuple[int, ...] = DEFAULT_ROTATIONS,
    psms: tuple[int, ...] = DEFAULT_PSMS,
    early_exit: float = 0.90,
) -> tuple[str | None, float]:
    """Read one reference designator from a silkscreen crop.

    Tries rotations x polarities x Tesseract page-segmentation modes, keeps
    tokens that normalize to ``[RCULJQDW]\\d{1,4}``, and votes by summed
    confidence. Returns ``(designator, confidence in [0, 1])`` or
    ``(None, 0.0)``. Stops early once a variant reaches ``early_exit``.
    """
    _configure_tesseract()
    scores: dict[str, float] = {}
    best_conf: dict[str, float] = {}
    try:
        for _name, variant in designator_variants(img, rotations):
            for psm in psms:
                for text, conf in _tesseract_tokens(variant, psm):
                    cand = normalize_designator(text)
                    if cand is None:
                        continue
                    scores[cand] = scores.get(cand, 0.0) + conf
                    best_conf[cand] = max(best_conf.get(cand, 0.0), conf)
                    if conf >= early_exit:
                        return cand, round(conf, 4)
    except pytesseract.TesseractNotFoundError:
        return None, 0.0
    if not scores:
        return None, 0.0
    winner = max(scores, key=lambda k: (scores[k], best_conf[k]))
    return winner, round(best_conf[winner], 4)


def read_designator_bytes(image_bytes: bytes) -> tuple[str | None, float]:
    """Decode image bytes and run :func:`read_designator`."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes for designator OCR.")
    return read_designator(img)


def tesseract_available() -> bool:
    """True when a Tesseract binary can be invoked."""
    _configure_tesseract()
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False
