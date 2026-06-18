"""OCR utilities for serial number extraction from hardware labels."""

import re
import io

import cv2
import numpy as np
import pytesseract
from PIL import Image

from repair_agent.config import settings

# Common hardware serial number patterns
SERIAL_PATTERNS = [
    re.compile(r"[A-Z]{2,4}[-\s]?[A-Z0-9]{4,10}", re.IGNORECASE),  # SN-ABC12345
    re.compile(r"S/?N[:\s]*([A-Z0-9\-]+)", re.IGNORECASE),          # S/N: ABC-12345
    re.compile(r"Serial[:\s]*([A-Z0-9\-]+)", re.IGNORECASE),        # Serial: ABC12345
    re.compile(r"[\d]{3,4}[-\s][A-Z0-9]{4,8}", re.IGNORECASE),      # 1234-ABCD
]

# Whitelist characters for OCR
TESSERACT_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-:"


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
    pil_img = Image.fromarray(img)
    text = pytesseract.image_to_string(
        pil_img,
        config=f"--psm 6 --oem 3 -c tessedit_char_whitelist={TESSERACT_WHITELIST}",
    )
    return text.strip()


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
    pil_img = Image.fromarray(img)
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
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
