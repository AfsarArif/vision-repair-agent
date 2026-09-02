"""Computer Vision utilities for defect detection and image preprocessing."""

import cv2
import numpy as np

from repair_agent.taxonomy import CANONICAL_CLASSES

DEFECT_CLASSES = list(CANONICAL_CLASSES)

# HSV ranges used only by the deprecated heuristic backend (synthetic / demo images).
DEFECT_COLOR_RANGES = {
    "spurious_copper": {
        "lower": np.array([0, 20, 10]),
        "upper": np.array([25, 255, 120]),
    },
    "mousebite": {
        "lower": np.array([30, 30, 60]),
        "upper": np.array([90, 180, 200]),
    },
}


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV BGR image."""
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes.")
    return img


def preprocess_for_contour_detection(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale, blur, and threshold for contour detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def find_defect_contours(
    thresh: np.ndarray, min_area: int = 500
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Find contours in thresholded image and return filtered list."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    results = [(c, cv2.boundingRect(c)) for c in contours]
    return results


def classify_defect_heuristic(
    img: np.ndarray, contour: np.ndarray, bbox: tuple[int, int, int, int]
) -> str:
    """Classify defect type based on contour properties and color analysis.

    Heuristics (deprecated; DeepPCB training replaces this in Phase B):
    - Elongated contour → open
    - Dark region → spurious_copper
    - Greenish hue → mousebite (legacy synthetic fixtures)
    - Bright region → spur
    """
    x, y, w, h = bbox
    aspect_ratio = w / max(h, 1)

    if aspect_ratio > 4 or (w > 5 * h):
        return "open"

    roi = img[y : y + h, x : x + w]
    if roi.size == 0:
        return "normal"

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    copper_mask = cv2.inRange(
        hsv_roi,
        DEFECT_COLOR_RANGES["spurious_copper"]["lower"],
        DEFECT_COLOR_RANGES["spurious_copper"]["upper"],
    )
    copper_ratio = cv2.countNonZero(copper_mask) / roi.size

    bite_mask = cv2.inRange(
        hsv_roi,
        DEFECT_COLOR_RANGES["mousebite"]["lower"],
        DEFECT_COLOR_RANGES["mousebite"]["upper"],
    )
    bite_ratio = cv2.countNonZero(bite_mask) / roi.size

    if copper_ratio > 0.3:
        return "spurious_copper"
    if bite_ratio > 0.2:
        return "mousebite"

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_val = gray_roi.mean()
    if mean_val < 50:
        return "spurious_copper"
    if mean_val > 200:
        return "spur"

    return "short"


def estimate_confidence(contours: list[np.ndarray], total_image_area: int) -> float:
    """Estimate defect detection confidence based on contour properties.

    Higher confidence when:
    - Multiple contours found (clear defect pattern)
    - Large total defect area relative to image
    """
    if not contours:
        return 0.95  # High confidence in "no defect" finding

    total_area = sum(cv2.contourArea(c) for c in contours)
    area_ratio = total_area / max(total_image_area, 1)

    # Scale: larger area ratio → more confident (capped at 0.99)
    confidence = min(0.5 + area_ratio * 5, 0.99)
    return round(confidence, 4)


def crop_region(img: np.ndarray, bbox: tuple[int, int, int, int]) -> bytes:
    """Crop the bounding box region from the image and return as PNG bytes."""
    x, y, w, h = bbox
    cropped = img[y : y + h, x : x + w]
    _, crop_encoded = cv2.imencode(".png", cropped)
    return crop_encoded.tobytes()
