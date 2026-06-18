"""Computer Vision utilities for defect detection and image preprocessing."""

import cv2
import numpy as np

DEFECT_CLASSES = ["burn_mark", "crack", "corrosion", "delamination", "normal"]

# HSV ranges for common defect types
DEFECT_COLOR_RANGES = {
    "burn_mark": {
        "lower": np.array([0, 20, 10]),
        "upper": np.array([25, 255, 120]),
    },
    "corrosion": {
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

    Heuristics:
    - Elongated contour (aspect ratio > 4) → crack
    - Dark region (low mean value) → burn_mark
    - Greenish hue → corrosion
    - Otherwise → delamination or normal
    """
    x, y, w, h = bbox
    aspect_ratio = w / max(h, 1)

    # Crack: very elongated
    if aspect_ratio > 4 or (w > 5 * h):
        return "crack"

    # Color-based classification on ROI
    roi = img[y : y + h, x : x + w]
    if roi.size == 0:
        return "normal"

    # Convert to HSV for color analysis
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Check burn mark (dark brown/black)
    burn_mask = cv2.inRange(hsv_roi, DEFECT_COLOR_RANGES["burn_mark"]["lower"], DEFECT_COLOR_RANGES["burn_mark"]["upper"])
    burn_ratio = cv2.countNonZero(burn_mask) / roi.size

    # Check corrosion (greenish)
    corrosion_mask = cv2.inRange(hsv_roi, DEFECT_COLOR_RANGES["corrosion"]["lower"], DEFECT_COLOR_RANGES["corrosion"]["upper"])
    corrosion_ratio = cv2.countNonZero(corrosion_mask) / roi.size

    if burn_ratio > 0.3:
        return "burn_mark"
    if corrosion_ratio > 0.2:
        return "corrosion"

    # Fallback: check mean brightness
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_val = gray_roi.mean()
    if mean_val < 50:
        return "burn_mark"
    if mean_val > 200:
        return "delamination"

    return "corrosion"


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
