"""Non-learned defect localization by comparing a test image to an aligned template."""

from __future__ import annotations

import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def absdiff_mask(
    test_img: np.ndarray,
    template_img: np.ndarray,
    blur_ksize: int = 5,
    diff_threshold: int = 30,
) -> np.ndarray:
    """Binary mask of pixels that differ between test and template."""
    test = _to_gray(test_img)
    template = _to_gray(template_img)
    if test.shape != template.shape:
        template = cv2.resize(template, (test.shape[1], test.shape[0]), interpolation=cv2.INTER_NEAREST)
    if blur_ksize > 1:
        test = cv2.GaussianBlur(test, (blur_ksize, blur_ksize), 0)
        template = cv2.GaussianBlur(template, (blur_ksize, blur_ksize), 0)
    diff = cv2.absdiff(test, template)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def localize_defects(
    test_img: np.ndarray,
    template_img: np.ndarray,
    min_area: int = 20,
    diff_threshold: int = 30,
) -> list[dict]:
    """Return detections `{cls, bbox (xywh), score}` without a class name.

    `cls` is `None` — template-diff is class-agnostic. Score is the mean
    absolute difference inside the box, scaled to 0–1.
    """
    mask = absdiff_mask(test_img, template_img, diff_threshold=diff_threshold)
    test_gray = _to_gray(test_img)
    template_gray = _to_gray(template_img)
    if template_gray.shape != test_gray.shape:
        template_gray = cv2.resize(
            template_gray, (test_gray.shape[1], test_gray.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[dict] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        roi_test = test_gray[y : y + h, x : x + w].astype(np.float32)
        roi_temp = template_gray[y : y + h, x : x + w].astype(np.float32)
        mean_diff = float(np.mean(np.abs(roi_test - roi_temp)))
        score = min(mean_diff / 255.0, 1.0)
        detections.append(
            {
                "cls": None,
                "bbox": (int(x), int(y), int(w), int(h)),
                "score": round(score, 4),
            }
        )
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections
