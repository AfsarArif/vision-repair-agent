"""Synthetic aligned PCB pairs for CI eval (no DeepPCB download)."""

from __future__ import annotations

import cv2
import numpy as np


def make_trace_template(width: int = 320, height: int = 320) -> np.ndarray:
    """Black board with three horizontal white traces (DeepPCB-like binary)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in (80, 160, 240):
        cv2.rectangle(img, (20, y - 8), (width - 20, y + 8), (255, 255, 255), -1)
    return img


def apply_open(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Cut a gap in the middle trace. Returns (image, xyxy gold box)."""
    out = img.copy()
    x1, y1, x2, y2 = 140, 148, 180, 172
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return out, (x1, y1, x2, y2)


def apply_short(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Bridge two traces with extra copper."""
    out = img.copy()
    x1, y1, x2, y2 = 150, 88, 170, 232
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), -1)
    return out, (x1, y1, x2, y2)


def encode_png(img: np.ndarray) -> bytes:
    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()
