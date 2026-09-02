"""YOLO detector loader (Phase B). Safe to import when weights are missing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from repair_agent.taxonomy import yolo_index_to_class


def detect_yolo(img: np.ndarray, weights_path: str, conf: float = 0.25) -> list[dict]:
    """Run Ultralytics YOLO and return `{cls, bbox (xywh), score}`.

    Raises ImportError if ultralytics is not installed, FileNotFoundError if
    weights are missing.
    """
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    from ultralytics import YOLO

    model = YOLO(str(path))
    results = model.predict(img, conf=conf, verbose=False)
    detections: list[dict] = []
    if not results:
        return detections
    result = results[0]
    if result.boxes is None:
        return detections
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        cls_idx = int(box.cls[0].item())
        score = float(box.conf[0].item())
        detections.append(
            {
                "cls": yolo_index_to_class(cls_idx),
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "score": round(score, 4),
            }
        )
    return detections
