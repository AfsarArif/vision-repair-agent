"""YOLO detector loader (Phase B). Safe to import when weights are missing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from repair_agent.taxonomy import yolo_index_to_class

_MODEL_CACHE: dict[str, object] = {}


def default_weights_path(data_dir: str | Path | None = None) -> Path:
    if data_dir is None:
        from repair_agent.config import settings

        data_dir = settings.DATA_DIR
    return Path(data_dir) / "processed" / "deeppcb" / "weights" / "best.pt"


def resolve_weights(
    explicit: str | None = None,
    data_dir: str | Path | None = None,
) -> Path | None:
    """Return a weights file path, or None if nothing is trained yet."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(default_weights_path(data_dir))
    for path in candidates:
        if path.is_file():
            return path
    return None


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()


def _load_model(weights_path: str):
    cached = _MODEL_CACHE.get(weights_path)
    if cached is not None:
        return cached
    from ultralytics import YOLO  # optional extra: poetry install --extras train

    model = YOLO(weights_path)
    _MODEL_CACHE[weights_path] = model
    return model


def detect_yolo(img: np.ndarray, weights_path: str, conf: float = 0.25) -> list[dict]:
    """Run Ultralytics YOLO and return `{cls, bbox (xywh), score}`.

    Raises ImportError if ultralytics is not installed, FileNotFoundError if
    weights are missing. The model is cached per weights path.
    """
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    model = _load_model(str(path.resolve()))
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
