"""Computer Vision node: heuristic, template-diff, or YOLO backend."""

from typing import Never

from repair_agent.agent.state import AgentState, Detection
from repair_agent.config import settings
from repair_agent.tools.cv_tools import (
    crop_region,
    decode_image,
    preprocess_for_contour_detection,
    find_defect_contours,
    classify_defect_heuristic,
    estimate_confidence,
)
from repair_agent.tools.template_diff import localize_defects
from repair_agent.tools.yolo_detector import detect_yolo, resolve_weights


def _empty_normal(backend: str = "heuristic") -> dict:
    return {
        "defect_type": "normal",
        "defect_confidence": 0.95,
        "defect_bbox": None,
        "cropped_image_bytes": None,
        "detections": [],
        "cv_backend": backend,
    }


def _from_detections(img, detections: list[Detection], backend: str) -> dict:
    if not detections:
        return _empty_normal(backend)
    ranked = sorted(detections, key=lambda d: float(d.get("score") or 0.0), reverse=True)
    top = ranked[0]
    bbox = top["bbox"]
    cls = top.get("cls") or "short"
    cropped = crop_region(img, bbox)
    return {
        "defect_type": cls,
        "defect_confidence": float(top.get("score") or 0.0),
        "defect_bbox": bbox,
        "cropped_image_bytes": cropped,
        "detections": ranked,
        "cv_backend": backend,
    }


def _heuristic_pipeline(img) -> dict:
    height, width = img.shape[:2]
    thresh = preprocess_for_contour_detection(img)
    contour_results = find_defect_contours(thresh)
    if not contour_results:
        return _empty_normal("heuristic")

    largest_contour = max(contour_results, key=lambda r: r[1][2] * r[1][3])
    contour, bbox = largest_contour
    defect_type = classify_defect_heuristic(img, contour, bbox)
    confidence = estimate_confidence([c for c, _ in contour_results], height * width)
    detections: list[Detection] = [
        {"cls": defect_type, "bbox": bbox, "score": confidence}
    ]
    return {
        "defect_type": defect_type,
        "defect_confidence": confidence,
        "defect_bbox": bbox,
        "cropped_image_bytes": crop_region(img, bbox),
        "detections": detections,
        "cv_backend": "heuristic",
    }


async def cv_node(state: AgentState) -> dict:
    """Detect defects using the configured CV backend."""
    img = decode_image(state["image_bytes"])
    backend = settings.CV_BACKEND
    template_bytes = state.get("template_image_bytes")

    if backend == "heuristic":
        return _heuristic_pipeline(img)

    if backend == "template_diff":
        if not template_bytes:
            return _heuristic_pipeline(img)
        template = decode_image(template_bytes)
        detections = localize_defects(img, template)
        return _from_detections(img, detections, "template_diff")

    if backend == "yolo":
        weights = resolve_weights(settings.YOLO_WEIGHTS or None)
        if weights is None:
            return _heuristic_pipeline(img)
        try:
            detections = detect_yolo(img, str(weights))
        except (FileNotFoundError, ImportError):
            return _heuristic_pipeline(img)
        return _from_detections(img, detections, "yolo")

    _exhaustive: Never = backend
    raise ValueError(f"Unknown CV backend: {_exhaustive}")
