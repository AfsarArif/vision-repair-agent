"""Computer Vision node for defect detection in hardware images."""

from repair_agent.agent.state import AgentState
from repair_agent.tools.cv_tools import (
    crop_region,
    decode_image,
    preprocess_for_contour_detection,
    find_defect_contours,
    classify_defect_heuristic,
    estimate_confidence,
)


async def cv_node(state: AgentState) -> dict:
    """Process input image through CV pipeline.

    Flow:
    1. Decode image bytes
    2. Preprocess (grayscale → blur → threshold)
    3. Find defect contours
    4. Classify defect type
    5. Estimate confidence
    6. Crop the largest defect region for OCR

    Args:
        state: Current AgentState with image_bytes.

    Returns:
        dict with defect_type, defect_confidence, defect_bbox, cropped_image_bytes.
    """
    img = decode_image(state["image_bytes"])
    height, width = img.shape[:2]
    total_area = height * width

    thresh = preprocess_for_contour_detection(img)
    contour_results = find_defect_contours(thresh)

    if not contour_results:
        return {
            "defect_type": "normal",
            "defect_confidence": 0.95,
            "defect_bbox": None,
            "cropped_image_bytes": None,
        }

    # Process largest defect contour
    largest_contour = max(contour_results, key=lambda r: r[1][2] * r[1][3])
    contour, bbox = largest_contour

    defect_type = classify_defect_heuristic(img, contour, bbox)
    confidence = estimate_confidence([c for c, _ in contour_results], total_area)
    cropped_bytes = crop_region(img, bbox)

    return {
        "defect_type": defect_type,
        "defect_confidence": confidence,
        "defect_bbox": bbox,
        "cropped_image_bytes": cropped_bytes,
    }
