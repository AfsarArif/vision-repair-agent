"""Self-correction: template-diff refine and/or silkscreen designator OCR."""

from repair_agent.agent.state import AgentState
from repair_agent.tools.cv_tools import crop_region, decode_image
from repair_agent.tools.ocr_tools import (
    decode_and_preprocess,
    estimate_ocr_confidence,
    extract_text,
    find_designator,
    find_serial_number,
)
from repair_agent.tools.template_diff import localize_defects


async def self_correct_node(state: AgentState) -> dict:
    """Refine localization with a template and/or read a reference designator.

    DeepPCB samples have templates and no silkscreen. Color boards may have
    designators and no template. Both branches are attempted; missing inputs
    are skipped.
    """
    updates: dict = {
        "correction_attempts": state.get("correction_attempts", 0) + 1,
        "self_correction_triggered": True,
        "correction_mode": None,
    }
    modes: list[str] = []

    template_bytes = state.get("template_image_bytes")
    if template_bytes:
        test = decode_image(state["image_bytes"])
        template = decode_image(template_bytes)
        detections = localize_defects(test, template)
        if detections:
            top = detections[0]
            bbox = top["bbox"]
            updates["detections"] = detections
            updates["defect_bbox"] = bbox
            updates["defect_confidence"] = float(top.get("score") or 0.0)
            updates["cropped_image_bytes"] = crop_region(test, bbox)
            if top.get("cls"):
                updates["defect_type"] = top["cls"]
            modes.append("template_diff")

    img_bytes = updates.get("cropped_image_bytes") or state.get("cropped_image_bytes") or state["image_bytes"]
    preprocessed = decode_and_preprocess(img_bytes)
    raw_text = extract_text(preprocessed)
    designator = find_designator(raw_text)
    serial = find_serial_number(raw_text)
    token = designator or serial
    ocr_conf = estimate_ocr_confidence(preprocessed)
    updates["ocr_confidence"] = ocr_conf
    updates["designator"] = designator
    updates["serial_number"] = token
    if token:
        modes.append("designator_ocr" if designator else "serial_ocr")

    updates["correction_mode"] = "+".join(modes) if modes else "none"
    return updates
