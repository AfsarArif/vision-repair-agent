"""OCR node for serial number extraction from hardware images."""

from repair_agent.agent.state import AgentState
from repair_agent.tools.ocr_tools import (
    decode_and_preprocess,
    extract_text,
    find_serial_number,
    estimate_ocr_confidence,
)


async def ocr_node(state: AgentState) -> dict:
    """Extract serial number from the cropped defect region (or full image).

    Uses the cropped_image_bytes if available, otherwise falls back to the
    full image.

    Args:
        state: Current AgentState with image_bytes and optional cropped_image_bytes.

    Returns:
        dict with serial_number, ocr_confidence, correction_attempts.
    """
    # Prefer cropped region; fall back to full image
    img_bytes = state.get("cropped_image_bytes") or state["image_bytes"]

    preprocessed = decode_and_preprocess(img_bytes)
    raw_text = extract_text(preprocessed)
    serial_number = find_serial_number(raw_text)
    ocr_conf = estimate_ocr_confidence(preprocessed)

    return {
        "serial_number": serial_number,
        "ocr_confidence": ocr_conf,
        "correction_attempts": state.get("correction_attempts", 0) + 1,
        "self_correction_triggered": True,
    }
