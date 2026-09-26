from repair_agent.agent.state import AgentState
from repair_agent.config import settings


def should_self_correct(state: AgentState) -> str:
    """Self-correct when any candidate box is in the uncertain band.

    The top box is almost always right on DeepPCB (val top-1 = 1.0); errors
    live in the low-scoring boxes, so the gate reads `min_confidence`.
    """
    confidence = state.get("min_confidence")
    if confidence is None:
        confidence = state.get("defect_confidence") or 0.0
    attempts = state.get("correction_attempts", 0)

    if confidence < settings.CONFIDENCE_THRESHOLD and attempts < settings.MAX_CORRECTION_RETRIES:
        return "self_correct"
    return "diagnose"


def correction_complete(state: AgentState) -> str:
    """After OCR + re-query: decide if another retry is needed or proceed to diagnosis."""
    attempts = state.get("correction_attempts", 0)
    serial = state.get("serial_number")

    # Template verification is deterministic; repeating it cannot change the result.
    if "template" in (state.get("correction_mode") or ""):
        return "diagnose"
    if serial and attempts < settings.MAX_CORRECTION_RETRIES:
        return "diagnose"
    if attempts >= settings.MAX_CORRECTION_RETRIES:
        return "diagnose"
    return "retry"
