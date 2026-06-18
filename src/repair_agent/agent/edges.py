from repair_agent.agent.state import AgentState
from repair_agent.config import settings


def should_self_correct(state: AgentState) -> str:
    """Trigger OCR self-correction if RAG confidence is below threshold."""
    confidence = state.get("defect_confidence", 0.0)
    attempts = state.get("correction_attempts", 0)

    if confidence < settings.CONFIDENCE_THRESHOLD and attempts < settings.MAX_CORRECTION_RETRIES:
        return "self_correct"
    return "diagnose"


def correction_complete(state: AgentState) -> str:
    """After OCR + re-query: decide if another retry is needed or proceed to diagnosis."""
    attempts = state.get("correction_attempts", 0)
    serial = state.get("serial_number")

    if serial and attempts < settings.MAX_CORRECTION_RETRIES:
        return "diagnose"
    if attempts >= settings.MAX_CORRECTION_RETRIES:
        return "diagnose"
    return "retry"
