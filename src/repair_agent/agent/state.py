from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Detection(TypedDict, total=False):
    cls: Optional[str]
    bbox: tuple[int, int, int, int]
    score: float


class AgentState(TypedDict):
    # Input
    image_bytes: bytes
    image_path: Optional[str]
    template_image_bytes: Optional[bytes]

    # CV outputs
    defect_type: Optional[str]
    defect_confidence: Optional[float]
    defect_bbox: Optional[tuple]
    cropped_image_bytes: Optional[bytes]
    detections: Optional[List[Detection]]
    # YOLO boxes down to YOLO_CANDIDATE_CONF, for template verification.
    candidate_detections: Optional[List[Detection]]
    # Lowest candidate score; the self-correct gate reads this.
    min_confidence: Optional[float]
    cv_backend: Optional[str]

    # OCR / designator outputs (serial_number also holds a designator when found)
    serial_number: Optional[str]
    designator: Optional[str]
    ocr_confidence: Optional[float]

    # RAG outputs
    rag_documents: Optional[List[dict]]
    rag_query: Optional[str]
    diagnosis: Optional[str]

    # Control flow
    correction_attempts: int
    self_correction_triggered: bool
    correction_mode: Optional[str]

    # Message history (LangGraph native)
    messages: Annotated[List[BaseMessage], add_messages]


def initial_agent_state(
    image_bytes: bytes,
    template_image_bytes: Optional[bytes] = None,
    image_path: Optional[str] = None,
) -> AgentState:
    """Build a complete AgentState for a new diagnose run."""
    return {
        "image_bytes": image_bytes,
        "image_path": image_path,
        "template_image_bytes": template_image_bytes,
        "defect_type": None,
        "defect_confidence": None,
        "defect_bbox": None,
        "cropped_image_bytes": None,
        "detections": None,
        "candidate_detections": None,
        "min_confidence": None,
        "cv_backend": None,
        "serial_number": None,
        "designator": None,
        "ocr_confidence": None,
        "rag_documents": None,
        "rag_query": None,
        "diagnosis": None,
        "correction_attempts": 0,
        "self_correction_triggered": False,
        "correction_mode": None,
        "messages": [],
    }
