from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Input
    image_bytes: bytes
    image_path: Optional[str]

    # CV outputs
    defect_type: Optional[str]
    defect_confidence: Optional[float]
    defect_bbox: Optional[tuple]
    cropped_image_bytes: Optional[bytes]

    # OCR outputs
    serial_number: Optional[str]
    ocr_confidence: Optional[float]

    # RAG outputs
    rag_documents: Optional[List[dict]]
    rag_query: Optional[str]
    diagnosis: Optional[str]

    # Control flow
    correction_attempts: int
    self_correction_triggered: bool

    # Message history (LangGraph native)
    messages: Annotated[List[BaseMessage], add_messages]
