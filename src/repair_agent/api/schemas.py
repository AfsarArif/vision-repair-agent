"""Pydantic request/response models for the Repair Agent API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DetectionOut(BaseModel):
    cls: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    score: float | None = None


class DiagnoseResponse(BaseModel):
    """Response model for the /diagnose endpoint."""

    session_id: str = Field(..., description="Unique diagnostic session ID")
    diagnosis: str = Field(..., description="Full diagnostic report text")
    defect_type: Optional[str] = Field(None, description="Detected defect classification")
    defect_confidence: Optional[float] = Field(None, description="CV confidence score (0.0-1.0)")
    serial_number: Optional[str] = Field(
        None, description="Silkscreen designator or serial if OCR found one"
    )
    designator: Optional[str] = Field(None, description="PCB reference designator if detected")
    correction_mode: Optional[str] = Field(
        None, description="template_diff, designator_ocr, both, or none"
    )
    detections: list[DetectionOut] = Field(default_factory=list)
    self_correction_triggered: bool = Field(
        False, description="Whether the self-correction loop was triggered"
    )
    correction_attempts: int = Field(0, description="Number of correction retries")
    rag_documents_used: int = Field(0, description="Number of RAG documents consulted")
    cv_backend: Optional[str] = Field(None, description="CV backend that produced detections")
    template_provided: bool = Field(
        False, description="Whether a DeepPCB-style template image was uploaded"
    )
    persistence_backend: str = Field(
        "memory", description="memory (lost on restart) or postgres (survives restart)"
    )


class CheckpointSummary(BaseModel):
    """Summary of the latest LangGraph checkpoint for a thread (thread_id == session_id)."""

    checkpoint_id: Optional[str] = None
    created_at: Optional[str] = None
    completed: bool = Field(..., description="True when the graph reached END (no next nodes)")
    next_nodes: list[str] = Field(default_factory=list)
    defect_type: Optional[str] = None
    defect_confidence: Optional[float] = None
    designator: Optional[str] = None
    serial_number: Optional[str] = None
    correction_mode: Optional[str] = None
    self_correction_triggered: bool = False
    correction_attempts: int = 0
    cv_backend: Optional[str] = None
    rag_query: Optional[str] = None
    rag_documents_used: int = 0
    detections: list[DetectionOut] = Field(default_factory=list)
    diagnosis: Optional[str] = None


class SessionResponse(BaseModel):
    """Response model for GET /sessions/{session_id}."""

    session_id: str
    status: str = Field(..., description="completed or failed")
    error: Optional[str] = None
    defect_type: Optional[str] = None
    defect_confidence: Optional[float] = None
    serial_number: Optional[str] = None
    designator: Optional[str] = None
    diagnosis: Optional[str] = None
    correction_mode: Optional[str] = None
    cv_backend: Optional[str] = None
    image_sha256: Optional[str] = None
    template_sha256: Optional[str] = None
    detections: list[DetectionOut] = Field(default_factory=list)
    rag_documents_used: int = 0
    self_correction_triggered: bool = False
    correction_attempts: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    persistence_backend: str
    checkpoint: Optional[CheckpointSummary] = Field(
        None, description="Final LangGraph state for this thread, if the checkpointer has it"
    )


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""

    status: str = Field("ok", description="Service health status")
    version: str = Field("1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., description="Error detail message")
    session_id: Optional[str] = Field(None, description="Session ID if available")
