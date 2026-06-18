"""Pydantic request/response models for the Repair Agent API."""

from typing import Optional

from pydantic import BaseModel, Field


class DiagnoseResponse(BaseModel):
    """Response model for the /diagnose endpoint."""

    session_id: str = Field(..., description="Unique diagnostic session ID")
    diagnosis: str = Field(..., description="Full diagnostic report text")
    defect_type: Optional[str] = Field(None, description="Detected defect classification")
    defect_confidence: Optional[float] = Field(None, description="CV confidence score (0.0-1.0)")
    serial_number: Optional[str] = Field(None, description="Extracted hardware serial number")
    self_correction_triggered: bool = Field(
        False, description="Whether the self-correction loop was triggered"
    )
    correction_attempts: int = Field(0, description="Number of correction retries")
    rag_documents_used: int = Field(0, description="Number of RAG documents consulted")


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""

    status: str = Field("ok", description="Service health status")
    version: str = Field("1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., description="Error detail message")
    session_id: Optional[str] = Field(None, description="Session ID if available")
