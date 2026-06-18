"""Diagnose endpoint — submit an image and receive a diagnostic report."""

import uuid
import structlog

from fastapi import APIRouter, File, HTTPException, UploadFile

from repair_agent.api.schemas import DiagnoseResponse, ErrorResponse
from repair_agent.agent.graph import get_agent_sync
from repair_agent.agent.state import AgentState
from repair_agent.config import settings

logger = structlog.get_logger()
router = APIRouter()

# Supported image MIME types
SUPPORTED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}


@router.post(
    "/diagnose",
    response_model=DiagnoseResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def diagnose(file: UploadFile = File(...)):
    """Submit a hardware image for automated diagnosis.

    The image is processed through:
    1. Computer Vision — defect detection and classification
    2. RAG — initial document retrieval
    3. OCR (conditional) — serial number extraction if confidence is low
    4. RAG (corrected) — precise document re-query with serial number
    5. Diagnosis synthesis — GPT-4o generates a structured report

    Args:
        file: UploadFile of the hardware image (PNG, JPEG, BMP, TIFF).

    Returns:
        DiagnoseResponse with full diagnostic report and metadata.
    """
    # Validate file type
    if file.content_type and file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Supported: {', '.join(SUPPORTED_TYPES)}",
        )

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    session_id = str(uuid.uuid4())

    try:
        agent = get_agent_sync()

        initial_state: AgentState = {
            "image_bytes": image_bytes,
            "image_path": None,
            "defect_type": None,
            "defect_confidence": None,
            "defect_bbox": None,
            "cropped_image_bytes": None,
            "serial_number": None,
            "ocr_confidence": None,
            "rag_documents": None,
            "rag_query": None,
            "diagnosis": None,
            "correction_attempts": 0,
            "self_correction_triggered": False,
            "messages": [],
        }

        config = {"configurable": {"thread_id": session_id}}
        final_state = await agent.ainvoke(initial_state, config=config)

        logger.info(
            "diagnosis_complete",
            session_id=session_id,
            defect_type=final_state.get("defect_type"),
            confidence=final_state.get("defect_confidence"),
            self_correction=final_state.get("self_correction_triggered"),
        )

        return DiagnoseResponse(
            session_id=session_id,
            diagnosis=final_state.get("diagnosis", "Unable to diagnose."),
            defect_type=final_state.get("defect_type"),
            defect_confidence=final_state.get("defect_confidence"),
            serial_number=final_state.get("serial_number"),
            self_correction_triggered=final_state.get("self_correction_triggered", False),
            correction_attempts=final_state.get("correction_attempts", 0),
            rag_documents_used=len(final_state.get("rag_documents") or []),
        )

    except Exception as e:
        logger.error("diagnosis_failed", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Diagnosis failed: {str(e)}",
        )
