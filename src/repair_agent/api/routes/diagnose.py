"""Diagnose endpoint — submit an image and receive a diagnostic report."""

import uuid
import structlog

from fastapi import APIRouter, File, HTTPException, UploadFile

from repair_agent.api.schemas import DiagnoseResponse, ErrorResponse
from repair_agent.agent.graph import get_agent_sync
from repair_agent.agent.state import initial_agent_state

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
    1. Computer Vision — heuristic, template-diff, or YOLO (see CV_BACKEND)
    2. RAG — retrieve public workmanship chunks for the defect class
    3. Self-correct (conditional) — template absdiff and/or designator OCR
    4. RAG (corrected) — re-query with designator if found
    5. Diagnosis — DeepSeek, citing retrieved sources only

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
        initial_state = initial_agent_state(image_bytes)

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
            designator=final_state.get("designator"),
            correction_mode=final_state.get("correction_mode"),
            detections=final_state.get("detections") or [],
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
