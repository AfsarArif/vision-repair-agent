"""Diagnose endpoint — submit an image and receive a diagnostic report."""

import hashlib
import uuid
from typing import Any, Optional

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from repair_agent.agent.state import initial_agent_state
from repair_agent.api.runtime import AgentRuntime, runtime_dependency
from repair_agent.api.schemas import DiagnoseResponse, ErrorResponse

logger = structlog.get_logger()
router = APIRouter()

# Supported image MIME types
SUPPORTED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}


def jsonable_detections(detections: Any) -> list[dict[str, Any]]:
    """Coerce CV detections (tuples / numpy scalars) into JSON-safe dicts."""
    out: list[dict[str, Any]] = []
    for d in detections or []:
        bbox = d.get("bbox")
        score = d.get("score")
        cls = d.get("cls")
        out.append(
            {
                "cls": None if cls is None else str(cls),
                "bbox": None if bbox is None else [int(v) for v in bbox],
                "score": None if score is None else float(score),
            }
        )
    return out


async def _read_upload(upload: UploadFile, label: str) -> bytes:
    if upload.content_type and upload.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported {label} type: {upload.content_type}. "
                f"Supported: {', '.join(sorted(SUPPORTED_TYPES))}"
            ),
        )
    try:
        data = await upload.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded {label}: {e}")
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty {label} uploaded.")
    return data


async def _persist(runtime: AgentRuntime, row: dict[str, Any]) -> None:
    try:
        await runtime.store.save(row)
    except Exception as e:  # persistence must not mask the diagnosis result
        logger.error("session_persist_failed", session_id=row.get("id"), error=str(e))


@router.post(
    "/diagnose",
    response_model=DiagnoseResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def diagnose(
    file: UploadFile = File(..., description="Test image (PNG, JPEG, BMP, TIFF)"),
    template: Optional[UploadFile] = File(
        None,
        description=(
            "Optional defect-free template of the same board (DeepPCB-style). "
            "Enables the template-diff self-correct path."
        ),
    ),
    runtime: AgentRuntime = Depends(runtime_dependency),
):
    """Submit a hardware image (and optionally its golden template) for diagnosis.

    The image is processed through:
    1. Computer Vision — heuristic, template-diff, or YOLO (see CV_BACKEND)
    2. RAG — retrieve public workmanship chunks for the defect class
    3. Self-correct (conditional) — template absdiff and/or designator OCR
    4. RAG (corrected) — re-query with designator if found
    5. Diagnosis — DeepSeek, citing retrieved sources only

    A ``diagnostic_sessions`` row is stored per call and the LangGraph thread uses
    ``thread_id == session_id``; fetch both later with ``GET /sessions/{session_id}``.
    """
    image_bytes = await _read_upload(file, "file")
    template_bytes = await _read_upload(template, "template") if template is not None else None

    session_id = str(uuid.uuid4())
    base_row: dict[str, Any] = {
        "id": session_id,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "template_sha256": (
            hashlib.sha256(template_bytes).hexdigest() if template_bytes is not None else None
        ),
    }

    try:
        initial_state = initial_agent_state(image_bytes, template_image_bytes=template_bytes)
        config = {"configurable": {"thread_id": session_id}}
        final_state = await runtime.agent.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.error("diagnosis_failed", session_id=session_id, error=str(e))
        await _persist(runtime, {**base_row, "status": "failed", "error": str(e)[:2000]})
        raise HTTPException(
            status_code=500,
            detail=f"Diagnosis failed: {str(e)}",
            headers={"X-Session-Id": session_id},
        )

    detections = jsonable_detections(final_state.get("detections"))
    rag_documents_used = len(final_state.get("rag_documents") or [])
    diagnosis_text = final_state.get("diagnosis") or "Unable to diagnose."

    await _persist(
        runtime,
        {
            **base_row,
            "status": "completed",
            "error": None,
            "defect_type": final_state.get("defect_type"),
            "defect_confidence": final_state.get("defect_confidence"),
            "serial_number": final_state.get("serial_number"),
            "designator": final_state.get("designator"),
            "diagnosis": diagnosis_text,
            "correction_mode": final_state.get("correction_mode"),
            "cv_backend": final_state.get("cv_backend"),
            "detections": detections,
            "rag_documents_used": rag_documents_used,
            "self_correction_triggered": bool(final_state.get("self_correction_triggered")),
            "correction_attempts": int(final_state.get("correction_attempts") or 0),
        },
    )

    logger.info(
        "diagnosis_complete",
        session_id=session_id,
        defect_type=final_state.get("defect_type"),
        confidence=final_state.get("defect_confidence"),
        self_correction=final_state.get("self_correction_triggered"),
        template=template_bytes is not None,
        persistence_backend=runtime.backend,
    )

    return DiagnoseResponse(
        session_id=session_id,
        diagnosis=diagnosis_text,
        defect_type=final_state.get("defect_type"),
        defect_confidence=final_state.get("defect_confidence"),
        serial_number=final_state.get("serial_number"),
        designator=final_state.get("designator"),
        correction_mode=final_state.get("correction_mode"),
        detections=detections,
        self_correction_triggered=final_state.get("self_correction_triggered", False),
        correction_attempts=final_state.get("correction_attempts", 0),
        rag_documents_used=rag_documents_used,
        cv_backend=final_state.get("cv_backend"),
        template_provided=template_bytes is not None,
        persistence_backend=runtime.backend,
    )
