"""Session lookup — stored DiagnosticSession row + LangGraph checkpoint summary."""

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException

from repair_agent.api.routes.diagnose import jsonable_detections
from repair_agent.api.runtime import AgentRuntime, runtime_dependency
from repair_agent.api.schemas import CheckpointSummary, ErrorResponse, SessionResponse

logger = structlog.get_logger()
router = APIRouter()


async def checkpoint_summary(runtime: AgentRuntime, session_id: str) -> CheckpointSummary | None:
    """Read the latest checkpoint for ``thread_id == session_id`` (None if absent)."""
    config = {"configurable": {"thread_id": session_id}}
    try:
        snapshot = await runtime.agent.aget_state(config)
    except Exception as e:
        logger.error("checkpoint_read_failed", session_id=session_id, error=str(e))
        return None
    values: dict[str, Any] = dict(snapshot.values or {})
    if not values:
        return None
    cfg = (snapshot.config or {}).get("configurable", {})
    return CheckpointSummary(
        checkpoint_id=cfg.get("checkpoint_id"),
        created_at=snapshot.created_at,
        completed=not snapshot.next,
        next_nodes=list(snapshot.next or ()),
        defect_type=values.get("defect_type"),
        defect_confidence=values.get("defect_confidence"),
        designator=values.get("designator"),
        serial_number=values.get("serial_number"),
        correction_mode=values.get("correction_mode"),
        self_correction_triggered=bool(values.get("self_correction_triggered")),
        correction_attempts=int(values.get("correction_attempts") or 0),
        cv_backend=values.get("cv_backend"),
        rag_query=values.get("rag_query"),
        rag_documents_used=len(values.get("rag_documents") or []),
        detections=jsonable_detections(values.get("detections")),
        diagnosis=values.get("diagnosis"),
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_session(session_id: str, runtime: AgentRuntime = Depends(runtime_dependency)):
    """Return the stored session row plus the LangGraph checkpoint's final state.

    With ``PERSISTENCE_BACKEND=memory`` sessions exist only until the process restarts;
    with ``postgres`` they survive restarts. Unknown IDs return 404.
    """
    row = await runtime.store.get(session_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    return SessionResponse(
        session_id=row["id"],
        status=row.get("status") or "completed",
        error=row.get("error"),
        defect_type=row.get("defect_type"),
        defect_confidence=row.get("defect_confidence"),
        serial_number=row.get("serial_number"),
        designator=row.get("designator"),
        diagnosis=row.get("diagnosis"),
        correction_mode=row.get("correction_mode"),
        cv_backend=row.get("cv_backend"),
        image_sha256=row.get("image_sha256"),
        template_sha256=row.get("template_sha256"),
        detections=row.get("detections") or [],
        rag_documents_used=row.get("rag_documents_used") or 0,
        self_correction_triggered=bool(row.get("self_correction_triggered")),
        correction_attempts=row.get("correction_attempts") or 0,
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        persistence_backend=runtime.backend,
        checkpoint=await checkpoint_summary(runtime, session_id),
    )
