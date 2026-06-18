"""Health check endpoint."""

from fastapi import APIRouter

from repair_agent.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Check API health status."""
    return HealthResponse(status="ok", version="1.0.0")
