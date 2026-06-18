"""FastAPI application entrypoint for the Vision Repair Agent."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from repair_agent.api.routes.diagnose import router as diagnose_router
from repair_agent.api.routes.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    # Startup
    import structlog

    logger = structlog.get_logger()
    logger.info("vision_repair_agent_startup", version="1.0.0")
    yield
    # Shutdown
    logger.info("vision_repair_agent_shutdown")


app = FastAPI(
    title="Vision Repair Agent",
    description="Autonomous hardware diagnostic AI using Computer Vision, OCR, and RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(diagnose_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")
