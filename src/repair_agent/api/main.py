"""FastAPI application entrypoint for the Vision Repair Agent."""

import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from repair_agent.api.routes.diagnose import router as diagnose_router
from repair_agent.api.routes.health import router as health_router
from repair_agent.api.routes.sessions import router as sessions_router
from repair_agent.api.runtime import create_runtime

logger = structlog.get_logger()


def create_app(
    persistence_backend: str | None = None,
    database_url: str | None = None,
) -> FastAPI:
    """Build a FastAPI app. Arguments override PERSISTENCE_BACKEND / DATABASE_URL.

    The agent runtime (graph + checkpointer + session store) is opened in the lifespan
    and closed on shutdown. If the ASGI server skips lifespan (e.g. httpx ASGITransport),
    it is created lazily on the first request and closed by ``app.state.aclose_runtime``.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("vision_repair_agent_startup", version="1.0.0")
        await app.state.get_runtime()
        try:
            yield
        finally:
            await app.state.aclose_runtime()
            logger.info("vision_repair_agent_shutdown")

    app = FastAPI(
        title="Vision Repair Agent",
        description="Autonomous hardware diagnostic AI using Computer Vision, OCR, and RAG.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.runtime = None
    lock = asyncio.Lock()

    async def get_runtime():
        if app.state.runtime is None:
            async with lock:
                if app.state.runtime is None:
                    app.state.runtime = await create_runtime(persistence_backend, database_url)
        return app.state.runtime

    async def aclose_runtime() -> None:
        runtime, app.state.runtime = app.state.runtime, None
        if runtime is not None:
            await runtime.aclose()

    app.state.get_runtime = get_runtime
    app.state.aclose_runtime = aclose_runtime

    # CORS — allow all origins for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(diagnose_router, prefix="/api/v1")
    app.include_router(sessions_router, prefix="/api/v1")
    app.include_router(health_router, prefix="/api/v1")
    return app


app = create_app()
