"""Per-app runtime: one compiled agent + checkpointer + DiagnosticSession store.

Opened in the FastAPI lifespan and closed on shutdown. With the ``memory`` backend
everything lives in-process; with ``postgres`` both the LangGraph checkpoints and the
``diagnostic_sessions`` rows survive a restart.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request

import structlog

from repair_agent.agent.graph import get_agent
from repair_agent.config import settings
from repair_agent.db.sessions import MemorySessionStore, PostgresSessionStore, SessionStore

logger = structlog.get_logger()


@dataclass
class AgentRuntime:
    agent: Any
    store: SessionStore
    backend: str
    _stack: AsyncExitStack = field(default_factory=AsyncExitStack, repr=False)

    async def aclose(self) -> None:
        try:
            await self.store.aclose()
        finally:
            await self._stack.aclose()


def resolve_backend(backend: str | None = None, database_url: str | None = None) -> tuple[str, str]:
    """Return ``(backend, database_url)``. Postgres needs both the setting and a URL."""
    backend = backend or settings.PERSISTENCE_BACKEND
    database_url = database_url if database_url is not None else settings.DATABASE_URL
    if backend == "postgres" and not database_url:
        logger.warning("persistence_postgres_without_database_url_falling_back_to_memory")
        backend = "memory"
    if backend not in ("memory", "postgres"):
        raise ValueError(f"Unknown PERSISTENCE_BACKEND: {backend!r}")
    return backend, database_url


async def create_runtime(
    backend: str | None = None, database_url: str | None = None
) -> AgentRuntime:
    backend, database_url = resolve_backend(backend, database_url)
    stack = AsyncExitStack()
    try:
        if backend == "postgres":
            from repair_agent.db.connection import create_async_db_engine

            agent = await stack.enter_async_context(get_agent(database_url))
            store: SessionStore = PostgresSessionStore(create_async_db_engine(database_url))
        else:
            agent = await stack.enter_async_context(get_agent(None))
            store = MemorySessionStore()
    except BaseException:
        await stack.aclose()
        raise
    logger.info("agent_runtime_ready", persistence_backend=backend)
    return AgentRuntime(agent=agent, store=store, backend=backend, _stack=stack)


async def runtime_dependency(request: Request) -> AgentRuntime:
    """FastAPI dependency returning this app's (lazily created) runtime."""
    return await request.app.state.get_runtime()
