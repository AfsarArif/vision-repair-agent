"""Database URL helpers and lazily-created engines.

Nothing here connects at import time: the default (memory) persistence path and the
test suite must work with no database and an empty ``DATABASE_URL``.

Two drivers are in play:

* SQLAlchemy async (``postgresql+asyncpg://``) for the ``diagnostic_sessions`` table.
* psycopg 3 (plain ``postgresql://`` conninfo) for LangGraph's ``AsyncPostgresSaver``.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from repair_agent.config import settings


def to_async_sqlalchemy_url(url: str) -> str:
    """Normalise any postgres URL to the ``postgresql+asyncpg`` SQLAlchemy dialect."""
    parsed = make_url(url)
    return parsed.set(drivername="postgresql+asyncpg").render_as_string(hide_password=False)


def to_psycopg_conninfo(url: str) -> str:
    """Convert a SQLAlchemy URL (e.g. ``postgresql+asyncpg://...``) to a libpq/psycopg URI.

    psycopg does not understand the ``+driver`` suffix, so ``postgresql+asyncpg://u:p@h/db``
    becomes ``postgresql://u:p@h/db``. Query parameters are preserved.
    """
    parsed = make_url(url)
    return parsed.set(drivername="postgresql").render_as_string(hide_password=False)


def create_async_db_engine(database_url: str | None = None) -> AsyncEngine:
    """Create a new async SQLAlchemy engine (caller owns ``await engine.dispose()``)."""
    url = database_url or settings.DATABASE_URL
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_async_engine(
        to_async_sqlalchemy_url(url), echo=False, pool_size=5, max_overflow=10, pool_pre_ping=True
    )


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@lru_cache(maxsize=1)
def get_sync_engine() -> Engine:
    """Sync engine for scripts / pgvector (langchain-postgres). Created on first use."""
    if not settings.SYNC_DATABASE_URL:
        raise RuntimeError("SYNC_DATABASE_URL is not set")
    return create_engine(settings.SYNC_DATABASE_URL, echo=False, pool_size=5)
