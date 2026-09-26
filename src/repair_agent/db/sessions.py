"""DiagnosticSession stores.

``MemorySessionStore`` lives only as long as the process (the default, no DB needed).
``PostgresSessionStore`` writes ``diagnostic_sessions`` rows via async SQLAlchemy so a
session survives an API restart. Both speak plain dicts keyed by the model's columns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from repair_agent.db.connection import make_sessionmaker
from repair_agent.db.models import DiagnosticSession

SESSION_COLUMNS: tuple[str, ...] = tuple(c.name for c in DiagnosticSession.__table__.columns)


class SessionStore(Protocol):
    async def save(self, row: dict[str, Any]) -> None: ...

    async def get(self, session_id: str) -> dict[str, Any] | None: ...

    async def aclose(self) -> None: ...


def _clean(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k in SESSION_COLUMNS}


class MemorySessionStore:
    """Process-local store: rows vanish when the process exits."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    async def save(self, row: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        existing = self._rows.get(row["id"], {"created_at": now})
        self._rows[row["id"]] = {**existing, **_clean(row), "updated_at": now}

    async def get(self, session_id: str) -> dict[str, Any] | None:
        row = self._rows.get(session_id)
        return dict(row) if row is not None else None

    async def aclose(self) -> None:
        self._rows.clear()


class PostgresSessionStore:
    """Upserts ``diagnostic_sessions`` rows. Requires ``alembic upgrade head``."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._sessionmaker = make_sessionmaker(engine)

    async def save(self, row: dict[str, Any]) -> None:
        values = _clean(row)
        stmt = pg_insert(DiagnosticSession).values(**values)
        update_cols = {k: stmt.excluded[k] for k in values if k != "id"}
        update_cols["updated_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
        stmt = stmt.on_conflict_do_update(index_elements=["id"], set_=update_cols)
        async with self._sessionmaker() as session:
            await session.execute(stmt)
            await session.commit()

    async def get(self, session_id: str) -> dict[str, Any] | None:
        async with self._sessionmaker() as session:
            obj = await session.get(DiagnosticSession, session_id)
            if obj is None:
                return None
            return {c: getattr(obj, c) for c in SESSION_COLUMNS}

    async def aclose(self) -> None:
        await self._engine.dispose()
