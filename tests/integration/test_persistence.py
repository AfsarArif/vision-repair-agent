"""Phase E — diagnose sessions survive a process restart (Postgres) but not with memory.

The "restart" is simulated by fully shutting down one FastAPI app (lifespan exit closes
the psycopg pool and SQLAlchemy engine) and building a brand-new app + agent instance.

Postgres tests use ``TEST_DATABASE_URL`` (falls back to ``DATABASE_URL``) and skip when
the server is unreachable or ``alembic upgrade head`` has not been run.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from repair_agent.api.main import create_app
from repair_agent.config import settings

MOCK_DOCS = [
    {
        "content": "Burn mark repair guide content.",
        "metadata": {"source": "guide.txt"},
        "score": 0.95,
    }
]
MOCK_DIAGNOSIS = "Mock persisted diagnosis report."


class _MockLLM:
    async def ainvoke(self, messages):
        return type("Response", (), {"content": MOCK_DIAGNOSIS})()


@pytest.fixture
def mocked_pipeline():
    """No network: stub RAG retrieval and the DeepSeek LLM."""
    with patch(
        "repair_agent.agent.nodes.rag_node.aretrieve", new=AsyncMock(return_value=MOCK_DOCS)
    ), patch("repair_agent.agent.nodes.diagnosis_node._get_llm", return_value=_MockLLM()):
        yield


@asynccontextmanager
async def running_app(backend: str, database_url: str = ""):
    """Start an app with its real lifespan, yield an HTTP client, then shut it down."""
    app = create_app(persistence_backend=backend, database_url=database_url)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield app, client


def _postgres_url() -> str:
    return os.environ.get("TEST_DATABASE_URL") or settings.DATABASE_URL


def _postgres_skip_reason(url: str) -> str | None:
    if not url:
        return "TEST_DATABASE_URL / DATABASE_URL not set"
    try:
        import psycopg

        from repair_agent.db.connection import to_psycopg_conninfo

        with psycopg.connect(to_psycopg_conninfo(url), connect_timeout=3) as conn:
            cols = {
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'diagnostic_sessions'"
                ).fetchall()
            }
    except Exception as e:  # pragma: no cover - depends on local infra
        return f"Postgres not reachable at {url.split('@')[-1]}: {e.__class__.__name__}"
    if not {"designator", "correction_mode", "cv_backend", "image_sha256", "detections"} <= cols:
        return "diagnostic_sessions missing Phase E columns — run `alembic upgrade head`"
    return None


@pytest.fixture
def postgres_url() -> str:
    url = _postgres_url()
    reason = _postgres_skip_reason(url)
    if reason:
        pytest.skip(reason)
    return url


async def _diagnose(client: AsyncClient, image: bytes, template: bytes | None = None) -> dict:
    files = {"file": ("test.png", image, "image/png")}
    if template is not None:
        files["template"] = ("template.png", template, "image/png")
    resp = await client.post("/api/v1/diagnose", files=files)
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestPostgresPersistence:
    async def test_session_survives_restart(
        self, postgres_url, mocked_pipeline, burn_mark_image_bytes, clean_image_bytes
    ):
        # --- process 1: diagnose, then shut down completely
        async with running_app("postgres", postgres_url) as (_, client):
            created = await _diagnose(client, burn_mark_image_bytes, template=clean_image_bytes)
            session_id = created["session_id"]
            assert created["persistence_backend"] == "postgres"
            assert created["template_provided"] is True
            before = (await client.get(f"/api/v1/sessions/{session_id}")).json()

        # --- process 2: brand-new app + agent + pool
        async with running_app("postgres", postgres_url) as (app2, client2):
            resp = await client2.get(f"/api/v1/sessions/{session_id}")
            assert resp.status_code == 200, resp.text
            after = resp.json()

            assert after == before
            assert after["session_id"] == session_id
            assert after["status"] == "completed"
            assert after["diagnosis"] == MOCK_DIAGNOSIS == created["diagnosis"]
            assert after["defect_type"] == created["defect_type"]
            assert after["correction_mode"] == created["correction_mode"]
            assert after["cv_backend"] == created["cv_backend"]
            assert after["detections"] == created["detections"]
            assert after["image_sha256"] == hashlib.sha256(burn_mark_image_bytes).hexdigest()
            assert after["template_sha256"] == hashlib.sha256(clean_image_bytes).hexdigest()

            ckpt = after["checkpoint"]
            assert ckpt is not None, "checkpointer lost the thread state"
            assert ckpt["completed"] is True
            assert ckpt["diagnosis"] == MOCK_DIAGNOSIS
            assert ckpt["defect_type"] == created["defect_type"]
            assert ckpt["correction_mode"] == created["correction_mode"]

            # The new agent's checkpointer holds the full thread state (thread_id == id).
            runtime = await app2.state.get_runtime()
            snapshot = await runtime.agent.aget_state(
                {"configurable": {"thread_id": session_id}}
            )
            assert snapshot.values["diagnosis"] == MOCK_DIAGNOSIS
            assert snapshot.values["image_bytes"] == burn_mark_image_bytes
            assert snapshot.values["template_image_bytes"] == clean_image_bytes

    async def test_unknown_session_404(self, postgres_url):
        async with running_app("postgres", postgres_url) as (_, client):
            resp = await client.get("/api/v1/sessions/does-not-exist")
            assert resp.status_code == 404

    async def test_failed_diagnosis_is_recorded(self, postgres_url, burn_mark_image_bytes):
        class _BrokenLLM:
            async def ainvoke(self, messages):
                raise RuntimeError("LLM unavailable")

        with patch(
            "repair_agent.agent.nodes.rag_node.aretrieve", new=AsyncMock(return_value=MOCK_DOCS)
        ), patch("repair_agent.agent.nodes.diagnosis_node._get_llm", return_value=_BrokenLLM()):
            async with running_app("postgres", postgres_url) as (_, client):
                resp = await client.post(
                    "/api/v1/diagnose",
                    files={"file": ("t.png", burn_mark_image_bytes, "image/png")},
                )
                assert resp.status_code == 500
                session_id = resp.headers["X-Session-Id"]
                row = (await client.get(f"/api/v1/sessions/{session_id}")).json()
                assert row["status"] == "failed"
                assert "LLM unavailable" in row["error"]
                assert row["image_sha256"] == hashlib.sha256(burn_mark_image_bytes).hexdigest()
                # The checkpointer kept the partial thread (CV/RAG ran, diagnosis did not).
                ckpt = row["checkpoint"]
                assert ckpt is not None and ckpt["completed"] is False
                assert ckpt["next_nodes"] == ["diagnosis"]


class TestMemoryPersistence:
    async def test_memory_backend_forgets_on_restart(self, mocked_pipeline, burn_mark_image_bytes):
        async with running_app("memory") as (_, client):
            created = await _diagnose(client, burn_mark_image_bytes)
            session_id = created["session_id"]
            assert created["persistence_backend"] == "memory"

            # Same process: row + checkpoint are both available.
            resp = await client.get(f"/api/v1/sessions/{session_id}")
            assert resp.status_code == 200
            body = resp.json()
            assert body["diagnosis"] == MOCK_DIAGNOSIS
            assert body["checkpoint"]["diagnosis"] == MOCK_DIAGNOSIS
            assert body["checkpoint"]["completed"] is True

        # "Restart": memory backend keeps nothing.
        async with running_app("memory") as (_, client2):
            resp = await client2.get(f"/api/v1/sessions/{session_id}")
            assert resp.status_code == 404

    async def test_postgres_without_url_falls_back_to_memory(
        self, mocked_pipeline, burn_mark_image_bytes
    ):
        async with running_app("postgres", "") as (_, client):
            created = await _diagnose(client, burn_mark_image_bytes)
            assert created["persistence_backend"] == "memory"


def test_psycopg_conninfo_strips_sqlalchemy_driver():
    from repair_agent.db.connection import to_async_sqlalchemy_url, to_psycopg_conninfo

    url = "postgresql+asyncpg://u:p%40ss@db.local:5433/app?sslmode=require"
    assert to_psycopg_conninfo(url) == "postgresql://u:p%40ss@db.local:5433/app?sslmode=require"
    assert to_async_sqlalchemy_url("postgresql://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"
