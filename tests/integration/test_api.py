"""Integration tests for the FastAPI endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from repair_agent.api.main import app
from repair_agent.api.schemas import DiagnoseResponse


@pytest.fixture
def async_client():
    """Create an async httpx client for testing the FastAPI app."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, async_client):
        response = await async_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"


class TestDiagnoseEndpoint:
    @pytest.mark.asyncio
    async def test_diagnose_with_valid_image(self, async_client, burn_mark_image_bytes):
        """Test POST /diagnose with a valid image file."""
        mock_docs = [
            {
                "content": "Burn mark repair guide content.",
                "metadata": {"source": "guide.txt"},
                "score": 0.95,
            },
        ]

        class MockLLM:
            async def ainvoke(self, messages):
                return type("Response", (), {"content": "Mock diagnosis report."})()

        with patch("repair_agent.agent.nodes.rag_node.aretrieve", new=AsyncMock(return_value=mock_docs)):
            with patch(
                "repair_agent.agent.nodes.diagnosis_node.ChatOpenAI",
                return_value=MockLLM(),
            ):
                response = await async_client.post(
                    "/api/v1/diagnose",
                    files={"file": ("test.png", burn_mark_image_bytes, "image/png")},
                )
                assert response.status_code == 200
                data = response.json()
                assert "session_id" in data
                assert "diagnosis" in data
                assert len(data["diagnosis"]) > 0

    @pytest.mark.asyncio
    async def test_diagnose_invalid_file_type(self, async_client):
        """Test POST /diagnose with an invalid file type returns 400."""
        response = await async_client.post(
            "/api/v1/diagnose",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_diagnose_empty_file(self, async_client):
        """Test POST /diagnose with an empty file returns 400."""
        response = await async_client.post(
            "/api/v1/diagnose",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_diagnose_no_file(self, async_client):
        """Test POST /diagnose without a file returns 422."""
        response = await async_client.post("/api/v1/diagnose")
        assert response.status_code == 422


class _MockLLM:
    async def ainvoke(self, messages):
        return type("Response", (), {"content": "Mock diagnosis report."})()


_MOCK_DOCS = [
    {"content": "Open circuit guide.", "metadata": {"source": "guide.txt"}, "score": 0.9},
]


class TestTemplateUploadAndSessions:
    @pytest.mark.asyncio
    async def test_diagnose_with_template_then_get_session(
        self, async_client, burn_mark_image_bytes, clean_image_bytes
    ):
        import hashlib

        captured = {}
        from repair_agent.agent.state import initial_agent_state as real_initial

        def spy(image_bytes, template_image_bytes=None, image_path=None):
            captured["template"] = template_image_bytes
            return real_initial(image_bytes, template_image_bytes=template_image_bytes)

        with patch(
            "repair_agent.agent.nodes.rag_node.aretrieve", new=AsyncMock(return_value=_MOCK_DOCS)
        ), patch(
            "repair_agent.agent.nodes.diagnosis_node._get_llm", return_value=_MockLLM()
        ), patch("repair_agent.api.routes.diagnose.initial_agent_state", side_effect=spy):
            resp = await async_client.post(
                "/api/v1/diagnose",
                files={
                    "file": ("test.png", burn_mark_image_bytes, "image/png"),
                    "template": ("template.png", clean_image_bytes, "image/png"),
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["template_provided"] is True
        assert captured["template"] == clean_image_bytes

        got = await async_client.get(f"/api/v1/sessions/{data['session_id']}")
        assert got.status_code == 200
        body = got.json()
        assert body["session_id"] == data["session_id"]
        assert body["diagnosis"] == data["diagnosis"]
        assert body["template_sha256"] == hashlib.sha256(clean_image_bytes).hexdigest()
        assert body["checkpoint"]["completed"] is True
        assert body["checkpoint"]["diagnosis"] == data["diagnosis"]

    @pytest.mark.asyncio
    async def test_diagnose_invalid_template_type(self, async_client, burn_mark_image_bytes):
        resp = await async_client.post(
            "/api/v1/diagnose",
            files={
                "file": ("test.png", burn_mark_image_bytes, "image/png"),
                "template": ("t.txt", b"nope", "text/plain"),
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_get_unknown_session_404(self, async_client):
        resp = await async_client.get("/api/v1/sessions/not-a-real-session")
        assert resp.status_code == 404
