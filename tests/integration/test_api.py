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

        with patch(
            "repair_agent.api.routes.diagnose.aretrieve",
            new=AsyncMock(return_value=mock_docs),
        ):
            # Need to patch the import path used in the routes module
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
