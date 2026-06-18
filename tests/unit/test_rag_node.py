"""Unit tests for the RAG node."""

from unittest.mock import AsyncMock, patch

import pytest

from repair_agent.agent.nodes.rag_node import rag_node, _build_query


class TestRAGNodeQuery:
    def test_build_query_without_serial(self):
        state = {
            "defect_type": "burn_mark",
            "serial_number": None,
        }
        query = _build_query(state)
        assert "burn_mark" in query
        assert "serial" not in query.lower()

    def test_build_query_with_serial(self):
        state = {
            "defect_type": "crack",
            "serial_number": "SN-ABC12345",
        }
        query = _build_query(state)
        assert "crack" in query
        assert "SN-ABC12345" in query

    def test_build_query_default_defect(self):
        state = {
            "defect_type": None,
            "serial_number": None,
        }
        query = _build_query(state)
        assert "unknown defect" in query.lower()


class TestRAGNode:
    @pytest.mark.asyncio
    async def test_rag_node_returns_documents(self):
        """Test that RAG node correctly formats its return values."""
        mock_docs = [
            {
                "content": "Test document content about burn marks.",
                "metadata": {"source": "test_guide.txt"},
                "score": 0.95,
            },
        ]

        state = {
            "defect_type": "burn_mark",
            "serial_number": None,
        }

        with patch(
            "repair_agent.agent.nodes.rag_node.aretrieve", new=AsyncMock(return_value=mock_docs)
        ):
            result = await rag_node(state)
            assert "rag_query" in result
            assert "rag_documents" in result
            assert result["rag_query"] is not None
            assert len(result["rag_documents"]) == 1
            assert result["rag_documents"][0]["content"] == mock_docs[0]["content"]

    @pytest.mark.asyncio
    async def test_rag_node_with_serial_number(self):
        mock_docs = [
            {
                "content": "Diagnostic procedure for SN-ABC12345 with crack defect.",
                "metadata": {"source": "serial_guide.txt"},
                "score": 0.98,
            },
        ]

        state = {
            "defect_type": "crack",
            "serial_number": "SN-ABC12345",
        }

        with patch(
            "repair_agent.agent.nodes.rag_node.aretrieve", new=AsyncMock(return_value=mock_docs)
        ):
            result = await rag_node(state)
            assert "SN-ABC12345" in result["rag_query"]
            assert "crack" in result["rag_query"]
            assert len(result["rag_documents"]) == 1

    @pytest.mark.asyncio
    async def test_rag_node_empty_results(self):
        state = {
            "defect_type": "unknown_defect",
            "serial_number": None,
        }

        with patch(
            "repair_agent.agent.nodes.rag_node.aretrieve", new=AsyncMock(return_value=[])
        ):
            result = await rag_node(state)
            assert result["rag_documents"] == []
            assert result["rag_query"] is not None
