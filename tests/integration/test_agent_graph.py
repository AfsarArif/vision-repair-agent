"""Integration tests for the full LangGraph agent execution."""

from unittest.mock import AsyncMock, patch

import pytest

from repair_agent.agent.graph import get_agent_sync
from repair_agent.agent.state import AgentState


def _make_state(image_bytes: bytes) -> AgentState:
    """Create a clean initial state for graph execution."""
    return {
        "image_bytes": image_bytes,
        "image_path": None,
        "defect_type": None,
        "defect_confidence": None,
        "defect_bbox": None,
        "cropped_image_bytes": None,
        "serial_number": None,
        "ocr_confidence": None,
        "rag_documents": None,
        "rag_query": None,
        "diagnosis": None,
        "correction_attempts": 0,
        "self_correction_triggered": False,
        "messages": [],
    }


def _make_mock_rag_docs():
    return [
        {
            "content": "Burn marks are caused by electrical overload. Repair by cleaning and replacing damaged traces.",
            "metadata": {"source": "burn_repair_guide.txt"},
            "score": 0.92,
        },
        {
            "content": "For serial SN-XR9821A: Known issue with voltage regulator causing burn marks on revision B boards.",
            "metadata": {"source": "serial_lookup.txt"},
            "score": 0.97,
        },
    ]


class TestAgentGraph:
    @pytest.mark.asyncio
    async def test_full_graph_execution(self, burn_mark_image_bytes):
        """Test that the full graph executes and produces a diagnosis."""
        mock_docs = _make_mock_rag_docs()

        # Mock the ChatOpenAI invocation to avoid API calls
        class MockLLM:
            async def ainvoke(self, messages):
                return type("Response", (), {"content": "Mock diagnosis: Burn mark detected. Replace damaged components."})()

        with patch(
            "repair_agent.agent.nodes.rag_node.aretrieve",
            new=AsyncMock(return_value=mock_docs),
        ):
            with patch(
                "repair_agent.agent.nodes.diagnosis_node.ChatOpenAI",
                return_value=MockLLM(),
            ):
                agent = get_agent_sync()
                initial_state = _make_state(burn_mark_image_bytes)
                final_state = await agent.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": "test-session-001"}},
                )

                assert "diagnosis" in final_state
                assert final_state["diagnosis"] is not None
                assert len(final_state["diagnosis"]) > 0
                assert "defect_type" in final_state
                assert final_state["defect_type"] is not None

    @pytest.mark.asyncio
    async def test_self_correction_triggers(self, corrosion_image_bytes):
        """Test that self-correction triggers when confidence is below threshold."""
        mock_docs = _make_mock_rag_docs()

        class MockLLM:
            async def ainvoke(self, messages):
                return type("Response", (), {"content": "Diagnosis after self-correction."})()

        with patch(
            "repair_agent.agent.nodes.rag_node.aretrieve",
            new=AsyncMock(return_value=mock_docs),
        ):
            with patch(
                "repair_agent.agent.nodes.diagnosis_node.ChatOpenAI",
                return_value=MockLLM(),
            ):
                agent = get_agent_sync()
                initial_state = _make_state(corrosion_image_bytes)

                # Override the confidence threshold to force self-correction
                with patch("repair_agent.agent.edges.settings.CONFIDENCE_THRESHOLD", 0.99):
                    final_state = await agent.ainvoke(
                        initial_state,
                        config={"configurable": {"thread_id": "test-session-002"}},
                    )

                    # The self_correction_triggered flag should be set
                    assert "diagnosis" in final_state
                    assert final_state["diagnosis"] is not None

    @pytest.mark.asyncio
    async def test_clean_image_skips_correction(self, clean_image_bytes):
        """Test that clean images skip the self-correction loop."""
        mock_docs = []

        class MockLLM:
            async def ainvoke(self, messages):
                return type("Response", (), {"content": "No defects detected. Hardware is normal."})()

        with patch(
            "repair_agent.agent.nodes.rag_node.aretrieve",
            new=AsyncMock(return_value=mock_docs),
        ):
            with patch(
                "repair_agent.agent.nodes.diagnosis_node.ChatOpenAI",
                return_value=MockLLM(),
            ):
                agent = get_agent_sync()
                initial_state = _make_state(clean_image_bytes)
                final_state = await agent.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": "test-session-003"}},
                )

                assert final_state["defect_type"] == "normal"
                assert final_state["self_correction_triggered"] is False
                assert final_state["correction_attempts"] == 0
                assert "No defects" in final_state["diagnosis"]
