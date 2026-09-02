"""Self-correct node tests."""

import pytest

from repair_agent.agent.nodes.self_correct_node import self_correct_node
from repair_agent.agent.state import initial_agent_state
from repair_agent.eval.synthetic import apply_open, encode_png, make_trace_template


@pytest.mark.asyncio
async def test_self_correct_uses_template_diff():
    template = make_trace_template()
    test, _gold = apply_open(template)
    state = initial_agent_state(encode_png(test), template_image_bytes=encode_png(template))
    result = await self_correct_node(state)
    assert result["self_correction_triggered"] is True
    assert result["correction_attempts"] == 1
    assert "template_diff" in (result.get("correction_mode") or "")
    assert result.get("defect_bbox") is not None
