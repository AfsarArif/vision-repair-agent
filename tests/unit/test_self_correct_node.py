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


@pytest.mark.asyncio
async def test_self_correct_verifies_yolo_candidates_against_template():
    template = make_trace_template()
    test, gold = apply_open(template)
    x1, y1, x2, y2 = gold
    on_defect = {"cls": "open", "bbox": (x1, y1, x2 - x1, y2 - y1), "score": 0.45}
    off_defect = {"cls": "spur", "bbox": (0, 0, 8, 8), "score": 0.6}
    state = initial_agent_state(encode_png(test), template_image_bytes=encode_png(template))
    state["candidate_detections"] = [on_defect, off_defect]
    result = await self_correct_node(state)
    assert "template_verify" in result["correction_mode"]
    assert [d["cls"] for d in result["detections"]] == ["open"]
    assert result["defect_type"] == "open"
