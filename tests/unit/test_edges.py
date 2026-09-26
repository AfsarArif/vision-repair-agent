"""Conditional edge routing."""

from repair_agent.agent.edges import correction_complete, should_self_correct
from repair_agent.config import settings


def test_gate_reads_min_confidence_not_top_score():
    state = {"defect_confidence": 0.97, "min_confidence": 0.5, "correction_attempts": 0}
    assert should_self_correct(state) == "self_correct"


def test_gate_skips_when_all_boxes_confident():
    state = {"defect_confidence": 0.97, "min_confidence": settings.CONFIDENCE_THRESHOLD + 0.05}
    assert should_self_correct(state) == "diagnose"


def test_gate_falls_back_to_defect_confidence():
    assert should_self_correct({"defect_confidence": 0.1, "correction_attempts": 0}) == "self_correct"


def test_template_verification_does_not_retry():
    state = {"correction_attempts": 1, "correction_mode": "template_verify", "serial_number": None}
    assert correction_complete(state) == "diagnose"
