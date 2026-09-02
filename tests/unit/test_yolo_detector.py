"""YOLO loader resolution and CV-node wiring (no Ultralytics required)."""

from pathlib import Path

import pytest

from repair_agent.agent.nodes.cv_node import cv_node
from repair_agent.agent.state import initial_agent_state
from repair_agent.config import settings
from repair_agent.tools.yolo_detector import resolve_weights


def test_resolve_weights_prefers_explicit(tmp_path: Path):
    explicit = tmp_path / "custom.pt"
    explicit.write_bytes(b"x")
    default_root = tmp_path / "data"
    default = default_root / "processed" / "deeppcb" / "weights" / "best.pt"
    default.parent.mkdir(parents=True)
    default.write_bytes(b"y")
    found = resolve_weights(str(explicit), data_dir=default_root)
    assert found == explicit


def test_resolve_weights_falls_back_to_default(tmp_path: Path):
    default_root = tmp_path / "data"
    default = default_root / "processed" / "deeppcb" / "weights" / "best.pt"
    default.parent.mkdir(parents=True)
    default.write_bytes(b"y")
    found = resolve_weights(None, data_dir=default_root)
    assert found == default


def test_resolve_weights_missing(tmp_path: Path):
    assert resolve_weights("/no/such.pt", data_dir=tmp_path) is None


@pytest.mark.asyncio
async def test_cv_node_yolo_backend(monkeypatch, burn_mark_image_bytes):
    monkeypatch.setattr(settings, "CV_BACKEND", "yolo")
    monkeypatch.setattr(
        "repair_agent.agent.nodes.cv_node.resolve_weights",
        lambda explicit=None: Path("/tmp/fake-best.pt"),
    )

    def fake_detect(img, weights_path, conf=0.25):
        assert weights_path.endswith("fake-best.pt")
        return [{"cls": "open", "bbox": (12, 8, 40, 30), "score": 0.91}]

    monkeypatch.setattr("repair_agent.agent.nodes.cv_node.detect_yolo", fake_detect)
    result = await cv_node(initial_agent_state(burn_mark_image_bytes))
    assert result["cv_backend"] == "yolo"
    assert result["defect_type"] == "open"
    assert result["defect_confidence"] == pytest.approx(0.91)
    assert result["defect_bbox"] == (12, 8, 40, 30)
    assert result["detections"][0]["cls"] == "open"


@pytest.mark.asyncio
async def test_cv_node_yolo_falls_back_without_weights(monkeypatch, clean_image_bytes):
    monkeypatch.setattr(settings, "CV_BACKEND", "yolo")
    monkeypatch.setattr(
        "repair_agent.agent.nodes.cv_node.resolve_weights",
        lambda explicit=None: None,
    )
    result = await cv_node(initial_agent_state(clean_image_bytes))
    assert result["defect_type"] == "normal"
    assert result["cv_backend"] == "heuristic"
