"""eval_ocr on synthetic gold (smoke only; NOT the FPIC gate)."""

import json

import pytest

from repair_agent.eval.ocr import eval_ocr, load_gold, write_synthetic_gold
from repair_agent.tools.ocr_tools import tesseract_available


def test_load_gold_filters_subset_and_family(tmp_path):
    gold = tmp_path / "g.jsonl"
    rows = [
        {"image": "a.png", "designator": "R1", "split": "fpic_ocr", "subset": "dev"},
        {"image": "b.png", "designator": "C2", "split": "fpic_ocr", "subset": "test"},
    ]
    gold.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert [r["image"] for r in load_gold(gold, "test")] == ["b.png"]
    assert [r["image"] for r in load_gold(gold, "dev")] == ["a.png"]
    assert len(load_gold(gold, "fpic_ocr")) == 2
    assert len(load_gold(gold, None)) == 2


def test_eval_ocr_refuses_deeppcb(tmp_path):
    gold = tmp_path / "g.jsonl"
    gold.write_text(json.dumps({"image": "x.jpg", "designator": "R1", "split": "deeppcb_test"}) + "\n")
    with pytest.raises(ValueError, match="DeepPCB"):
        eval_ocr(gold, split=None)


def test_eval_ocr_unknown_engine(tmp_path):
    gold = tmp_path / "g.jsonl"
    gold.write_text("")
    with pytest.raises(ValueError, match="engine"):
        eval_ocr(gold, engine="nope")


@pytest.mark.skipif(not tesseract_available(), reason="tesseract binary not installed")
def test_eval_ocr_synthetic_smoke(tmp_path):
    gold = write_synthetic_gold(tmp_path, n=6, seed=3, subset="dev")
    report = eval_ocr(gold, split="dev")
    assert report["stage"] == "ocr"
    assert report["n"] == 6
    assert report["engine"] == "tesseract"
    assert report["split"] == "dev"
    assert report["dataset"] == "synthetic"
    assert report["passes_gate"] is None  # synthetic never counts as the gate
    assert 0.0 <= report["exact_match"] <= 1.0
