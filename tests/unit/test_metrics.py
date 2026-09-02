"""Pure metric helpers."""

from repair_agent.eval.metrics import (
    average_precision_at_iou,
    exact_match,
    iou_xyxy,
    mean_reciprocal_rank,
    recall_at_k,
)


def test_iou_identical():
    box = (0, 0, 10, 10)
    assert iou_xyxy(box, box) == 1.0


def test_iou_no_overlap():
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_ap_perfect_match():
    preds = [{"bbox": (0, 0, 10, 10), "cls": "open", "score": 0.9}]
    gts = [{"bbox": (0, 0, 10, 10), "cls": "open"}]
    assert average_precision_at_iou(preds, gts) == 1.0


def test_recall_at_k():
    assert recall_at_k(["adapter-open", "adapter-short"], ["adapter-open"], k=5) == 1.0
    assert recall_at_k(["adapter-short"], ["adapter-open"], k=5) == 0.0


def test_mrr():
    assert mean_reciprocal_rank(["a", "b"], ["b"]) == 0.5


def test_exact_match_normalizes():
    assert exact_match("r12", "R12") is True
    assert exact_match("C3", "U1") is False
