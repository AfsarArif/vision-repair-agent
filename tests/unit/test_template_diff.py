"""Template-diff localization on synthetic PCB pairs."""

from repair_agent.eval.metrics import box_xywh_to_xyxy, iou_xyxy
from repair_agent.eval.synthetic import apply_open, apply_short, make_trace_template
from repair_agent.tools.template_diff import localize_defects


def _best_iou(test, template, gold_xyxy) -> float:
    detections = localize_defects(test, template, min_area=8, diff_threshold=20)
    best = 0.0
    for det in detections:
        best = max(best, iou_xyxy(box_xywh_to_xyxy(tuple(det["bbox"])), gold_xyxy))
    return best


def test_template_diff_finds_open():
    template = make_trace_template()
    test, gold = apply_open(template)
    assert _best_iou(test, template, gold) >= 0.3


def test_template_diff_finds_short():
    template = make_trace_template()
    test, gold = apply_short(template)
    assert _best_iou(test, template, gold) >= 0.3


def test_identical_images_have_no_defects():
    template = make_trace_template()
    detections = localize_defects(template, template, min_area=20, diff_threshold=30)
    assert detections == []
