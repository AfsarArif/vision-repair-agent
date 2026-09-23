"""Template-verified detection fusion."""

from repair_agent.tools.fusion import verify_with_template


def _det(score, bbox=(10, 10, 20, 20), cls="open"):
    return {"cls": cls, "bbox": bbox, "score": score}


BLOB = [{"cls": None, "bbox": (12, 12, 10, 10), "score": 0.5}]


def test_confident_boxes_kept_without_diff_support():
    fused = verify_with_template([_det(0.9, bbox=(300, 300, 10, 10))], BLOB, op_conf=0.55, keep_conf=0.8)
    assert len(fused) == 1


def test_low_band_box_recovered_when_diff_covers_it():
    fused = verify_with_template([_det(0.45)], BLOB, op_conf=0.55, recover_conf=0.4, keep_conf=0.8)
    assert fused and fused[0]["template_verified"] is True


def test_mid_band_box_dropped_without_diff_support():
    fused = verify_with_template(
        [_det(0.6, bbox=(300, 300, 10, 10))], BLOB, op_conf=0.55, keep_conf=0.8
    )
    assert fused == []


def test_no_blobs_keeps_operating_point_boxes():
    fused = verify_with_template([_det(0.6), _det(0.45)], [], op_conf=0.55, keep_conf=0.8)
    assert [d["score"] for d in fused] == [0.6]
