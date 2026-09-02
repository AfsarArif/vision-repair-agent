"""Unit tests for canonical defect taxonomy."""

import pytest

from repair_agent.taxonomy import (
    CANONICAL_CLASSES,
    class_to_yolo_index,
    deeppcb_id_to_class,
    yolo_index_to_class,
)


def test_deeppcb_ids_cover_six_classes():
    names = [deeppcb_id_to_class(i) for i in range(1, 7)]
    assert names == [
        "open",
        "short",
        "mousebite",
        "spur",
        "spurious_copper",
        "pin_hole",
    ]


def test_yolo_roundtrip():
    for name in ("open", "pin_hole"):
        assert yolo_index_to_class(class_to_yolo_index(name)) == name


def test_unknown_deeppcb_id():
    with pytest.raises(ValueError):
        deeppcb_id_to_class(0)


def test_normal_is_canonical_not_yolo():
    assert "normal" in CANONICAL_CLASSES
    with pytest.raises(ValueError):
        class_to_yolo_index("normal")
