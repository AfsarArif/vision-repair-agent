"""Canonical PCB fabrication-defect classes shared by CV, RAG, and eval."""

from typing import Literal

DefectClass = Literal[
    "open",
    "short",
    "mousebite",
    "spur",
    "spurious_copper",
    "pin_hole",
    "missing_hole",
    "normal",
]

CANONICAL_CLASSES: tuple[DefectClass, ...] = (
    "open",
    "short",
    "mousebite",
    "spur",
    "spurious_copper",
    "pin_hole",
    "missing_hole",
    "normal",
)

DETECTION_CLASSES: tuple[str, ...] = tuple(c for c in CANONICAL_CLASSES if c != "normal")

# DeepPCB annotation `type` is 1-based (0 is unused background).
DEEPPCB_ID_TO_CLASS: dict[int, str] = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "spurious_copper",
    6: "pin_hole",
}

CLASS_TO_DEEPPCB_ID: dict[str, int] = {v: k for k, v in DEEPPCB_ID_TO_CLASS.items()}

YOLO_CLASS_NAMES: tuple[str, ...] = (
    "open",
    "short",
    "mousebite",
    "spur",
    "spurious_copper",
    "pin_hole",
)

PKU_NAME_TO_CLASS: dict[str, str] = {
    "open_circuit": "open",
    "open circuit": "open",
    "short": "short",
    "mouse_bite": "mousebite",
    "mouse bite": "mousebite",
    "spur": "spur",
    "spurious_copper": "spurious_copper",
    "spurious copper": "spurious_copper",
    "missing_hole": "missing_hole",
    "missing hole": "missing_hole",
}

RETIRED_CLASSES: frozenset[str] = frozenset(
    {"burn_mark", "crack", "corrosion", "delamination"}
)


def is_canonical(name: str) -> bool:
    return name in CANONICAL_CLASSES


def deeppcb_id_to_class(type_id: int) -> str:
    """Map a DeepPCB type integer to a canonical class name."""
    if type_id not in DEEPPCB_ID_TO_CLASS:
        raise ValueError(f"Unknown DeepPCB type id: {type_id}")
    return DEEPPCB_ID_TO_CLASS[type_id]


def yolo_index_to_class(index: int) -> str:
    """Map a 0-based YOLO class index to a canonical name."""
    if index < 0 or index >= len(YOLO_CLASS_NAMES):
        raise ValueError(f"Unknown YOLO class index: {index}")
    return YOLO_CLASS_NAMES[index]


def class_to_yolo_index(name: str) -> int:
    """Map a canonical DeepPCB class to a 0-based YOLO index."""
    try:
        return YOLO_CLASS_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"Class {name!r} is not a DeepPCB YOLO class") from exc
