"""Designator OCR tools (Phase D) on PIL-rendered synthetic silkscreen crops."""

import numpy as np
import pytest

from repair_agent.eval.ocr import render_designator_crop
from repair_agent.tools.ocr_tools import (
    designator_variants,
    normalize_designator,
    read_designator,
    tesseract_available,
)

needs_tesseract = pytest.mark.skipif(not tesseract_available(), reason="tesseract binary not installed")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("R12", "R12"),
        ("r12", "R12"),
        ("C1O", "C10"),
        ("UI7", "U17"),
        ("Ul", "U1"),
        ("L5S", "L55"),
        ("R 1 2", "R12"),
        ("R12.", "R12"),
        ("0I", None),
        ("X12", None),
        ("R", None),
        ("R12345", None),
        ("", None),
    ],
)
def test_normalize_designator(raw, expected):
    assert normalize_designator(raw) == expected


def test_variants_are_dark_on_white_and_cover_rotations():
    crop = render_designator_crop("R12", font_size=14)
    variants = list(designator_variants(crop, rotations=(0, 90, 270)))
    names = [n for n, _ in variants]
    assert names == ["r0_pos", "r0_neg", "r90_pos", "r90_neg", "r270_pos", "r270_neg"]
    first = variants[0][1]
    assert first.dtype == np.uint8 and first.ndim == 2
    # Light-on-dark silkscreen gets flipped: border (background) is white.
    assert first[0].mean() > 200
    # Upscaled to a Tesseract-friendly height.
    assert min(first.shape) >= 48


def test_variants_empty_crop():
    assert list(designator_variants(np.zeros((0, 0, 3), np.uint8))) == []


@needs_tesseract
@pytest.mark.parametrize("text", ["R12", "C47", "U3", "Q105"])
def test_read_designator_horizontal(text):
    crop = render_designator_crop(text, font_size=22, seed=1)
    pred, conf = read_designator(crop)
    assert pred == text
    assert 0.0 < conf <= 1.0


@needs_tesseract
@pytest.mark.parametrize("rotation", [90, 270])
def test_read_designator_rotated(rotation):
    crop = render_designator_crop("C33", font_size=22, rotation=rotation, board=(15, 40, 110))
    pred, _ = read_designator(crop)
    assert pred == "C33"


@needs_tesseract
def test_read_designator_dark_on_light():
    crop = render_designator_crop("R8", font_size=22, board=(235, 235, 235), ink=(20, 20, 20))
    pred, _ = read_designator(crop)
    assert pred == "R8"


@needs_tesseract
def test_read_designator_blank_returns_none():
    blank = np.full((20, 50, 3), (20, 90, 30), np.uint8)
    assert read_designator(blank) == (None, 0.0)
