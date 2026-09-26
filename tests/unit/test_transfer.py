"""Transfer eval: preprocess, tiling, and scoring with a fake detector (no weights)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from repair_agent.eval.transfer import (
    PREPROCESS_VARIANTS,
    SHARED_CLASSES,
    binarize,
    eval_transfer,
    eval_transfer_variants,
    nms_merge,
    parse_preprocess,
    predict_image,
    tile_origins,
)


def _gold(tmp_path: Path, size=(900, 1500)) -> Path:
    h, w = size
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 1] = 60
    img[100:200, 100:800, 1] = 200  # bright "copper" trace on dark mask
    path = tmp_path / "a.jpg"
    cv2.imwrite(str(path), img)
    row = {
        "image_id": "a",
        "image": str(path),
        "split": "pku_holdout",
        "boxes": [
            {"bbox": [1000, 700, 1040, 740], "cls": "open"},
            {"bbox": [100, 100, 140, 140], "cls": "spur"},
            {"bbox": [300, 300, 340, 340], "cls": "missing_hole"},
        ],
    }
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return gold


def test_parse_preprocess():
    assert parse_preprocess("none") == ("none", False)
    assert parse_preprocess("tile") == ("none", True)
    assert parse_preprocess("binarize_tile") == ("binarize", True)
    assert parse_preprocess("binarize_gray_tile") == ("binarize_gray", True)
    for v in PREPROCESS_VARIANTS:
        parse_preprocess(v)
    with pytest.raises(ValueError):
        parse_preprocess("sharpen")


def test_binarize_copper_black_substrate_white():
    img = np.zeros((40, 40, 3), np.uint8)
    img[:, :, 1] = 50
    img[10:30, 10:30, 1] = 210
    out = binarize(img)
    assert out.shape == (40, 40, 3) and out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}
    assert out[20, 20, 0] == 0  # copper -> black (DeepPCB polarity)
    assert out[2, 2, 0] == 255  # substrate -> white


def test_tile_origins_cover_edges():
    assert tile_origins(500, 640) == [0]
    starts = tile_origins(1586, 640, 128)
    assert starts[0] == 0 and starts[-1] == 1586 - 640
    assert all(b - a <= 640 for a, b in zip(starts, starts[1:]))


def test_nms_merge_is_class_aware():
    dets = [
        {"bbox": (0, 0, 10, 10), "cls": "open", "score": 0.9},
        {"bbox": (1, 0, 11, 10), "cls": "open", "score": 0.5},
        {"bbox": (1, 0, 11, 10), "cls": "short", "score": 0.4},
    ]
    kept = nms_merge(dets)
    assert [(d["cls"], d["score"]) for d in kept] == [("open", 0.9), ("short", 0.4)]


def test_predict_image_tiling_returns_full_image_coords():
    img = np.zeros((900, 1500, 3), np.uint8)
    calls = []

    def fake(crop, weights, conf=0.25):
        calls.append(crop.shape)
        # One detection at a fixed tile-local spot.
        return [{"cls": "open", "bbox": (10, 20, 30, 40), "score": 0.8}]

    preds = predict_image(img, fake, "w.pt", 0.25, "tile")
    assert all(s[0] <= 640 and s[1] <= 640 for s in calls)
    assert len(calls) == len(tile_origins(900)) * len(tile_origins(1500))
    boxes = {tuple(p["bbox"]) for p in preds}
    last_x, last_y = tile_origins(1500)[-1], tile_origins(900)[-1]
    assert (10.0, 20.0, 40.0, 60.0) in boxes
    assert (last_x + 10.0, last_y + 20.0, last_x + 40.0, last_y + 60.0) in boxes


def test_eval_transfer_scores_shared_classes_only(tmp_path: Path):
    gold = _gold(tmp_path)

    def fake(img, weights, conf=0.25):
        return [
            {"cls": "open", "bbox": (1000, 700, 40, 40), "score": 0.9},  # TP
            {"cls": "short", "bbox": (500, 500, 30, 30), "score": 0.6},  # FP
            {"cls": "pin_hole", "bbox": (300, 300, 40, 40), "score": 0.7},  # out of vocab
        ]

    report = eval_transfer(gold, "fake.pt", preprocess="none", detector=fake)
    assert report["stage"] == "transfer"
    assert report["n_images"] == 1
    assert report["preprocess"] == "none"
    assert set(report["per_class"]) == set(SHARED_CLASSES)
    assert report["per_class"]["open"]["f1"] == 1.0
    assert report["per_class"]["short"]["fp"] == 1
    assert report["per_class"]["spur"]["fn"] == 1
    # open=1, short/spur/mousebite/spurious_copper=0 -> 1/5
    assert report["macro_f1"] == pytest.approx(0.2)
    assert report["missing_hole"]["n_gold"] == 1
    assert report["missing_hole"]["recall"] == 0.0
    assert report["pin_hole_predictions"]["n_pred"] == 1
    # class-agnostic: open + pin_hole box on missing_hole are hits
    assert report["class_agnostic"]["tp"] == 2


def test_eval_transfer_variants_runs_all(tmp_path: Path):
    gold = _gold(tmp_path)

    def fake(img, weights, conf=0.25):
        assert img.ndim == 3
        return []

    out = eval_transfer_variants(gold, "fake.pt", detector=fake)
    assert [r["preprocess"] for r in out["variants"]] == list(PREPROCESS_VARIANTS)
    assert all(r["macro_f1"] == 0.0 for r in out["variants"])


def test_eval_transfer_missing_weights_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        eval_transfer(_gold(tmp_path), str(tmp_path / "nope.pt"))


def test_predict_image_uses_batch_fast_path():
    img = np.zeros((700, 700, 3), np.uint8)
    seen = {}

    def fake(img, weights, conf=0.25):  # pragma: no cover - must not be called
        raise AssertionError("per-image path used")

    def batch(images, weights, conf=0.25):
        seen["n"] = len(images)
        return [[{"cls": "spur", "bbox": (0, 0, 5, 5), "score": 0.5}] for _ in images]

    fake.batch = batch
    preds = predict_image(img, fake, "w.pt", 0.25, "binarize_tile")
    assert seen["n"] == 4
    assert len(preds) == 4  # distinct tile origins, no overlap between boxes
