"""Tests for DeepPCB pair discovery, official split, and YOLO export layout."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from repair_agent.data.deeppcb import (
    export_splits,
    find_pairs,
    normalize_list_stem,
    parse_annotation,
    split_pairs,
    to_yolo_line,
)


def _write_jpeg(path: Path, color: tuple[int, int, int] = (0, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path, format="JPEG")


def _tiny_deeppcb(root: Path, n_train: int = 8, n_test: int = 4) -> Path:
    pcb = root / "PCBData"
    img_dir = pcb / "group00041" / "00041"
    ann_dir = pcb / "group00041" / "00041_not"
    train_lines = []
    test_lines = []
    for i in range(n_train + n_test):
        stem = f"00041{i:03d}"
        _write_jpeg(img_dir / f"{stem}_test.jpg", (255, 255, 255))
        _write_jpeg(img_dir / f"{stem}_temp.jpg", (0, 0, 0))
        (ann_dir).mkdir(parents=True, exist_ok=True)
        (ann_dir / f"{stem}.txt").write_text("10 10 30 22 1\n", encoding="utf-8")
        rel_img = f"group00041/00041/{stem}.jpg"
        rel_ann = f"group00041/00041_not/{stem}.txt"
        line = f"{rel_img} {rel_ann}\n"
        if i < n_train:
            train_lines.append(line)
        else:
            test_lines.append(line)
    (pcb / "trainval.txt").write_text("".join(train_lines), encoding="utf-8")
    (pcb / "test.txt").write_text("".join(test_lines), encoding="utf-8")
    return pcb


def test_normalize_list_stem_strips_jpg_without_test_suffix():
    assert normalize_list_stem("group20085/20085/20085000.jpg") == "20085000"
    assert normalize_list_stem("00041000_test.jpg") == "00041000"


def test_parse_annotation_space_and_comma(tmp_path: Path):
    path = tmp_path / "a.txt"
    path.write_text("10 20 40 50 1\n1,2,8,9,3\n0 0 1 1 0\n", encoding="utf-8")
    boxes = parse_annotation(path)
    assert boxes[0] == {"bbox": [10, 20, 40, 50], "cls": "open"}
    assert boxes[1]["cls"] == "mousebite"
    assert len(boxes) == 2


def test_to_yolo_uses_actual_image_size():
    box = {"bbox": [0, 0, 32, 24], "cls": "open"}
    line = to_yolo_line(box, img_w=64, img_h=48)
    idx, cx, cy, w, h = line.split()
    assert idx == "0"
    assert float(cx) == 0.25
    assert float(cy) == 0.25
    assert float(w) == 0.5
    assert float(h) == 0.5


def test_find_pairs_uses_not_folder(tmp_path: Path):
    _tiny_deeppcb(tmp_path, n_train=3, n_test=2)
    pairs = find_pairs(tmp_path)
    assert len(pairs) == 5
    test, template, ann = pairs[0]
    assert test.name.endswith("_test.jpg")
    assert template.name.endswith("_temp.jpg")
    assert any(part.endswith("_not") for part in ann.parts)


def test_official_split_does_not_mix_test_into_train(tmp_path: Path):
    _tiny_deeppcb(tmp_path, n_train=8, n_test=4)
    pairs = find_pairs(tmp_path)
    splits = split_pairs(pairs, raw_root=tmp_path)
    train_stems = {p[0].name for p in splits["train"] + splits["val"]}
    test_stems = {p[0].name for p in splits["test"]}
    assert train_stems.isdisjoint(test_stems)
    assert len(splits["test"]) == 4
    assert len(splits["train"]) + len(splits["val"]) == 8
    assert splits["val"]


def test_export_keeps_templates_out_of_train_images(tmp_path: Path):
    raw = tmp_path / "raw"
    _tiny_deeppcb(raw, n_train=8, n_test=4)
    pairs = find_pairs(raw)
    splits = split_pairs(pairs, raw_root=raw)
    out = tmp_path / "processed"
    gold = export_splits(splits, out)
    train_images = list((out / "images" / "train").glob("*"))
    assert train_images
    assert not any(p.name.endswith("_temp.jpg") for p in train_images)
    assert list((out / "templates" / "train").glob("*_temp.jpg"))
    assert gold.is_file()
    assert (out / "deeppcb.yaml").is_file()
    label = next((out / "labels" / "train").glob("*.txt"))
    body = label.read_text(encoding="utf-8").strip()
    assert body.startswith("0 ")
