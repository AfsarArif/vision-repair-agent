"""FPIC OCR gold prep on a tiny fake FPIC layout (no download)."""

import csv
import json

import cv2
import numpy as np

from repair_agent.data.fpic import (
    assign_subsets,
    export_fpic,
    is_designator,
    iter_text_boxes,
    ocr_annotation_csvs,
    parse_vertices,
)


def _write_board(raw, stem, rows, header=None):
    (raw / "pcb_image").mkdir(parents=True, exist_ok=True)
    (raw / "ocr_annotation").mkdir(parents=True, exist_ok=True)
    img = np.full((200, 300, 3), (30, 90, 20), np.uint8)
    cv2.imwrite(str(raw / "pcb_image" / f"{stem}.png"), img)
    header = header or ["Instance ID", "Vertices", "Text", "Class", "Logo", "Orientation", "Notes"]
    with (raw / "ocr_annotation" / f"{stem}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def test_is_designator():
    assert is_designator("R12")
    assert is_designator(" c3 ")
    assert is_designator("W1234")
    assert not is_designator("PWR")
    assert not is_designator("R12345")
    assert not is_designator("C4V")
    assert not is_designator("")


def test_parse_vertices_formats():
    assert parse_vertices("[[10, 20], [30, 20], [30, 40], [10, 40]]") == [
        (10, 20), (30, 20), (30, 40), (10, 40)
    ]
    assert parse_vertices("[[[1 2]\n [3 4]]]") == [(1, 2), (3, 4)]
    assert parse_vertices("") == []
    assert parse_vertices("[5]") == []


def test_iter_text_boxes_case_insensitive(tmp_path):
    _write_board(
        tmp_path, "s1_front",
        [["0", "[[10,10],[40,10],[40,25],[10,25]]", "R1", "Board", "", "90", ""]],
        header=["instance id", "vertices", "text", "class", "logo", "orientation", "notes"],
    )
    (box,) = list(iter_text_boxes(tmp_path / "ocr_annotation" / "s1_front.csv"))
    assert box.text == "R1" and box.orientation == 90 and box.bbox == (10, 10, 40, 25)


def test_assign_subsets_deterministic_and_grouped():
    sources = [f"img{i}.png" for i in range(20)] * 3
    a = assign_subsets(sources, dev_frac=0.2, seed=0)
    b = assign_subsets(reversed(sources), dev_frac=0.2, seed=0)
    assert a == b
    assert sum(v == "dev" for v in a.values()) == 4
    assert set(a.values()) == {"dev", "test"}


def test_export_fpic_writes_crops_and_gold(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    for i in range(5):
        _write_board(
            raw, f"s{i}_front",
            [
                ["0", "[[10,10],[40,10],[40,25],[10,25]]", f"R{i + 1}", "Board", "", "0", ""],
                ["1", "[[50,50],[70,50],[70,90],[50,90]]", "C12", "Board", "", "90", ""],
                ["2", "[[100,10],[160,10],[160,30],[100,30]]", "PWR", "Board", "", "0", ""],
                ["3", "[[100,100],[160,100],[160,130],[100,130]]", "U7", "Device", "Texas Instruments", "0", ""],
                ["4", "[[1,1],[2,1],[2,2],[1,2]]", "R9", "Board", "", "0", ""],  # too small
            ],
        )
    assert len(ocr_annotation_csvs(raw)) == 5
    summary = export_fpic(raw, out, dev_frac=0.2)
    assert summary["n"] == 10
    assert summary["skipped_small"] == 5
    rows = [json.loads(line) for line in (out / "gold_ocr.jsonl").read_text().splitlines()]
    assert len(rows) == 10
    assert {r["split"] for r in rows} == {"fpic_ocr"}
    assert {r["subset"] for r in rows} == {"dev", "test"}
    assert {r["designator"] for r in rows} >= {"R1", "C12"}
    assert "PWR" not in {r["designator"] for r in rows}
    # Subset is per source image: no board straddles dev and test.
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source_image"], set()).add(r["subset"])
    assert all(len(v) == 1 for v in by_src.values())
    for r in rows:
        crop = cv2.imread(r["image"])
        assert crop is not None and crop.shape[0] > 0
