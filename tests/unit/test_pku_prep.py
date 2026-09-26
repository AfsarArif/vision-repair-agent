"""Tests for PKU-Market-PCB VOC parsing and holdout gold export (synthetic fixtures)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from repair_agent.data.pku import (
    SPLIT_NAME,
    build_gold_rows,
    find_samples,
    normalize_pku_name,
    parse_voc,
    write_gold,
)

_VOC = """<annotation>
  <filename>{fname}</filename>
  <size><width>64</width><height>48</height><depth>3</depth></size>
  {objects}
</annotation>"""

_OBJ = """<object><name>{name}</name><bndbox>
  <xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax>
</bndbox></object>"""


def _xml(path: Path, objs: list[tuple[str, int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(_OBJ.format(name=n, x1=a, y1=b, x2=c, y2=d) for n, a, b, c, d in objs)
    path.write_text(_VOC.format(fname=path.stem, objects=body), encoding="utf-8")


def _jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), (0, 120, 0)).save(path, format="JPEG")


def test_normalize_pku_name_variants():
    assert normalize_pku_name("open_circuit") == "open"
    assert normalize_pku_name("Mouse_bite") == "mousebite"
    assert normalize_pku_name("spurious copper") == "spurious_copper"
    assert normalize_pku_name("Missing-hole") == "missing_hole"
    assert normalize_pku_name("burn_mark") is None


def test_parse_voc_maps_names_and_skips_unknown(tmp_path: Path):
    xml = tmp_path / "a.xml"
    _xml(xml, [("open_circuit", 1, 2, 10, 12), ("spur", 5, 5, 9, 9), ("weird", 0, 0, 3, 3)])
    parsed = parse_voc(xml)
    assert parsed["width"] == 64 and parsed["height"] == 48
    assert parsed["boxes"] == [
        {"bbox": [1, 2, 10, 12], "cls": "open"},
        {"bbox": [5, 5, 9, 9], "cls": "spur"},
    ]
    assert parsed["unknown"] == ["weird"]


def test_find_samples_mirror_and_original_layouts_skip_rotation(tmp_path: Path):
    # Mirror layout: stem_annotation.xml next to the jpg.
    _jpg(tmp_path / "Short" / "short01.jpg")
    _xml(tmp_path / "Short" / "short01_annotation.xml", [("short", 1, 1, 8, 8)])
    # Original zip layout: images/<cls>/x.jpg + Annotations/<cls>/x.xml
    _jpg(tmp_path / "images" / "Spur" / "01_spur_01.jpg")
    _xml(tmp_path / "Annotations" / "Spur" / "01_spur_01.xml", [("spur", 2, 2, 6, 6)])
    # Augmented copy must be ignored.
    _jpg(tmp_path / "rotation" / "Spur_rotation" / "01_spur_01.jpg")
    _xml(tmp_path / "rotation" / "Spur_rotation" / "01_spur_01.xml", [("spur", 2, 2, 6, 6)])
    # Image without annotation is ignored.
    _jpg(tmp_path / "Short" / "short02.jpg")

    samples = find_samples(tmp_path)
    names = sorted(img.name for img, _ in samples)
    assert names == ["01_spur_01.jpg", "short01.jpg"]


def test_build_and_write_gold_rows(tmp_path: Path):
    _jpg(tmp_path / "raw" / "Mouse_bite" / "mouse_bite01.jpg")
    _xml(
        tmp_path / "raw" / "Mouse_bite" / "mouse_bite01_annotation.xml",
        [("mouse_bite", 3, 4, 20, 30), ("missing_hole", 1, 1, 5, 5)],
    )
    rows, stats = build_gold_rows(find_samples(tmp_path / "raw"))
    assert stats["images"] == 1 and stats["boxes"] == 2
    assert stats["per_class"] == {"mousebite": 1, "missing_hole": 1}
    row = rows[0]
    assert row["image_id"] == "mouse_bite01"
    assert row["split"] == SPLIT_NAME == "pku_holdout"
    assert Path(row["image"]).is_file()
    assert row["boxes"][0] == {"bbox": [3, 4, 20, 30], "cls": "mousebite"}

    out = write_gold(rows, tmp_path / "out" / "gold_holdout.jsonl")
    loaded = [json.loads(line) for line in out.read_text().splitlines()]
    assert loaded == rows
