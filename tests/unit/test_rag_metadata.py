"""Unit tests for corpus metadata helpers."""

from pathlib import Path

from repair_agent.rag.metadata import (
    defect_classes_to_meta,
    manifest_entry_for_path,
    matches_defect_filter,
    parse_defect_classes,
    parse_frontmatter,
)


def test_parse_frontmatter_extracts_adapter_fields():
    text = (
        "# Open circuit\n"
        "source_id: adapter-open\n"
        "defect_classes: open\n"
        "\n"
        "Body text about opens.\n"
    )
    meta, body = parse_frontmatter(text)
    assert meta["source_id"] == "adapter-open"
    assert meta["defect_classes"] == "open"
    assert "Body text" in body


def test_defect_classes_roundtrip():
    classes = ["open", "short", "open"]
    meta = defect_classes_to_meta(classes)
    assert parse_defect_classes(meta) == ["open", "short"]


def test_matches_defect_filter_allows_general_docs():
    assert matches_defect_filter({"defect_classes": ""}, "open")
    assert matches_defect_filter({"defect_classes": "open,short"}, "open")
    assert not matches_defect_filter({"defect_classes": "short"}, "open")


def test_manifest_entry_for_path_by_filename(tmp_path: Path):
    manifest = {"ecss-q-st-70-61c.pdf": {"source_id": "ecss-q-st-70-61c"}}
    entry = manifest_entry_for_path(tmp_path / "ecss-q-st-70-61c.pdf", manifest)
    assert entry["source_id"] == "ecss-q-st-70-61c"


def test_load_markdown_keeps_adapter_frontmatter_in_metadata(tmp_path: Path):
    from repair_agent.rag.ingestor import _load_markdown

    path = tmp_path / "open.md"
    path.write_text(
        "# Open circuit\nsource_id: adapter-open\ndefect_classes: open\n\nBody.\n",
        encoding="utf-8",
    )
    doc = _load_markdown(path, manifest={})
    assert doc.metadata["source_id"] == "adapter-open"
    assert doc.metadata["defect_classes"] == "open"
    assert "source_id" not in doc.page_content
