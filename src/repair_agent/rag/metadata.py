"""Corpus metadata: manifest lookup, frontmatter parsing, defect-class helpers."""

from __future__ import annotations

import json
from pathlib import Path

from repair_agent.config import settings
from repair_agent.taxonomy import DETECTION_CLASSES

MANIFEST_PATH = Path(settings.CORPUS_DIR) / "manifest.json"

# Legacy CI stubs at corpus root — not part of the public RAG index.
STUB_TXT_NAMES = frozenset(
    {
        "burn_mark_repair_guide.txt",
        "corrosion_treatment_guide.txt",
        "capacitor_crack_detection.txt",
        "delamination_repair.txt",
        "serial_number_lookup_guide.txt",
    }
)

INGEST_GLOBS = (
    ("adapters/**/*.md", "md"),
    ("wikipedia/**/*.md", "md"),
    ("pdfs/**/*.pdf", "pdf"),
)


def load_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.is_file():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def parse_defect_classes(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def defect_classes_to_meta(classes: list[str]) -> str:
    """FAISS metadata values must be scalars; store classes as CSV."""
    return ",".join(sorted(set(classes)))


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse `key: value` metadata lines from adapters and Wikipedia extracts."""
    known_keys = {"source_id", "defect_classes", "license", "source_url"}
    meta: dict[str, str] = {}
    body_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if ":" in stripped and not stripped.startswith("#"):
            key, value = stripped.split(":", 1)
            normalized = key.strip().lower()
            if normalized in known_keys:
                meta[normalized] = value.strip()
                continue
        body_lines.append(line)
    body = "\n".join(body_lines).lstrip("\n")
    return meta, body


def manifest_entry_for_path(path: Path, manifest: dict[str, dict]) -> dict:
    name = path.name
    if name in manifest:
        return manifest[name]
    stem = path.stem.lower()
    for key, entry in manifest.items():
        if Path(key).stem.lower() == stem:
            return entry
    return {}


def matches_defect_filter(metadata: dict, defect_class: str | None) -> bool:
    if not defect_class:
        return True
    classes = parse_defect_classes(metadata.get("defect_classes"))
    if not classes:
        return True
    return defect_class in classes


def validate_manifest_classes(manifest: dict[str, dict]) -> None:
    for entry in manifest.values():
        for cls in parse_defect_classes(entry.get("defect_classes")):
            if cls not in DETECTION_CLASSES:
                raise ValueError(f"Unknown defect class in manifest: {cls!r}")
