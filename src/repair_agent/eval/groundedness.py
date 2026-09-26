"""Diagnosis groundedness: does the report cite only what was retrieved?

Two layers, reported separately:

1. Deterministic (no LLM): bracketed citations must be retrieved source ids,
   standard / clause references must appear verbatim in retrieved text, and
   the report must name the detector's class.
2. LLM judge (optional): every factual claim is supported by the retrieved
   passages. Needs an API key; skipped otherwise.
"""

from __future__ import annotations

import json
import re

from repair_agent.taxonomy import CANONICAL_CLASSES

CITATION_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.()\-]*)\]")
STANDARD_RE = re.compile(
    r"\b(IPC[- ]?[A-Z]{0,2}-?\d{3,4}[A-Z]?|J-STD-\d{3}[A-Z]?|NASA-STD-\d{4}\.\d+[A-Z]?"
    r"|ECSS-[A-Z]-ST-\d{2}-\d{2}[A-Z]?|MIL-(?:STD|PRF)-\d+[A-Z]?)\b",
    re.IGNORECASE,
)
CLAUSE_RE = re.compile(
    r"\b(?:clause|section|paragraph|para\.?|table|figure|§)\s*(\d+(?:\.\d+)+[a-z]?)",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return re.sub(r"[\s\-]+", "", text).lower()


def retrieved_ids(docs: list[dict]) -> set[str]:
    ids: set[str] = set()
    for d in docs:
        meta = d.get("metadata") or {}
        if meta.get("source_id"):
            ids.add(str(meta["source_id"]))
    return ids


def extract_citations(diagnosis: str) -> list[str]:
    """Bracketed tokens that look like source ids (skip markdown checkboxes etc.)."""
    out = []
    for token in CITATION_RE.findall(diagnosis):
        if token.lower() in {"x", " "} or token.isdigit():
            continue
        out.append(token)
    return out


def extract_references(diagnosis: str) -> list[str]:
    refs = [m.group(0) for m in STANDARD_RE.finditer(diagnosis)]
    refs += [m.group(0) for m in CLAUSE_RE.finditer(diagnosis)]
    return refs


def score_diagnosis(diagnosis: str, docs: list[dict], detected_class: str | None) -> dict:
    """Deterministic groundedness checks for one report."""
    ids = retrieved_ids(docs)
    corpus_text = _norm(" ".join(d.get("content", "") for d in docs))
    citations = extract_citations(diagnosis)
    bad_citations = sorted({c for c in citations if c not in ids})
    refs = extract_references(diagnosis)
    unsupported_refs = sorted({r for r in refs if _norm(r) not in corpus_text})
    lowered = diagnosis.lower()
    names = {detected_class, (detected_class or "").replace("_", " ")} - {None, ""}
    class_named = bool(detected_class) and any(n.lower() in lowered for n in names)
    other_classes = [
        c for c in CANONICAL_CLASSES
        if c not in (detected_class, "normal") and re.search(rf"\b{re.escape(c)}\b", lowered)
    ]
    grounded = bool(citations) and not bad_citations and not unsupported_refs and class_named
    return {
        "grounded": grounded,
        "n_citations": len(citations),
        "bad_citations": bad_citations,
        "unsupported_references": unsupported_refs,
        "class_named": class_named,
        "other_classes_mentioned": other_classes,
    }


JUDGE_SYSTEM = """You grade whether a PCB inspection report is grounded in the passages it was given.
A claim is supported only if a passage states it or it follows directly from one.
Detector facts in the header (class, confidence, correction mode) count as given.
Reply with JSON only: {"supported_claims": int, "unsupported_claims": int, "unsupported": [short quotes], "verdict": "grounded" | "ungrounded"}.
Verdict is "grounded" only when unsupported_claims == 0."""


def judge_messages(diagnosis: str, docs: list[dict], header: str) -> list[tuple[str, str]]:
    passages = "\n\n".join(
        f"[{(d.get('metadata') or {}).get('source_id', '?')}]\n{d.get('content', '')}" for d in docs
    )
    user = f"Detector header:\n{header}\n\nPassages:\n{passages}\n\nReport:\n{diagnosis}"
    return [("system", JUDGE_SYSTEM), ("user", user)]


def parse_judge(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"verdict": "unparsed", "raw": text[:500]}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"verdict": "unparsed", "raw": text[:500]}
    data["verdict"] = str(data.get("verdict", "unparsed")).lower()
    return data
