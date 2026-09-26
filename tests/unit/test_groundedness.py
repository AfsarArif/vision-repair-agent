"""Deterministic diagnosis groundedness checks."""

from repair_agent.eval.groundedness import parse_judge, score_diagnosis

DOCS = [
    {
        "content": "An open is a break in a copper path. See NASA-STD-8739.3 for open joints.",
        "metadata": {"source_id": "adapter-open"},
    },
    {"content": "Flying probe testers check continuity.", "metadata": {"source_id": "wikipedia-pcbm"}},
]


def test_grounded_report_passes():
    text = "Defect: open. A break in copper [adapter-open]; confirm with flying probe [wikipedia-pcbm]. NASA-STD-8739.3 covers open joints [adapter-open]."
    result = score_diagnosis(text, DOCS, "open")
    assert result["grounded"] is True


def test_unretrieved_citation_and_invented_clause_fail():
    text = "Defect: open [ipc-a-610]. Reject per IPC-A-610 clause 10.2.1 [adapter-open]."
    result = score_diagnosis(text, DOCS, "open")
    assert result["grounded"] is False
    assert result["bad_citations"] == ["ipc-a-610"]
    assert "IPC-A-610" in result["unsupported_references"]
    assert "clause 10.2.1" in result["unsupported_references"]


def test_report_must_name_detected_class():
    result = score_diagnosis("Copper issue [adapter-open].", DOCS, "spurious_copper")
    assert result["class_named"] is False
    assert result["grounded"] is False


def test_class_with_underscore_matches_spaced_form():
    result = score_diagnosis("Spurious copper island [adapter-open].", DOCS, "spurious_copper")
    assert result["class_named"] is True


def test_parse_judge_extracts_json():
    parsed = parse_judge('Sure: {"supported_claims": 3, "unsupported_claims": 0, "verdict": "Grounded"}')
    assert parsed["verdict"] == "grounded"
    assert parse_judge("no json")["verdict"] == "unparsed"
