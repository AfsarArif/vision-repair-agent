"""RAG prompt templates for PCB defect retrieval and diagnosis synthesis."""

RAG_QUERY_TEMPLATE = (
    "Bare-board PCB fabrication defect {defect_type}: inspection notes, "
    "accept/reject criteria, and disposition or rework"
)

RAG_QUERY_WITH_DESIGNATOR_TEMPLATE = (
    "Bare-board PCB defect {defect_type} near reference designator {designator}: "
    "inspection notes and disposition"
)

DIAGNOSIS_SYSTEM_PROMPT = """You are an AOI assistant for bare printed circuit boards.
Given detector output and retrieved public workmanship text, produce:
1. Defect classification (use the detector class names: open, short, mousebite, spur, spurious_copper, pin_hole, missing_hole, normal)
2. What the defect means electrically
3. Typical disposition (scrap, electrical test, isolate/etch, jumper) grounded in the retrieved text
4. Citations: cite each claim with the bracketed source id of the retrieved document it came from, e.g. [adapter-open]

Do not invent IPC clause numbers, part numbers, or NASA paragraph ids that were not retrieved.
If documentation is missing, say so."""

DIAGNOSIS_USER_TEMPLATE = """Defect Type: {defect_type}
Confidence: {defect_confidence:.2%}
Designator / serial: {serial_number}
Self-Correction Applied: {self_correction_triggered}
Correction Mode: {correction_mode}
Correction Attempts: {correction_attempts}

Retrieved Documentation:
{docs_text}

Provide a diagnostic report. Cite only the documents above, using their bracketed source ids."""

RETRIEVAL_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
