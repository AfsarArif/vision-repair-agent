"""RAG prompt templates for hardware diagnostic retrieval and synthesis."""

RAG_QUERY_TEMPLATE = "Hardware diagnostic procedure for {defect_type}"

RAG_QUERY_WITH_SERIAL_TEMPLATE = (
    "Hardware diagnostic for serial number {serial_number} with defect: {defect_type}"
)

DIAGNOSIS_SYSTEM_PROMPT = """You are an expert hardware diagnostic AI. Given defect detection results
and retrieved documentation, produce a structured diagnosis including:
1. Defect classification
2. Root cause analysis
3. Recommended repair actions (step-by-step)
4. Parts or references from documentation
Be concise, factual, and cite the document sources by their metadata filenames."""

DIAGNOSIS_USER_TEMPLATE = """Defect Type: {defect_type}
Confidence: {defect_confidence:.2%}
Serial Number: {serial_number}
Self-Correction Applied: {self_correction_triggered}
Correction Attempts: {correction_attempts}

Retrieved Documentation:
{docs_text}

Provide a complete diagnostic report."""

RETRIEVAL_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
