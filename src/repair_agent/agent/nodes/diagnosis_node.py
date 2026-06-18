"""Diagnosis synthesis node using DeepSeek to produce a structured diagnostic report."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from repair_agent.agent.state import AgentState
from repair_agent.config import settings
from repair_agent.rag.prompts import DIAGNOSIS_SYSTEM_PROMPT, DIAGNOSIS_USER_TEMPLATE


def _format_docs(state: AgentState) -> str:
    """Format retrieved RAG documents into a single text block for the LLM prompt."""
    rag_docs = state.get("rag_documents") or []
    if not rag_docs:
        return "No documentation retrieved."

    formatted = []
    for d in rag_docs:
        source = d.get("metadata", {}).get("source", "unknown")
        content = d.get("content", "")
        formatted.append(f"[Doc: {source}]\n{content}")

    return "\n\n".join(formatted)


def _get_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance pointed at DeepSeek's API."""
    return ChatOpenAI(
        model=settings.DEEPSEEK_LLM_MODEL,
        temperature=0,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base=settings.DEEPSEEK_BASE_URL,
    )


async def diagnosis_node(state: AgentState) -> dict:
    """Synthesize a final diagnosis from CV results, OCR output, and RAG documents.

    Uses DeepSeek to generate a structured diagnostic report including:
    1. Defect classification
    2. Root cause analysis
    3. Recommended repair actions
    4. Parts and document references

    Args:
        state: Full AgentState with CV, OCR, and RAG outputs.

    Returns:
        dict with diagnosis text.
    """
    llm = _get_llm()

    docs_text = _format_docs(state)

    user_message = DIAGNOSIS_USER_TEMPLATE.format(
        defect_type=state.get("defect_type", "unknown"),
        defect_confidence=state.get("defect_confidence", 0.0),
        serial_number=state.get("serial_number", "not detected"),
        self_correction_triggered=state.get("self_correction_triggered", False),
        correction_attempts=state.get("correction_attempts", 0),
        docs_text=docs_text,
    )

    response = await llm.ainvoke(
        [
            SystemMessage(content=DIAGNOSIS_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    )

    return {"diagnosis": response.content}
