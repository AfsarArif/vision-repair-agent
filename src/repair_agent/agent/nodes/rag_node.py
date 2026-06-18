"""RAG query node for retrieving relevant hardware diagnostic documents."""

from repair_agent.agent.state import AgentState
from repair_agent.rag.retriever import aretrieve
from repair_agent.rag.prompts import RAG_QUERY_TEMPLATE, RAG_QUERY_WITH_SERIAL_TEMPLATE


def _build_query(state: AgentState) -> str:
    """Build a RAG query from the current state.

    Uses serial number if available for precise lookup; otherwise uses defect type.
    """
    defect = state.get("defect_type", "unknown defect")
    serial = state.get("serial_number")

    if serial:
        return RAG_QUERY_WITH_SERIAL_TEMPLATE.format(
            serial_number=serial, defect_type=defect
        )
    return RAG_QUERY_TEMPLATE.format(defect_type=defect)


async def rag_node(state: AgentState) -> dict:
    """Query the pgvector retriever for relevant diagnostic documents.

    Builds a query from the current defect type and (optionally) serial number,
    then retrieves the top-k matching document chunks.

    Args:
        state: Current AgentState with defect_type and optional serial_number.

    Returns:
        dict with rag_query and rag_documents.
    """
    query = _build_query(state)
    rag_docs = await aretrieve(query)

    return {
        "rag_query": query,
        "rag_documents": rag_docs,
    }
