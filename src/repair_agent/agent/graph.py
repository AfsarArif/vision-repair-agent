"""LangGraph agent graph with self-correcting diagnostic workflow."""

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from repair_agent.agent.state import AgentState
from repair_agent.agent.nodes.cv_node import cv_node
from repair_agent.agent.nodes.rag_node import rag_node
from repair_agent.agent.nodes.ocr_node import ocr_node
from repair_agent.agent.nodes.diagnosis_node import diagnosis_node
from repair_agent.agent.edges import should_self_correct, correction_complete


def build_graph(checkpointer=None):
    """Build the LangGraph StateGraph for the vision repair agent.

    Graph flow:
        CV → RAG(initial) → [conditional] → OCR → RAG(corrected) → [conditional] → Diagnosis → END
                              ↓                                              ↓
                           Diagnosis                                    Diagnosis
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("cv", cv_node)
    graph.add_node("rag_initial", rag_node)
    graph.add_node("ocr", ocr_node)
    graph.add_node("rag_corrected", rag_node)
    graph.add_node("diagnosis", diagnosis_node)

    # Entry → CV
    graph.set_entry_point("cv")

    # CV → Initial RAG
    graph.add_edge("cv", "rag_initial")

    # Initial RAG → conditional: self-correct or diagnose
    graph.add_conditional_edges(
        "rag_initial",
        should_self_correct,
        {
            "self_correct": "ocr",
            "diagnose": "diagnosis",
        },
    )

    # OCR → Corrected RAG
    graph.add_edge("ocr", "rag_corrected")

    # Corrected RAG → conditional: retry or diagnose
    graph.add_conditional_edges(
        "rag_corrected",
        correction_complete,
        {
            "retry": "ocr",
            "diagnose": "diagnosis",
        },
    )

    # Diagnosis → END
    graph.add_edge("diagnosis", END)

    return graph.compile(checkpointer=checkpointer)


async def get_agent(db_connection_string: str | None = None):
    """Create a compiled agent.

    If db_connection_string is provided, uses PostgreSQL-backed checkpointing.
    Otherwise uses in-memory checkpointing (no persistence but works without Docker).
    """
    if db_connection_string:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        checkpointer = AsyncPostgresSaver.from_conn_string(db_connection_string)
        await checkpointer.setup()
        return build_graph(checkpointer)

    return build_graph(checkpointer=MemorySaver())


def get_agent_sync():
    """Create a compiled agent with in-memory checkpointing."""
    return build_graph(checkpointer=MemorySaver())
