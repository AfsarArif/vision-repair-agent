"""LangGraph agent graph with self-correcting diagnostic workflow."""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from repair_agent.agent.state import AgentState
from repair_agent.agent.nodes.cv_node import cv_node
from repair_agent.agent.nodes.rag_node import rag_node
from repair_agent.agent.nodes.self_correct_node import self_correct_node
from repair_agent.agent.nodes.diagnosis_node import diagnosis_node
from repair_agent.agent.edges import should_self_correct, correction_complete


def build_graph(checkpointer=None):
    """Build the LangGraph StateGraph for the vision repair agent.

    Graph flow:
        CV → RAG(initial) → [conditional] → self_correct → RAG(corrected) → Diagnosis → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("cv", cv_node)
    graph.add_node("rag_initial", rag_node)
    graph.add_node("self_correct", self_correct_node)
    graph.add_node("rag_corrected", rag_node)
    graph.add_node("diagnosis", diagnosis_node)

    graph.set_entry_point("cv")
    graph.add_edge("cv", "rag_initial")
    graph.add_conditional_edges(
        "rag_initial",
        should_self_correct,
        {
            "self_correct": "self_correct",
            "diagnose": "diagnosis",
        },
    )
    graph.add_edge("self_correct", "rag_corrected")
    graph.add_conditional_edges(
        "rag_corrected",
        correction_complete,
        {
            "retry": "self_correct",
            "diagnose": "diagnosis",
        },
    )
    graph.add_edge("diagnosis", END)

    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def get_agent(db_connection_string: str | None = None) -> AsyncIterator[Any]:
    """Async context manager yielding a compiled agent with the right checkpointer.

    * ``db_connection_string`` given → PostgreSQL checkpointing via ``AsyncPostgresSaver``
      over a psycopg ``AsyncConnectionPool``. Accepts a SQLAlchemy-style URL
      (``postgresql+asyncpg://...``) and converts it to psycopg conninfo. The pool is
      opened on enter, checkpoint tables are created (``setup()``), and the pool is
      closed on exit — use it from the FastAPI lifespan.
    * otherwise → in-memory ``MemorySaver`` (no persistence, no Docker).

        async with get_agent(url) as agent:
            await agent.ainvoke(state, config={"configurable": {"thread_id": sid}})
    """
    if not db_connection_string:
        yield build_graph(checkpointer=MemorySaver())
        return

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from repair_agent.db.connection import to_psycopg_conninfo

    pool = AsyncConnectionPool(
        conninfo=to_psycopg_conninfo(db_connection_string),
        max_size=10,
        open=False,
        # Required by AsyncPostgresSaver: autocommit for setup(), dict rows, and no
        # server-side prepared statements (safe behind pgbouncer).
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open(wait=True, timeout=10)
    try:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        yield build_graph(checkpointer)
    finally:
        await pool.close()


def get_agent_sync():
    """Create a compiled agent with in-memory checkpointing."""
    return build_graph(checkpointer=MemorySaver())
