"""Unit tests for defect-aware retrieval filtering."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from repair_agent.rag.retriever import _filter_docs


def _doc(defect_classes: str):
    return SimpleNamespace(metadata={"defect_classes": defect_classes})


def test_filter_docs_prefers_matching_classes():
    docs = [
        _doc("short"),
        _doc("open,short"),
        _doc(""),
        _doc("mousebite"),
    ]
    filtered = _filter_docs(docs, "open", k=2)
    assert len(filtered) == 2
    assert all(
        not d.metadata["defect_classes"] or "open" in d.metadata["defect_classes"]
        for d in filtered
    )


def test_filter_docs_falls_back_when_no_match():
    docs = [_doc("short"), _doc("mousebite")]
    filtered = _filter_docs(docs, "open", k=1)
    assert filtered == docs[:1]


@pytest.mark.asyncio
async def test_aretrieve_passes_defect_class_to_search():
    from repair_agent.rag import retriever

    fake_doc = SimpleNamespace(
        page_content="open trace",
        metadata={"defect_classes": "open", "source_id": "adapter-open"},
        score=None,
    )

    with patch.object(retriever, "get_vectorstore") as mock_store:
        mock_store.return_value.similarity_search.return_value = [fake_doc]
        results = await retriever.aretrieve("open circuit", k=3, defect_class="open")
        mock_store.return_value.similarity_search.assert_called_once()
        assert results[0]["metadata"]["source_id"] == "adapter-open"


def test_rerank_orders_by_cross_encoder_score():
    from repair_agent.rag import retriever

    docs = [SimpleNamespace(page_content=t, metadata={}) for t in ("a", "bb", "ccc")]

    class FakeCE:
        def predict(self, pairs):
            return [len(text) for _, text in pairs]

    with patch.object(retriever, "_get_reranker", return_value=FakeCE()):
        ranked = retriever.rerank("q", docs, "fake")
    assert [d.page_content for d in ranked] == ["ccc", "bb", "a"]


@pytest.mark.asyncio
async def test_aretrieve_reranks_filtered_candidates(monkeypatch):
    from repair_agent.rag import retriever

    docs = [
        SimpleNamespace(page_content="x", metadata={"defect_classes": "short"}),
        SimpleNamespace(page_content="yy", metadata={"defect_classes": "open"}),
        SimpleNamespace(page_content="zzz", metadata={"defect_classes": ""}),
    ]

    class FakeCE:
        def predict(self, pairs):
            return [len(text) for _, text in pairs]

    monkeypatch.setattr(retriever.settings, "RAG_FETCH_K", 20)
    with patch.object(retriever, "get_vectorstore") as mock_store, patch.object(
        retriever, "_get_reranker", return_value=FakeCE()
    ):
        mock_store.return_value.similarity_search.return_value = docs
        results = await retriever.aretrieve("q", k=2, defect_class="open", reranker="fake")
        assert mock_store.return_value.similarity_search.call_args.kwargs["k"] == 20
    assert [r["content"] for r in results] == ["zzz", "yy"]
