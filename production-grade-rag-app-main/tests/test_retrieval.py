"""Unit tests for the retrieval pipeline (app/core/retrieval.py) — all offline.

RRF and the tokenizer are pure functions; hybrid_search / rerank / retrieve are
tested with monkeypatched retrievers and a stubbed Voyage call, so no
embeddings, reranking API, or network are involved.
"""

import pytest
from langchain_core.documents import Document

import app.core.retrieval as retr
from app.core.retrieval import _tokenize, hybrid_search, rerank, reciprocal_rank_fusion


# ---------- reciprocal_rank_fusion ----------

def test_rrf_worked_example():
    fused = reciprocal_rank_fusion([["C1", "C2", "C3"], ["C3", "C4", "C1"]])
    assert set(fused[:2]) == {"C1", "C3"}     # each is #1 in one list
    assert set(fused[2:]) == {"C2", "C4"}


def test_rrf_agreement_beats_single_first():
    # A is rank 2 in BOTH lists; B is rank 1 in ONE. With k=60, agreement wins.
    fused = reciprocal_rank_fusion([["B", "A"], ["X", "A"]])
    assert fused[0] == "A"


def test_rrf_handles_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# ---------- _tokenize ----------

def test_tokenize_lowercases_and_splits_punctuation():
    assert _tokenize("Hello, World! 26B") == ["hello", "world", "26b"]


# ---------- hybrid_search (now returns the full fused pool) ----------

def _doc(section_id: str) -> Document:
    return Document(page_content=section_id, metadata={"section_id": section_id})


class _FakeDense:
    def __init__(self, docs):
        self._docs = docs

    def as_retriever(self, search_kwargs=None):
        return self

    async def ainvoke(self, question):
        return self._docs


class _FakeBM25:
    def __init__(self, docs):
        self._docs = docs
        self.k = None

    def invoke(self, question):
        return self._docs


@pytest.mark.asyncio
async def test_hybrid_search_returns_full_fused_pool(monkeypatch):
    dense = [_doc("C1"), _doc("C2"), _doc("C3")]
    bm25 = [_doc("C3"), _doc("C4"), _doc("C1")]
    monkeypatch.setattr(retr, "get_vectorstore", lambda: _FakeDense(dense))
    monkeypatch.setattr(retr, "_get_bm25_retriever", lambda pool: _FakeBM25(bm25))

    out = await hybrid_search("q", pool=10)
    # No top-k cut here — retrieve() owns that. Full fused ordering returned.
    assert [d.metadata["section_id"] for d in out] == ["C1", "C3", "C2", "C4"]


@pytest.mark.asyncio
async def test_hybrid_search_dense_only_when_no_bm25(monkeypatch):
    dense = [_doc("C1"), _doc("C2")]
    monkeypatch.setattr(retr, "get_vectorstore", lambda: _FakeDense(dense))
    monkeypatch.setattr(retr, "_get_bm25_retriever", lambda pool: None)

    out = await hybrid_search("q", pool=10)
    assert [d.metadata["section_id"] for d in out] == ["C1", "C2"]


# ---------- rerank ----------

def test_rerank_reorders_and_cuts(monkeypatch):
    docs = [_doc("A"), _doc("B"), _doc("C")]

    def fake(query, texts, *, top_k):
        return [{"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.5},
                {"index": 0, "relevance_score": 0.1}]

    monkeypatch.setattr(retr, "_voyage_rerank", fake)
    out = rerank("q", docs, k=2)
    assert [d.metadata["section_id"] for d in out] == ["B", "C"]


def test_rerank_sorts_defensively(monkeypatch):
    # Unsorted API response → rerank must sort by score desc itself.
    docs = [_doc("A"), _doc("B")]
    monkeypatch.setattr(
        retr, "_voyage_rerank",
        lambda query, texts, *, top_k: [
            {"index": 0, "relevance_score": 0.2},
            {"index": 1, "relevance_score": 0.8},
        ],
    )
    out = rerank("q", docs, k=2)
    assert [d.metadata["section_id"] for d in out] == ["B", "A"]


def test_rerank_empty_makes_no_api_call(monkeypatch):
    called = False

    def fake(*args, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(retr, "_voyage_rerank", fake)
    assert rerank("q", [], k=5) == []
    assert called is False


# ---------- retrieve (orchestration) ----------

@pytest.mark.asyncio
async def test_retrieve_runs_hybrid_then_rerank(monkeypatch):
    candidates = [_doc("A"), _doc("B"), _doc("C")]

    async def fake_hybrid(question, *, pool=None):
        return candidates

    monkeypatch.setattr(retr, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(retr, "rerank", lambda q, docs, *, k: docs[:k][::-1])

    out = await retr.retrieve("q", k=2)
    assert [d.metadata["section_id"] for d in out] == ["B", "A"]


@pytest.mark.asyncio
async def test_retrieve_empty_candidates_returns_empty(monkeypatch):
    async def fake_hybrid(question, *, pool=None):
        return []

    # Real rerank runs and short-circuits on empty input (no API call).
    monkeypatch.setattr(retr, "hybrid_search", fake_hybrid)
    out = await retr.retrieve("q", k=5)
    assert out == []
