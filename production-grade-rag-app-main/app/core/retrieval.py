"""
Retrieval pipeline: hybrid candidates -> rerank -> top-k.

  hybrid_search: dense (Chroma) + BM25 (keyword), fused with Reciprocal Rank
                 Fusion into a candidate pool.
  rerank:        Voyage cross-encoder re-scores that pool.
  retrieve:      the full pipeline the query path calls.

Chroma is the single source of truth. The BM25 index is a derived, in-memory
view rebuilt from Chroma's stored chunk texts (cached; invalidated on ingest).
"""

import asyncio
import re
import time

import httpx
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.config import settings
from app.core.vectorstore import get_vectorstore

_RRF_K = 60  # RRF smoothing constant (standard default)
_RERANK_URL = "https://api.voyageai.com/v1/rerank"
_MAX_RERANK_RETRIES = 4

_bm25_cache: BM25Retriever | None = None


def _tokenize(text: str) -> list[str]:
    # Better than BM25Retriever's default whitespace split: lowercase + split on
    # non-alphanumerics so casing/punctuation don't fragment keyword matches.
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], *, k: int = _RRF_K
) -> list[str]:
    """Merge ranked lists of keys into one, ordered by summed reciprocal rank."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def invalidate_bm25_cache() -> None:
    """Drop the cached BM25 index so it rebuilds from Chroma on next query."""
    global _bm25_cache
    _bm25_cache = None


def _get_bm25_retriever(pool: int) -> BM25Retriever | None:
    global _bm25_cache
    if _bm25_cache is None:
        data = get_vectorstore().get()          # all chunks: ids/documents/metadatas
        if not data["documents"]:
            return None                          # empty store (e.g. before first ingest)
        _bm25_cache = BM25Retriever.from_texts(
            texts=data["documents"], metadatas=data["metadatas"],
            preprocess_func=_tokenize,
        )
    _bm25_cache.k = pool
    return _bm25_cache


async def hybrid_search(
    question: str, *, pool: int | None = None
) -> list[Document]:
    """Dense + BM25 fused by RRF. Returns the full fused candidate pool; the
    final top-k cut is retrieve()'s job, after reranking."""
    pool = pool or settings.retrieval_pool

    dense_retriever = get_vectorstore().as_retriever(search_kwargs={"k": pool})
    dense_docs = await dense_retriever.ainvoke(question)

    bm25 = _get_bm25_retriever(pool)
    bm25_docs = bm25.invoke(question) if bm25 is not None else []

    ranked_lists = [
        [d.metadata["section_id"] for d in dense_docs],
        [d.metadata["section_id"] for d in bm25_docs],
    ]
    fused_ids = reciprocal_rank_fusion(ranked_lists)

    # Map section_id -> Document (either retriever's copy; content is identical).
    by_id = {d.metadata["section_id"]: d for d in bm25_docs + dense_docs}
    return [by_id[sid] for sid in fused_ids if sid in by_id]


def _voyage_rerank(query: str, texts: list[str], *, top_k: int) -> list[dict]:
    """Call Voyage rerank. Returns a list of result dicts, each shaped
    {"index": <int position in `texts`>, "relevance_score": <float>}, e.g.
    [{"index": 2, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.34}].
    Same httpx-direct approach as VoyageEmbeddings (SDK gated to old Python)."""
    for attempt in range(_MAX_RERANK_RETRIES):
        resp = httpx.post(
            _RERANK_URL,
            headers={"Authorization": f"Bearer {settings.voyage_api_key}"},
            json={"query": query, "documents": texts,
                  "model": settings.rerank_model, "top_k": top_k},
            timeout=60.0,
        )
        if resp.status_code == 429 and attempt < _MAX_RERANK_RETRIES - 1:
            wait = float(resp.headers.get("retry-after", 0)) or min(2 ** attempt, 30)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        # Voyage nests the ranked list under "data" (verified against the live API).
        return resp.json()["data"]


def rerank(query: str, documents: list[Document], *, k: int) -> list[Document]:
    """Re-score candidates with the cross-encoder; return the top-k reordered.
    Higher relevance_score = more relevant, so we sort descending."""
    if not documents:
        return []
    results = _voyage_rerank(query, [d.page_content for d in documents], top_k=k)
    ranked = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
    return [documents[r["index"]] for r in ranked[:k]]


async def retrieve(
    question: str, *, k: int | None = None, pool: int | None = None
) -> list[Document]:
    """Full retrieval pipeline: hybrid candidates -> rerank -> top-k."""
    k = k or settings.retrieval_k
    candidates = await hybrid_search(question, pool=pool)
    # rerank is a blocking REST call → offload so the event loop isn't blocked.
    return await asyncio.to_thread(rerank, question, candidates, k=k)
