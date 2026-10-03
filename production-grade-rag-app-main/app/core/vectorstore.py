"""
Builds the Chroma vector store and a similarity retriever over document chunks.

Chunks (one per source section — see app/core/chunking.py) are embedded with
Voyage AI and stored directly in Chroma; retrieval is a plain top-k similarity
search. Each chunk's metadata (section_id, section_title, locator, ...) rides
along for scoring and citations.

Note: the voyageai SDK is not used because all of its releases are gated to
Python <3.13 or <3.14. Instead, VoyageEmbeddings calls the Voyage AI REST API
directly via httpx, which has no Python version restrictions.
"""

import time
from functools import lru_cache

import httpx
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import settings

_MAX_EMBED_RETRIES = 6

# Module-level pacing: every embedding request (a bulk ingest committed in small
# groups, or a single query) counts toward the SAME voyage_rpm/min budget, so
# the throttle must persist ACROSS calls. Reranking has its own endpoint/limit.
_last_embed_ts = 0.0


def _throttle_embeddings() -> None:
    global _last_embed_ts
    interval = 60.0 / settings.voyage_rpm
    gap = time.monotonic() - _last_embed_ts
    if _last_embed_ts and gap < interval:
        time.sleep(interval - gap)
    _last_embed_ts = time.monotonic()


def _estimate_tokens(text: str) -> int:
    # ~4 chars/token for English; avoids a tokenizer dependency.
    return max(1, len(text) // 4)


class VoyageEmbeddings(Embeddings):
    """
    Thin LangChain Embeddings wrapper that calls the Voyage AI REST API.
    Implements the two methods LangChain requires: embed_documents and embed_query.
    """

    _API_URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(_MAX_EMBED_RETRIES):
            _throttle_embeddings()  # pace to <= voyage_rpm/min across all callers
            response = httpx.post(
                self._API_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"input": texts, "model": self._model},
                timeout=60.0,
            )
            if response.status_code == 429 and attempt < _MAX_EMBED_RETRIES - 1:
                retry_after = response.headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after else min(2 ** attempt, 30))
                continue
            response.raise_for_status()
            items = response.json()["data"]
            # Voyage returns results in order, but sort by index to be safe.
            return [item["embedding"] for item in sorted(items, key=lambda x: x["index"])]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Keep each request well under voyage_tpm (0.6 margin covers token-
        # estimate error). _embed's throttle enforces the per-minute request
        # cap, so no per-call pacing is needed here.
        cap = max(1, int(settings.voyage_tpm / settings.voyage_rpm * 0.6))
        out: list[list[float]] = []
        batch: list[str] = []
        batch_tokens = 0
        for text in texts:
            n = _estimate_tokens(text)
            if batch and batch_tokens + n > cap:
                out.extend(self._embed(batch))
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += n
        if batch:
            out.extend(self._embed(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    embeddings = VoyageEmbeddings(
        api_key=settings.voyage_api_key,
        model=settings.embedding_model,
    )
    return Chroma(
        collection_name="doc_chunks",
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
