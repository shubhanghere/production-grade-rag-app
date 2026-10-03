"""Unit tests for the chunk-based ingest path (app/core/rag.py).

A fake vector store (monkeypatched in) records what gets written, so we can
assert the cost-saving skip + upsert behavior without embedding or network.
"""

import pytest

import app.core.rag as rag
from app.config import settings
from app.core.ingestion.base import ParsedSection
from app.core.rag import DocumentTooLargeError, _slugify


class _FakeStore:
    """Minimal stand-in for the Chroma wrapper used by _store_chunks."""

    def __init__(self):
        self.docs = {}                 # chunk_id -> content_hash
        self.last_write = "UNSET"      # sentinel: stays UNSET if no write happens

    def get(self, ids):
        present = [(i, {"content_hash": self.docs[i]}) for i in ids if i in self.docs]
        return {"ids": [i for i, _ in present], "metadatas": [m for _, m in present]}

    def add_documents(self, documents, ids):
        self.last_write = list(ids)
        for d, i in zip(documents, ids):
            self.docs[i] = d.metadata["content_hash"]


@pytest.fixture
def store(monkeypatch):
    fake = _FakeStore()
    monkeypatch.setattr(rag, "get_vectorstore", lambda: fake)
    return fake


def _section(slug="introduction", title="Introduction", text="hello",
             source="owmol"):
    return ParsedSection(
        source=source, locator="u", slug=slug,
        title=title, subheadings=[], text=text,
    )


# ---------- _store_chunks: skip-unchanged + upsert ----------

def test_first_ingest_writes_chunk(store):
    rag.ingest_sections([_section()], release="26b")
    assert store.last_write == ["26b/owmol/introduction#0"]


def test_reingest_identical_writes_nothing(store):
    rag.ingest_sections([_section()], release="26b")
    store.last_write = "UNSET"
    rag.ingest_sections([_section()], release="26b")     # same content_hash
    assert store.last_write == "UNSET"                   # add_documents not called


def test_changed_content_rewrites_same_id(store):
    rag.ingest_sections([_section(text="hello")], release="26b")
    rag.ingest_sections([_section(text="CHANGED")], release="26b")
    assert store.last_write == ["26b/owmol/introduction#0"]   # same id → upsert
    assert len(store.docs) == 1                               # no duplicate


# ---------- ingest_document ----------

@pytest.mark.asyncio
async def test_ingest_document_returns_section_id(store):
    section_id = await rag.ingest_document("hello world", source="notes")
    assert section_id == "uploads/notes/notes"


@pytest.mark.asyncio
async def test_ingest_document_rejects_oversize(store):
    big = "x" * (settings.max_upload_chars + 1)
    with pytest.raises(DocumentTooLargeError):
        await rag.ingest_document(big, source="huge")


# ---------- _slugify ----------

def test_slugify_normalizes():
    assert _slugify("Hello World!") == "hello-world"
    assert _slugify("a__b--c") == "a-b-c"


def test_slugify_falls_back_when_empty():
    assert _slugify("") == "upload"
    assert _slugify("!!!") == "upload"
