from langchain_core.documents import Document

from app.core.rag import RagResult, _format_context, source_citations


def test_ragresult_shape():
    r = RagResult(answer="a", documents=[])
    assert r.answer == "a"
    assert r.documents == []


# ---------- _format_context ----------

def test_format_context_prefixes_section_title():
    docs = [Document(page_content="body", metadata={"section_title": "Intro"})]
    assert _format_context(docs) == "[Intro]\nbody"


def test_format_context_without_title():
    docs = [Document(page_content="body", metadata={})]
    assert _format_context(docs) == "body"


def test_format_context_joins_multiple_chunks():
    docs = [
        Document(page_content="a", metadata={"section_title": "A"}),
        Document(page_content="b", metadata={"section_title": "B"}),
    ]
    assert _format_context(docs) == "[A]\na\n\n---\n\n[B]\nb"


# ---------- source_citations ----------

def test_source_citations_dedupes_and_preserves_order():
    docs = [
        Document(page_content="", metadata={
            "section_id": "a", "section_title": "Alpha", "locator": "http://x/a"}),
        Document(page_content="", metadata={
            "section_id": "b", "section_title": "Beta", "locator": "http://x/b"}),
        Document(page_content="", metadata={          # duplicate section → dropped
            "section_id": "a", "section_title": "Alpha", "locator": "http://x/a"}),
    ]
    cites = source_citations(docs)
    assert [c["section_id"] for c in cites] == ["a", "b"]
    assert cites[0] == {"title": "Alpha", "url": "http://x/a", "section_id": "a"}


def test_source_citations_title_and_url_fallbacks():
    docs = [Document(page_content="", metadata={"section_id": "s1"})]
    assert source_citations(docs) == [{"title": "s1", "url": "", "section_id": "s1"}]


def test_source_citations_skips_missing_section_id():
    docs = [Document(page_content="", metadata={})]
    assert source_citations(docs) == []
