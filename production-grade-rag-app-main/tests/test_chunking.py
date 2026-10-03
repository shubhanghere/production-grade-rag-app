"""Unit tests for plain section-aware chunking (app/core/chunking.py)."""

from app.core.chunking import chunk_sections, make_section_id, section_to_chunks
from app.core.ingestion.base import ParsedSection

_META_KEYS = {
    "release", "source", "section_id", "section_title",
    "locator", "chunk_index", "content_hash",
}


def _section(slug="introduction", title="Introduction", text="body text",
             source="owmol"):
    return ParsedSection(
        source=source, locator=f"https://x/{slug}.html", slug=slug,
        title=title, subheadings=[], text=text,
    )


def test_make_section_id_shape():
    assert make_section_id("26b", "owmol", "introduction") == "26b/owmol/introduction"


def test_one_chunk_per_section():
    assert len(section_to_chunks(_section(), release="26b")) == 1


def test_ids_derived_from_section():
    c = section_to_chunks(_section(), release="26b")[0]
    assert c.section_id == "26b/owmol/introduction"
    assert c.chunk_id == "26b/owmol/introduction#0"


def test_title_prepended_to_text():
    c = section_to_chunks(_section(title="Introduction", text="body text"),
                          release="26b")[0]
    assert c.text == "Introduction\n\nbody text"


def test_empty_title_gives_body_only():
    c = section_to_chunks(_section(title="", text="body text"), release="26b")[0]
    assert c.text == "body text"


def test_metadata_keys_and_all_primitive():
    c = section_to_chunks(_section(), release="26b")[0]
    assert set(c.metadata) == _META_KEYS
    assert all(isinstance(v, (str, int, float, bool)) for v in c.metadata.values())


def test_content_hash_deterministic():
    a = section_to_chunks(_section(), release="26b")[0]
    b = section_to_chunks(_section(), release="26b")[0]
    assert a.metadata["content_hash"] == b.metadata["content_hash"]


def test_content_hash_changes_with_text():
    a = section_to_chunks(_section(text="one"), release="26b")[0]
    b = section_to_chunks(_section(text="two"), release="26b")[0]
    assert a.metadata["content_hash"] != b.metadata["content_hash"]


def test_chunk_sections_maps_all_with_unique_ids():
    secs = [_section(slug="a"), _section(slug="b"), _section(slug="c")]
    chunks = chunk_sections(secs, release="26b")
    assert len(chunks) == 3
    assert len({c.chunk_id for c in chunks}) == 3
