"""
Turn parsed sections into embeddable chunks with stable ids and metadata.

Plain section-aware strategy: ONE chunk per section. No size limits, overlap,
or sub-splitting — the source's own section boundaries define the chunks. The
function returns a list (always length 1 today) so a future oversize-split can
slot in without changing callers.

`section_id` is the stable, deterministic identity of a section
(`release/source/slug`, e.g. "26b/owmol/introduction"). It is what the golden
set's gold_doc_id points at and what retrieval_recall scores against, so it must
never change for the same source section.
"""

from dataclasses import dataclass
from hashlib import sha256

from app.core.ingestion.base import ParsedSection


@dataclass(frozen=True)
class Chunk:
    chunk_id: str        # unique store id: f"{section_id}#{chunk_index}"
    section_id: str      # stable id of the source section
    text: str            # text to embed
    metadata: dict       # flat, primitive-valued (Chroma-compatible)


def make_section_id(release: str, source: str, slug: str) -> str:
    return f"{release}/{source}/{slug}"


def section_to_chunks(section: ParsedSection, *, release: str) -> list[Chunk]:
    section_id = make_section_id(release, section.source, section.slug)

    # Prepend the title so the embedding carries the topic, not just the body.
    text = (
        f"{section.title}\n\n{section.text}".strip()
        if section.title else section.text
    )
    chunk_index = 0  # one chunk per section today; index is forward-compat
    metadata = {
        "release": release,
        "source": section.source,
        "section_id": section_id,
        "section_title": section.title,
        "locator": section.locator,          # URL — powers Stage 5 citations
        "chunk_index": chunk_index,
        "content_hash": sha256(text.encode()).hexdigest(),
    }
    return [
        Chunk(
            chunk_id=f"{section_id}#{chunk_index}",
            section_id=section_id,
            text=text,
            metadata=metadata,
        )
    ]


def chunk_sections(sections: list[ParsedSection], *, release: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(section_to_chunks(section, release=release))
    return chunks
