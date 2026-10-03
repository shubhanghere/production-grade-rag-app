"""Shared types for source parsers. Every parser (HTML, PDF, ...) emits
ParsedSection so downstream chunking is format-agnostic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedSection:
    source: str             # logical collection id supplied by the caller
    locator: str            # where it came from (URL, file path + anchor, ...)
    slug: str               # human-readable id hint, unique within a source
    title: str
    subheadings: list[str]
    text: str
