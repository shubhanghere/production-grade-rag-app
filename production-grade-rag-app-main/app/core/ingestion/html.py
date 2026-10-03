"""
Parse a set of linked HTML pages into sections.

A "linked HTML document" is a sequence of pages connected by standard
rel="next" navigation — each page a self-contained topic with a heading,
optional subheadings, and body text. We walk the next-chain from a start page
to enumerate the whole document without executing JavaScript.

This module ONLY parses pages into ParsedSection records. Assigning stable ids,
chunking, and attaching metadata happen downstream. Source-specific quirks
(which element holds the body, how "next" is marked) are passed in by the
caller, so this parser stays generic across HTML sources.
"""

import time
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.core.ingestion.base import ParsedSection

# Web-standard defaults. Callers override for sources that don't follow them.
_DEFAULT_CONTENT_SELECTORS = ("main", "article", "div.body", "body")
_FETCH_RETRIES = 3


def _extract_content(soup: BeautifulSoup, selectors: tuple[str, ...]):
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            return node
    return None


def parse_page(
    html: str,
    *,
    source: str,
    url: str,
    content_selectors: tuple[str, ...] = _DEFAULT_CONTENT_SELECTORS,
) -> ParsedSection:
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""

    content = _extract_content(soup, content_selectors)
    if content:
        for noise in content.find_all(["script", "style", "nav"]):
            noise.decompose()

    subheadings = (
        [h.get_text(" ", strip=True) for h in content.find_all(["h2", "h3"])]
        if content else []
    )
    text = content.get_text("\n", strip=True) if content else ""

    slug = urlparse(url).path.rsplit("/", 1)[-1].removesuffix(".html")
    return ParsedSection(
        source=source, locator=url, slug=slug,
        title=title, subheadings=subheadings, text=text,
    )


def _next_url(html: str, current_url: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    # rel="next" is a web standard; it commonly appears as a <link> in the head
    # and/or an <a> in the body. Match either by looking for the rel alone.
    link = soup.find(attrs={"rel": "next"})
    href = link.get("href") if link else None
    return urljoin(current_url, href) if href else None


def crawl_html_pages(
    start_url: str,
    *,
    source: str,
    content_selectors: tuple[str, ...] = _DEFAULT_CONTENT_SELECTORS,
    max_pages: int = 500,
    delay_seconds: float = 0.5,
    client: httpx.Client | None = None,
) -> list[ParsedSection]:
    """Walk the rel=next chain from start_url, one ParsedSection per page.

    Polite (delay between requests) and guarded by both max_pages and a visited
    set, so a malformed/looping next-link can neither spin nor run away. Pages
    with no body text (covers/indexes) are skipped.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=60.0, follow_redirects=True)
    sections: list[ParsedSection] = []
    visited: set[str] = set()
    url: str | None = start_url
    try:
        while url and url not in visited and len(sections) < max_pages:
            visited.add(url)
            # Retry transient network errors (timeouts, dropped connections) so
            # one slow page can't abort a long crawl.
            for attempt in range(_FETCH_RETRIES):
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    break
                except httpx.TransportError:
                    if attempt == _FETCH_RETRIES - 1:
                        raise
                    time.sleep(2 ** attempt)
            html = resp.text
            section = parse_page(
                html, source=source, url=url,
                content_selectors=content_selectors,
            )
            if section.text.strip():
                sections.append(section)
            url = _next_url(html, url)
            if url:
                time.sleep(delay_seconds)
    finally:
        if owns_client:
            client.close()
    return sections
