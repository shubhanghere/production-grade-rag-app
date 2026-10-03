"""Unit tests for the linked-HTML parser (app/core/ingestion/html.py).

All offline: parse_page/_next_url take HTML strings, and crawl_html_pages takes
an injected fake client so no network is touched.
"""

from app.core.ingestion.html import crawl_html_pages, parse_page, _next_url


def _page(title="", body="", next_href=None, *, container='<div class="body">'):
    head = f'<link rel="next" href="{next_href}">' if next_href else ""
    h1 = f"<h1>{title}</h1>" if title else ""
    return (
        f"<html><head>{head}</head><body>{h1}"
        f"{container}{body}</div></body></html>"
    )


# ---------- parse_page ----------

def test_parse_page_extracts_fields():
    html = _page(
        title="Intro",
        body="<h2>Sub A</h2><h3>Sub B</h3><p>Body text here.</p>",
    )
    sec = parse_page(html, source="book", url="https://x/y/page1.html")
    assert sec.title == "Intro"
    assert sec.slug == "page1"            # derived from URL filename
    assert sec.source == "book"
    assert sec.locator == "https://x/y/page1.html"
    assert sec.subheadings == ["Sub A", "Sub B"]
    assert "Body text here." in sec.text


def test_parse_page_strips_noise():
    html = _page(title="T", body="<p>keep</p><script>evil()</script>")
    sec = parse_page(html, source="book", url="https://x/p.html")
    assert "keep" in sec.text
    assert "evil()" not in sec.text       # <script> decomposed


def test_parse_page_honors_content_selectors():
    html = "<html><body><h1>T</h1><main><p>in main</p></main></body></html>"
    sec = parse_page(
        html, source="book", url="https://x/p.html",
        content_selectors=("main",),
    )
    assert "in main" in sec.text


# ---------- _next_url ----------

def test_next_url_resolves_relative_link():
    html = _page(title="T", body="<p>b</p>", next_href="page2.html")
    assert _next_url(html, "https://x/y/page1.html") == "https://x/y/page2.html"


def test_next_url_none_when_absent():
    html = _page(title="End", body="<p>b</p>")
    assert _next_url(html, "https://x/y/page1.html") is None


# ---------- crawl_html_pages ----------

class _FakeResp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class _FakeClient:
    def __init__(self, pages):
        self.pages = pages

    def get(self, url):
        return _FakeResp(self.pages[url])

    def close(self):
        pass


def _crawl(pages, start, **kw):
    return crawl_html_pages(
        start, source="book", client=_FakeClient(pages),
        delay_seconds=0, **kw,
    )


def test_crawl_walks_chain_in_order():
    pages = {
        "https://x/p1.html": _page("One", "<p>a</p>", next_href="p2.html"),
        "https://x/p2.html": _page("Two", "<p>b</p>", next_href="p3.html"),
        "https://x/p3.html": _page("Three", "<p>c</p>"),
    }
    secs = _crawl(pages, "https://x/p1.html")
    assert [s.slug for s in secs] == ["p1", "p2", "p3"]
    assert [s.title for s in secs] == ["One", "Two", "Three"]


def test_crawl_skips_empty_body_pages():
    pages = {
        "https://x/cover.html": _page(body="", next_href="real.html"),
        "https://x/real.html": _page("Real", "<p>content</p>"),
    }
    secs = _crawl(pages, "https://x/cover.html")
    assert [s.slug for s in secs] == ["real"]   # empty cover skipped


def test_crawl_respects_max_pages():
    pages = {
        "https://x/p1.html": _page("One", "<p>a</p>", next_href="p2.html"),
        "https://x/p2.html": _page("Two", "<p>b</p>", next_href="p3.html"),
        "https://x/p3.html": _page("Three", "<p>c</p>"),
    }
    secs = _crawl(pages, "https://x/p1.html", max_pages=2)
    assert [s.slug for s in secs] == ["p1", "p2"]


def test_crawl_retries_transient_fetch_errors(monkeypatch):
    import httpx
    monkeypatch.setattr("app.core.ingestion.html.time.sleep", lambda s: None)

    class _FlakyClient(_FakeClient):
        def __init__(self, pages, fails):
            super().__init__(pages)
            self.fails = fails
            self.calls = 0

        def get(self, url):
            self.calls += 1
            if self.calls <= self.fails:
                raise httpx.ReadTimeout("transient")
            return super().get(url)

    pages = {"https://x/p1.html": _page("One", "<p>a</p>")}  # single page, no next
    secs = crawl_html_pages(
        "https://x/p1.html", source="book",
        client=_FlakyClient(pages, fails=2), delay_seconds=0,
    )
    assert [s.slug for s in secs] == ["p1"]   # survived 2 transient failures


def test_crawl_does_not_loop():
    pages = {
        "https://x/a.html": _page("A", "<p>a</p>", next_href="b.html"),
        "https://x/b.html": _page("B", "<p>b</p>", next_href="a.html"),  # back to a
    }
    secs = _crawl(pages, "https://x/a.html")
    assert [s.slug for s in secs] == ["a", "b"]  # visited-set stops the loop
