"""
Ingest a linked HTML documentation source into the vector store.

Generic over any HTML source that uses rel="next" navigation; corpus specifics
(start URL, source id, release, content selector) are passed as CLI args, so the
engine stays domain-agnostic.

Example (the Online Help source):
  python -m scripts.ingest_html \
      --start-url https://docs.oracle.com/en/cloud/saas/warehouse-management/26b/owmol/index.html \
      --source owmol --release 26b --content-selector div.body

NOTE: running this performs a live crawl AND live embedding. Use --dry-run to
validate crawl+chunk without embedding or storing anything.
"""

import argparse

from app.core.chunking import chunk_sections
from app.core.ingestion.html import crawl_html_pages
from app.core.rag import ingest_sections


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest a linked HTML source.")
    p.add_argument("--start-url", required=True)
    p.add_argument("--source", required=True,
                   help="Logical collection id stored in metadata (e.g. owmol).")
    p.add_argument("--release", required=True,
                   help="Release tag stored in metadata (e.g. 26b).")
    p.add_argument("--content-selector", action="append", dest="content_selectors",
                   help="CSS selector(s) for the body region; repeatable. "
                        "Omit to use web-standard containers.")
    p.add_argument("--max-pages", type=int, default=500)
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--keep-untitled", action="store_true",
                   help="Keep pages with no <h1> (cover/index). Default: skip them.")
    p.add_argument("--dry-run", action="store_true",
                   help="Crawl + chunk and report counts; no embedding or storage.")
    p.add_argument("--commit-group", type=int, default=8,
                   help="Sections written to Chroma per commit. Smaller = more "
                        "resumable (a re-run skips already-stored chunks). Default 8.")
    args = p.parse_args()

    crawl_kwargs = {"source": args.source, "max_pages": args.max_pages,
                    "delay_seconds": args.delay}
    if args.content_selectors:
        crawl_kwargs["content_selectors"] = tuple(args.content_selectors)

    sections = crawl_html_pages(args.start_url, **crawl_kwargs)
    print(f"Crawled {len(sections)} non-empty page(s).")

    if not args.keep_untitled:
        kept = [s for s in sections if s.title.strip()]
        if len(kept) != len(sections):
            print(f"Skipped {len(sections) - len(kept)} untitled page(s) (cover/index).")
        sections = kept

    if args.dry_run:
        chunks = chunk_sections(sections, release=args.release)
        print(f"[dry-run] {len(sections)} sections -> {len(chunks)} chunks. "
              "No embedding/storage.")
        for c in chunks[:5]:
            print("  ", c.section_id)
        return

    # Commit in small groups so each is persisted to Chroma immediately. With
    # content-hash idempotency, a re-run skips already-stored chunks and resumes.
    total = len(sections)
    done = 0
    for i in range(0, total, args.commit_group):
        ingest_sections(sections[i:i + args.commit_group], release=args.release)
        done += len(sections[i:i + args.commit_group])
        print(f"  ingested {done}/{total} sections", flush=True)
    print(f"Done — {total} section(s) ingested "
          f"(release={args.release}, source={args.source}).")


if __name__ == "__main__":
    main()
