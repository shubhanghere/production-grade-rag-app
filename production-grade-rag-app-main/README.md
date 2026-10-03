# Production-Grade RAG App

[![CI](https://github.com/jasalazar/production-grade-rag-app/actions/workflows/ci.yml/badge.svg)](https://github.com/jasalazar/production-grade-rag-app/actions/workflows/ci.yml)

![WMS Wiki demo — asking a question and getting a cited answer](assets/demo.gif)

A domain-agnostic, production-oriented Retrieval-Augmented Generation (RAG) engine built on LangChain and FastAPI. It ingests a documentation corpus, retrieves the most relevant passages with a **hybrid + reranked** pipeline, and answers questions with an LLM — returning **source citations** with every answer.

The engine is deliberately **corpus-neutral**: nothing about any specific dataset lives in the application code. A corpus is supplied at runtime — a URL to crawl, or files uploaded through the UI — so the same engine serves any documentation set.

## Retrieval pipeline

```
ingest:  documents → section-aware chunks (+ metadata) → Chroma (dense) + BM25 (lexical)
query:   dense + BM25 (in parallel) → Reciprocal Rank Fusion → cross-encoder rerank → top-k → LLM → answer + citations
```

- **Section-aware chunking** — documents are split along their own structure (one chunk per section), each with a stable `section_id` and metadata (`release`, `source`, `section_title`, `url`, `content_hash`). Retrieval and citations key off these.
- **Hybrid retrieval** — dense semantic search (Voyage embeddings in Chroma) runs alongside BM25 keyword search, so both paraphrases *and* exact terms (codes, field names) are found.
- **Reciprocal Rank Fusion (RRF)** — the two ranked lists are merged by rank position, which is scale-agnostic and needs no score normalization.
- **Reranking** — a Voyage cross-encoder (`rerank-2.5`) re-scores the fused candidate pool for a sharper final ordering.
- **Citations** — every answer surfaces the source sections (title + link) it drew from.
- **Injection-hardened prompting** — retrieved context and the user question are fenced as untrusted data; the system prompt's rules cannot be overridden by their contents.

## Measurement-first

Retrieval and answer quality are evaluated against a **golden set** via LangSmith, using three binary evaluators:

- `retrieval_recall` (deterministic) — did retrieval surface the correct section?
- `groundedness` (LLM judge) — is the answer supported by the retrieved context?
- `correctness` (LLM judge) — does the answer match the reference answer?

The judge uses a **different** model from the one under test, so the system never grades its own homework. Because each retrieval stage is independently callable, the variants (dense → hybrid → hybrid + rerank) can be compared head-to-head on these metrics.

## Ingestion

Two ways in, both feeding the same *chunk → embed → store* pipeline:

- **Bulk** — `scripts/ingest_html.py` crawls a linked HTML documentation source (standard `rel="next"` navigation) and ingests the whole set. Corpus specifics are passed as CLI arguments, so the engine stays generic.
- **Ad-hoc** — the `/documents` endpoint (and the UI's upload panel) ingests a single pasted or uploaded document.

Ingestion is **idempotent and cost-aware**: chunks are stored under a stable id (re-ingesting never duplicates), and a chunk is re-embedded only when its content actually changed. Bulk ingestion commits incrementally and respects the embedding provider's rate limits (token-aware batching + request pacing + backoff), so a long run is **resumable**.

## Tech stack

| Concern | Choice |
|---|---|
| API / server | FastAPI |
| Orchestration | LangChain |
| Vector store | Chroma (persistent, local) |
| Embeddings & reranking | Voyage AI (`voyage-3`, `rerank-2.5`) |
| Generation | Anthropic Claude |
| Eval & tracing | LangSmith |
| Tests | pytest |

## Project layout

```
app/
  core/
    ingestion/      HTML source parsing → ParsedSection
    chunking.py     ParsedSection → Chunk (stable ids + metadata)
    vectorstore.py  Chroma + Voyage embeddings (rate-limit aware)
    retrieval.py    hybrid search + RRF + rerank
    rag.py          ingest, query, citations
  api/              FastAPI routes (chat, documents)
  schemas/          request/response models
eval/               golden set, evaluators, experiment runner
scripts/            ingest + dataset-builder CLIs
tests/              unit tests
frontend/           minimal UI (chat + ingest)
```

## Setup

Requires Python 3 (developed on 3.14) and API keys for Anthropic, Voyage AI, and LangSmith.

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Create a `.env` with your keys:
   ```
   ANTHROPIC_API_KEY=...
   VOYAGE_API_KEY=...
   LANGSMITH_API_KEY=...
   ```

## Usage

**Ingest a corpus** (bulk, from a linked-HTML source):
```
python -m scripts.ingest_html \
    --start-url https://example.com/docs/index.html \
    --source mydocs --release v1 --content-selector div.body
```
Add `--dry-run` to validate the crawl and chunking without embedding or storing anything.

**Run the app:**
```
uvicorn app.main:app --reload --port 8001
```
Then open http://127.0.0.1:8001/ — ask questions in the chat panel, or add documents in the ingest panel.

## Testing

```
pytest -v
```

Tests run automatically in CI (GitHub Actions) on every push and pull request — see `.github/workflows/ci.yml`.

## Evaluation

Build the golden dataset and run an experiment (requires `LANGSMITH_API_KEY`):
```
python -m scripts.build_dataset
python -m eval.run_eval
```

## Roadmap

- **Conditional / doc-type-aware ingestion** — detect a document's structure and route it to the right parser (HTML, PDF, and structure-agnostic fallbacks), and adapt retrieval to the query type. Related items under consideration: image/diagram handling, dynamic result counts driven by rerank scores, and output-sanitization / observability hardening.
