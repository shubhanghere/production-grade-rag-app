"""
Run eval experiments against the LangSmith golden dataset.

Run as:   python -m eval.run_eval [--variant dense|hybrid|rerank|all]
Requires: LANGSMITH_API_KEY, ANTHROPIC_API_KEY, VOYAGE_API_KEY in the environment.
Prereq:   python -m scripts.build_dataset   (creates/updates the dataset first)

Three retrieval variants share the SAME generation step, golden set, and
evaluators, so the experiments are directly comparable in LangSmith:

  dense   — Chroma similarity only
  hybrid  — dense + BM25 fused with RRF
  rerank  — hybrid pool re-scored by the Voyage cross-encoder (the full pipeline)

This is the "how much does each stage help?" comparison.
"""

import argparse
import asyncio

from dotenv import load_dotenv
from langchain_core.documents import Document
from langsmith import aevaluate

from app.config import settings
from app.core.rag import generate_answer
from app.core.retrieval import hybrid_search, retrieve
from app.core.vectorstore import get_vectorstore
from eval.evaluators import EVALUATORS
from scripts.build_dataset import DATASET_NAME

load_dotenv(override=True)  # LangSmith key for aevaluate (see build_dataset)


async def _dense(question: str) -> list[Document]:
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": settings.retrieval_k})
    return await retriever.ainvoke(question)


async def _hybrid(question: str) -> list[Document]:
    return (await hybrid_search(question))[: settings.retrieval_k]


async def _rerank(question: str) -> list[Document]:
    return await retrieve(question)


VARIANTS = {"dense": _dense, "hybrid": _hybrid, "rerank": _rerank}


def _make_target(retrieval_fn):
    async def target(inputs: dict) -> dict:
        docs = await retrieval_fn(inputs["question"])
        answer = await generate_answer(inputs["question"], docs)
        return {
            "answer": answer,
            "context": "\n\n---\n\n".join(d.page_content for d in docs),
            "retrieved_section_ids": [d.metadata.get("section_id") for d in docs],
        }

    return target


async def _run_variant(name: str) -> None:
    results = await aevaluate(
        _make_target(VARIANTS[name]),
        data=DATASET_NAME,
        evaluators=EVALUATORS,
        experiment_prefix=f"wms-26b-{name}",
        max_concurrency=4,
    )
    print(f"[{name}] experiment complete: {results.experiment_name}")


async def main() -> None:
    p = argparse.ArgumentParser(description="Run retrieval-variant eval experiments.")
    p.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    args = p.parse_args()

    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    for name in names:
        await _run_variant(name)


if __name__ == "__main__":
    asyncio.run(main())
