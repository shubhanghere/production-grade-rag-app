"""
LangSmith evaluators for the WMS Cloud 26B RAG eval harness.

Three evaluators, all returning binary 1/0 scores:

  retrieval_recall  (deterministic)  did retrieval surface the gold section?
  groundedness      (LLM judge)       is the answer supported by retrieved context?
  correctness       (LLM judge)       does the answer match the reference answer?

The judge uses a DIFFERENT model from the app under test (settings.eval_judge_model,
default claude-opus-4-8) so the system is not grading its own homework.

The target function (see eval/run_eval.py) must expose these run outputs:
  - "answer":                str
  - "context":               str   (concatenated retrieved doc text)
  - "retrieved_section_ids": list[str]
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field

from app.config import settings


# ---------- deterministic: retrieval recall ----------

def retrieval_recall(run: Run, example: Example) -> dict | None:
    """1 if the gold section was retrieved, else 0. Skipped (None) until the
    example has a gold_section_id assigned by Stage 2 ingestion."""
    gold_section_id = (example.metadata or {}).get("gold_section_id")
    if not gold_section_id:
        return None  # not yet scorable — LangSmith records no score for this example

    retrieved = run.outputs.get("retrieved_section_ids", []) if run.outputs else []
    score = 1 if gold_section_id in retrieved else 0
    return {"key": "retrieval_recall", "score": score}


# ---------- LLM judge plumbing ----------

class _Judgement(BaseModel):
    score: int = Field(description="1 if the criterion is met, 0 if not")
    reasoning: str = Field(description="One sentence justification")


def _judge_llm():
    # Separate model from the app's generator → independent judge.
    return ChatAnthropic(
        model=settings.eval_judge_model,
        anthropic_api_key=settings.anthropic_api_key,
    ).with_structured_output(_Judgement)


_GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict grader. Decide whether the ANSWER is fully supported by "
     "the CONTEXT. Score 1 only if every claim in the answer is backed by the "
     "context; score 0 if any claim is unsupported or contradicted. A refusal "
     "for lack of information, when the context indeed lacks it, scores 1."),
    ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
])

_CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict grader. Decide whether the ANSWER is consistent with the "
     "REFERENCE answer for the QUESTION. Score 1 if it conveys the same essential "
     "facts (wording may differ); score 0 if it is missing key facts, adds wrong "
     "facts, or contradicts the reference."),
    ("human", "QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer}"),
])


def groundedness(run: Run, example: Example) -> dict:
    chain = _GROUNDEDNESS_PROMPT | _judge_llm()
    result = chain.invoke({
        "context": (run.outputs or {}).get("context", ""),
        "answer": (run.outputs or {}).get("answer", ""),
    })
    return {"key": "groundedness", "score": result.score, "comment": result.reasoning}


def correctness(run: Run, example: Example) -> dict:
    chain = _CORRECTNESS_PROMPT | _judge_llm()
    result = chain.invoke({
        "question": (example.inputs or {}).get("question", ""),
        "reference": (example.outputs or {}).get("reference_answer", ""),
        "answer": (run.outputs or {}).get("answer", ""),
    })
    return {"key": "correctness", "score": result.score, "comment": result.reasoning}


EVALUATORS = [retrieval_recall, groundedness, correctness]
