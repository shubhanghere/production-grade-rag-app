from types import SimpleNamespace

from eval.evaluators import retrieval_recall


def _run(retrieved_ids):
    return SimpleNamespace(outputs={"retrieved_section_ids": retrieved_ids})


def _example(gold_section_id):
    return SimpleNamespace(metadata={"gold_section_id": gold_section_id})


def test_recall_skipped_when_no_gold_id():
    # gold_section_id unset (Stage 1 reality) → None means "no score recorded".
    assert retrieval_recall(_run(["a", "b"]), _example(None)) is None


def test_recall_hit_scores_1():
    result = retrieval_recall(_run(["x", "gold-42"]), _example("gold-42"))
    assert result == {"key": "retrieval_recall", "score": 1}


def test_recall_miss_scores_0():
    result = retrieval_recall(_run(["x", "y"]), _example("gold-42"))
    assert result == {"key": "retrieval_recall", "score": 0}


def test_recall_handles_empty_outputs():
    # Defensive: a target that returned nothing shouldn't crash the evaluator.
    result = retrieval_recall(SimpleNamespace(outputs=None), _example("gold-42"))
    assert result == {"key": "retrieval_recall", "score": 0}
