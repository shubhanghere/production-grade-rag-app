from collections import Counter

from eval.golden_set import GOLDEN_SET, RELEASE, by_query_type


def test_fifteen_examples():
    # TODO: rename (e.g. test_expected_example_count) and update this literal
    # once we append more examples beyond the initial 15.
    assert len(GOLDEN_SET) == 15


def test_ids_unique():
    ids = [ex.id for ex in GOLDEN_SET]
    assert len(set(ids)) == len(ids)


def test_balanced_five_each():
    counts = Counter(ex.query_type for ex in GOLDEN_SET)
    assert counts == {"definitional": 5, "configuration": 5, "api_reference": 5}


def test_every_example_well_formed():
    for ex in GOLDEN_SET:
        assert ex.question.strip()
        assert ex.reference_answer.strip()
        assert ex.release == RELEASE
        assert ex.gold_section_id  # every example maps to a real ingested section


def test_by_query_type_filters():
    assert {ex.id for ex in by_query_type("definitional")} == {
        "def-01", "def-02", "def-03", "def-04", "def-05"
    }
