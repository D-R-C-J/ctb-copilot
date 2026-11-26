from ctb_copilot.search import rank_lexical, expand_linked_evidence, rank_semantic
from ctb_copilot.corpus import load_passages
import pytest
import pathlib

@pytest.fixture
def passages():
    path = pathlib.Path(__file__).resolve().parent.parent / "data" / "passages.jsonl"
    return load_passages(path)

@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("colete", [{"id": "mbft-703-03", "score": 1}]),
        ("COLETE, colete", [{"id": "mbft-703-03", "score": 1}]),
        ("semaforo", []),
        ("703-03", [{'id': 'mbft-703-03', 'score': 2}]),
        ("999-99", [])
    ],
)
def test_lexical_search(passages: list[dict], query: str, expected: list[dict]) -> None:
    assert rank_lexical(query, passages, 5) == expected

@pytest.mark.parametrize(
    ("semantic_ranking", "limit", "expected"),
    [
        (["ctb-244-ii", "ctb-244-x", "ctb-244-i"], 3, ["ctb-244-ii", "mbft-704-81", "ctb-244-x"]),
        (["ctb-244-ii", "ctb-244-x", "mbft-704-81"], 3, ["ctb-244-ii", "mbft-704-81", "ctb-244-x"]),
        (["ctb-244-i", "ctb-244-x", "mbft-703-01"], 3, ["ctb-244-i", "ctb-244-x", "mbft-703-01"]),
        ([], 3, []),
        (["ctb-244-ii"], 0, []),
        (["ctb-244-ii", "ctb-244-x"], 1, ["ctb-244-ii"])
    ],
)

def test_linked_evidence(semantic_ranking, passages, limit, expected):
    ranked_dicts = [{"id": pid, "score": 0.9 - (i * 0.1)} for i, pid in enumerate(semantic_ranking)]
    assert expand_linked_evidence(ranked_dicts, passages, limit) == expected

@pytest.fixture
def semantic_inputs():
    import numpy as np

    class FakeModel:
        def encode(self, text, *, normalize_embeddings):
            assert text == "query: piloto sem capacete"
            assert normalize_embeddings is True
            return np.array([1.0, 0.0])

    # Put the weaker match first so the test can detect a missing sort.
    index = {
        "weaker": np.array([0.0, 1.0]),
        "stronger": np.array([0.8, 0.6]),
    }
    return index, FakeModel()


@pytest.mark.parametrize(
    ("limit", "expected_ids", "expected_scores"),
    [
        (2, ["stronger", "weaker"], [0.8, 0.0]),
        (1, ["stronger"], [0.8]),
    ],
)
def test_rank_semantic(semantic_inputs, limit, expected_ids, expected_scores):
    index, model = semantic_inputs

    results = rank_semantic("piloto sem capacete", index, model, limit)

    assert [hit["id"] for hit in results] == expected_ids
    assert [hit["score"] for hit in results] == pytest.approx(expected_scores)


def test_rank_semantic_empty_index(semantic_inputs):
    _, model = semantic_inputs

    assert rank_semantic("piloto sem capacete", {}, model) == []


def test_rank_semantic_rejects_mismatched_dimensions(semantic_inputs):
    index, model = semantic_inputs
    mismatched_index = {"weaker": index["weaker"][:1]}

    with pytest.raises(ValueError, match="Query's vector numbers quantity"):
        rank_semantic("piloto sem capacete", mismatched_index, model)
