from ctb_copilot.search import rank_lexical
from ctb_copilot.corpus import load_passages
import pytest
import pathlib

@pytest.fixture
def passages():
    path = pathlib.Path(__file__).parent.parent / "src" / "ctb_copilot" / "data" / "passages.jsonl"
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
def test_search(passages: list[dict], query: str, expected: list[dict]) -> None:
    assert rank_lexical(query, passages, 5) == expected