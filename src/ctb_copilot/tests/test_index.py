from unittest.mock import Mock

import numpy as np
import pytest

from ctb_copilot.index import build_index


def test_build_index_pairs_passage_ids_with_embeddings():
    passages = [
        {"id": "ctb-a", "text": "Sem capacete."},
        {"id": "mbft-a", "text": "Passageiro na garupa."},
    ]
    model = Mock()
    model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])

    index = build_index(passages, model)

    model.encode.assert_called_once_with(
        ["passage: Sem capacete.", "passage: Passageiro na garupa."],
        normalize_embeddings=True,
    )
    assert {passage_id: vector.tolist() for passage_id, vector in index.items()} == {
        "ctb-a": [1.0, 0.0],
        "mbft-a": [0.0, 1.0],
    }


def test_build_index_returns_empty_index_for_empty_passages():
    model = Mock()
    model.encode.return_value = []

    assert build_index([], model) == {}


@pytest.mark.parametrize("embedding_count", [1, 3], ids=["too-few", "too-many"])
def test_build_index_rejects_mismatched_embedding_count(embedding_count):
    passages = [{"id": "first", "text": "Primeiro."}, {"id": "second", "text": "Segundo."}]
    model = Mock()
    model.encode.return_value = np.zeros((embedding_count, 2))

    with pytest.raises(ValueError):
        build_index(passages, model)
