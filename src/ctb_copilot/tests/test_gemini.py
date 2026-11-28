import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from ctb_copilot.gemini import generate_assessment


@pytest.fixture
def fake_client():
    return SimpleNamespace(interactions=SimpleNamespace(create=Mock()))


@pytest.mark.parametrize(
    "response",
    [
        {
            "outcome": "candidate",
            "ctb_passage_id": "ctb-a",
            "mbft_passage_id": "mbft-a",
            "explanation": "Proposta fictícia.",
        },
        {"outcome": "ask_clarification", "question": "Quem conduzia a moto?"},
        {"outcome": "insufficient_evidence", "explanation": "As fontes não cobrem o relato."},
    ],
    ids=["candidate", "clarification", "insufficient-evidence"],
)
def test_generate_assessment_parses_response_and_usage(fake_client, response):
    fake_client.interactions.create.return_value = SimpleNamespace(
        output_text=json.dumps(response),
        usage=SimpleNamespace(total_input_tokens=100, total_output_tokens=20, total_tokens=120),
    )
    passages = [{"id": "ctb-a", "text": "Trecho fictício.", "score": 0.8}]

    result = generate_assessment("Pessoa sem capacete.", passages, fake_client, "fake-model")

    assert result == {
        **response,
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    }
    fake_client.interactions.create.assert_called_once()
    request = fake_client.interactions.create.call_args.kwargs
    assert request["model"] == "fake-model"
    assert request["response_format"]["mime_type"] == "application/json"
    assert json.loads(request["input"]) == {
        "observation": "Pessoa sem capacete.",
        "evidence": [{"id": "ctb-a", "text": "Trecho fictício."}],
    }


@pytest.mark.parametrize(
    "response",
    [
        {"question": "Quem conduzia a moto?"},
        {"outcome": "ask_clarification"},
        {"outcome": "insufficient_evidence"},
        {"outcome": "candidate", "ctb_passage_id": "ctb-a", "explanation": "Proposta fictícia."},
    ],
    ids=["missing-outcome", "missing-question", "missing-explanation", "missing-candidate-id"],
)
def test_generate_assessment_rejects_incomplete_response(fake_client, response):
    fake_client.interactions.create.return_value = SimpleNamespace(
        output_text=json.dumps(response),
        usage=None,
    )

    with pytest.raises(ValidationError):
        generate_assessment("Pessoa sem capacete.", [], fake_client, "fake-model")
