from unittest.mock import Mock

import pytest

from ctb_copilot import cli


@pytest.fixture
def passages():
    return [
        {"id": "ctb-a", "text": "capacete", "locator": "Artigo A", "url": "https://example.org/ctb"},
        {"id": "mbft-a", "text": "capacete", "locator": "Ficha A", "url": "https://example.org/mbft"},
        {"id": "mbft-b", "text": "viseira", "locator": "Ficha B", "url": "https://example.org/other"},
    ]


@pytest.fixture
def fake_generation(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", Mock())
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL_NAME", "fake-model")
    monkeypatch.setattr(cli.genai, "Client", Mock(return_value=object()))
    generate = Mock()
    monkeypatch.setattr(cli, "generate_assessment", generate)
    return generate


@pytest.mark.parametrize(
    ("command", "method", "limit"),
    [("search", "lexical", 3), ("ask", "semantic", 5)],
)
def test_parser_defaults(command, method, limit):
    args = cli.build_parser().parse_args([command, "capacete"])

    assert (args.command, args.query, args.method, args.limit) == (command, "capacete", method, limit)


def test_search_displays_results_without_gemini(passages, monkeypatch, capsys):
    make_client = Mock(side_effect=AssertionError("Search should run offline"))
    monkeypatch.setattr(cli.genai, "Client", make_client)

    results = cli.run_cli(passages, ["search", "capacete", "--limit", "1"])

    assert [hit["id"] for hit in results] == ["ctb-a"]
    assert "Artigo A" in capsys.readouterr().out
    make_client.assert_not_called()


def test_ask_prints_candidate_sources(passages, fake_generation, capsys):
    fake_generation.return_value = {
        "outcome": "candidate",
        "ctb_passage_id": "ctb-a",
        "mbft_passage_id": "mbft-a",
        "explanation": "Proposta fictícia.",
    }

    result = cli.run_cli(passages, ["ask", "capacete", "--method", "lexical"])

    observation, evidence, _, model_name = fake_generation.call_args.args
    assert observation == "capacete"
    sent_ids = [passage["id"] for passage in evidence]
    assert sent_ids == ["ctb-a", "mbft-a"]
    assert result["retrieved_ids"] == sent_ids
    assert model_name == "fake-model"
    assert result["ctb_url"] == passages[0]["url"]
    assert result["mbft_url"] == passages[1]["url"]
    output = capsys.readouterr().out
    assert result["explanation"] in output
    assert result["ctb_url"] in output and result["mbft_url"] in output


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"outcome": "ask_clarification", "question": "Quem conduzia a moto?"}, "Quem conduzia a moto?"),
        (
            {"outcome": "insufficient_evidence", "explanation": "As fontes não cobrem o relato."},
            "As fontes não cobrem o relato.",
        ),
    ],
)
def test_ask_displays_non_candidate_response(passages, fake_generation, capsys, response, message):
    fake_generation.return_value = response.copy()

    result = cli.run_cli(passages, ["ask", "capacete", "--method", "lexical"])

    assert result == {**response, "retrieved_ids": ["ctb-a", "mbft-a"]}
    output = capsys.readouterr().out
    assert message in output
    assert "Fontes:" not in output


def test_ask_rejects_passage_outside_retrieved_evidence(passages, fake_generation):
    fake_generation.return_value = {
        "outcome": "candidate",
        "ctb_passage_id": "ctb-a",
        "mbft_passage_id": "mbft-b",
        "explanation": "Proposta fictícia.",
    }

    with pytest.raises(ValueError, match="against retrieved evidence"):
        cli.run_cli(passages, ["ask", "capacete", "--method", "lexical"])
