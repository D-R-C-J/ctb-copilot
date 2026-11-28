import json

import pytest

from ctb_copilot.corpus import load_passages


@pytest.fixture
def ctb_passage():
    return {
        "id": "ctb-a",
        "document_type": "CTB",
        "locator": "Artigo A",
        "url": "https://example.org/ctb",
        "source_version": "Versão fictícia",
        "verified_on": "2026-01-01",
        "text": "Trecho fictício sobre fiscalização.",
    }


@pytest.fixture
def mbft_passage(ctb_passage):
    return {
        **ctb_passage,
        "id": "mbft-a",
        "document_type": "MBFT",
        "locator": "Ficha A",
        "url": "https://example.org/mbft",
        "ctb_passage_id": ctb_passage["id"],
    }


def write_passages(path, *passages):
    lines = [json.dumps(passage, ensure_ascii=False) + "\n" for passage in passages]
    path.write_text("".join(lines), encoding="utf-8")


def test_load_passages_accepts_reference_to_later_ctb_passage(tmp_path, ctb_passage, mbft_passage):
    path = tmp_path / "passages.jsonl"
    write_passages(path, mbft_passage, ctb_passage)

    assert load_passages(path) == [mbft_passage, ctb_passage]


def test_load_passages_returns_empty_list_for_empty_file(tmp_path):
    path = tmp_path / "passages.jsonl"
    write_passages(path)

    assert load_passages(path) == []


def test_load_passages_rejects_missing_required_field(tmp_path, ctb_passage):
    path = tmp_path / "passages.jsonl"
    del ctb_passage["text"]
    write_passages(path, ctb_passage)

    with pytest.raises(ValueError, match="Missing required fields.*text"):
        load_passages(path)


def test_load_passages_rejects_unknown_document_type(tmp_path, ctb_passage):
    path = tmp_path / "passages.jsonl"
    write_passages(path, {**ctb_passage, "document_type": "UNKNOWN"})

    with pytest.raises(ValueError, match="Invalid document type"):
        load_passages(path)


def test_load_passages_rejects_duplicate_ids(tmp_path, ctb_passage):
    path = tmp_path / "passages.jsonl"
    write_passages(path, ctb_passage, ctb_passage)

    with pytest.raises(ValueError, match="Duplicate passage ID found: ctb-a"):
        load_passages(path)


@pytest.mark.parametrize("ctb_id", ["missing", "mbft-a"], ids=["missing-target", "non-ctb-target"])
def test_load_passages_rejects_invalid_mbft_reference(tmp_path, ctb_passage, mbft_passage, ctb_id):
    path = tmp_path / "passages.jsonl"
    write_passages(path, ctb_passage, {**mbft_passage, "ctb_passage_id": ctb_id})

    with pytest.raises(ValueError, match="MBFT passage references non-existent CTB passage"):
        load_passages(path)


def test_load_passages_rejects_malformed_json(tmp_path):
    path = tmp_path / "passages.jsonl"
    path.write_text('{"id":\n', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_passages(path)
