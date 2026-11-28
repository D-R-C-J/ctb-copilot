import pytest
from ctb_copilot.evaluate import score_assessment, score_retrieval

CANDIDATE_CASE = {
    "id": "case-002",
    "expected_outcome": "candidate",
    "expected_passage_ids": ["ctb-244-ii", "mbft-704-81"],
}
CLARIFICATION_CASE = {
    "id": "case-001",
    "expected_outcome": "ask_clarification",
}

CORRECT_CANDIDATE = {
    "outcome": "candidate",
    "ctb_passage_id": "ctb-244-ii",
    "mbft_passage_id": "mbft-704-81",
}
CLARIFICATION = {
    "outcome": "ask_clarification",
    "question": "A pessoa sem capacete conduzia a moto ou estava na garupa?",
}

@pytest.mark.parametrize(
    ("expected_case", "actual_assessment", "expected_score"),
    [
        pytest.param(
            CANDIDATE_CASE,
            CORRECT_CANDIDATE,
            {"case_id": "case-002", "outcome_match": True, "ctb_match": True, "mbft_match": True},
            id="correct-pair",
        ),
        pytest.param(
            CANDIDATE_CASE,
            {**CORRECT_CANDIDATE, "mbft_passage_id": "mbft-703-03"},
            {"case_id": "case-002", "outcome_match": True, "ctb_match": True, "mbft_match": False},
            id="wrong-ficha",
        ),
        pytest.param(
            CANDIDATE_CASE,
            CLARIFICATION,
            {"case_id": "case-002", "outcome_match": False, "ctb_match": False, "mbft_match": False},
            id="unexpected-clarification",
        ),
        pytest.param(
            CLARIFICATION_CASE,
            CLARIFICATION,
            {"case_id": "case-001", "outcome_match": True, "ctb_match": None, "mbft_match": None},
            id="expected-clarification",
        ),
    ],
)

def test_score_assessment(expected_case, actual_assessment, expected_score):
    assert score_assessment(expected_case, actual_assessment) == expected_score

@pytest.mark.parametrize(
    ("expected_case", "retrieved_ids", "expected_score"),
    [
        pytest.param(
            CANDIDATE_CASE,
            ["ctb-244-ii", "mbft-704-81"],
            {
                "case_id": "case-002",
                "ctb_retrieved": True,
                "mbft_retrieved": True,
                "both_retrieved": True,
            },
            id="both-retrieved",
        ),
        pytest.param(
            CANDIDATE_CASE,
            ["ctb-244-ii", "mbft-703-03"],
            {
                "case_id": "case-002",
                "ctb_retrieved": True,
                "mbft_retrieved": False,
                "both_retrieved": False,
            },
            id="ctb-only",
        ),
        pytest.param(
            CANDIDATE_CASE,
            ["mbft-704-81"],
            {
                "case_id": "case-002",
                "ctb_retrieved": False,
                "mbft_retrieved": True,
                "both_retrieved": False,
            },
            id="ficha-only",
        ),
        pytest.param(
            CANDIDATE_CASE,
            [],
            {
                "case_id": "case-002",
                "ctb_retrieved": False,
                "mbft_retrieved": False,
                "both_retrieved": False,
            },
            id="empty-results",
        ),
        pytest.param(
            CLARIFICATION_CASE,
            ["ctb-244-ii"],
            {
                "case_id": "case-001",
                "ctb_retrieved": None,
                "mbft_retrieved": None,
                "both_retrieved": None,
            },
            id="clarification-with-results",
        ),
    ],
)

def test_score_retrieval(expected_case, retrieved_ids, expected_score):
    assert score_retrieval(expected_case, retrieved_ids) == expected_score