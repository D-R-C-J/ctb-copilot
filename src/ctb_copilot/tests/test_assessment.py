import pytest
from ctb_copilot.assessment import validate_citations

@pytest.mark.parametrize(
    ("gemini_ids", "ids_sent", "expected"),
    [
        (["ctb-244-ii", "mbft-704-81"], ["ctb-244-ii", "mbft-704-81", "ctb-244-x"], True),
        (["ctb-244-ii", "mbft-703-03"], ["ctb-244-ii", "mbft-704-81", "ctb-244-x"], False),
        ([], ["ctb-244-ii", "mbft-704-81", "ctb-244-x"], True),
        (["ctb-244-ii", "mbft-703-03"], [], False),       
    ],
)

def test_validate_citations(gemini_ids, ids_sent, expected):
    assert validate_citations(gemini_ids, ids_sent) == expected
