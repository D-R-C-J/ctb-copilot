def score_assessment(expected_case, actual_assessment):
    score_record ={
        "case_id": expected_case["id"],
        "outcome_match": expected_case["expected_outcome"] == actual_assessment["outcome"],
        "ctb_match": None,
        "mbft_match": None,
    }

    if expected_case["expected_outcome"] == "candidate":
        score_record["ctb_match"] = False
        score_record["mbft_match"] = False

        if actual_assessment["outcome"] == "candidate":
            expected_ids = expected_case["expected_passage_ids"]
            score_record["ctb_match"] = actual_assessment["ctb_passage_id"] == expected_ids[0]
            score_record["mbft_match"] = actual_assessment["mbft_passage_id"] == expected_ids[1]
    return score_record

## Uses expected case and passage IDs sent to Gemini
def score_retrieval(expected_case, retrieved_ids):
    if expected_case["expected_outcome"] != "candidate":
        return {
            "case_id": expected_case["id"],
            "ctb_retrieved": None,
            "mbft_retrieved": None,
            "both_retrieved": None
        }
    expected_ids = expected_case["expected_passage_ids"]
    retrieved_set = set(retrieved_ids)
    ctb_match = expected_ids[0] in retrieved_set
    mbft_match = expected_ids[1] in retrieved_set

    return {
        "case_id": expected_case["id"],
        "ctb_retrieved": ctb_match,
        "mbft_retrieved": mbft_match,
        "both_retrieved": ctb_match and mbft_match,
    }