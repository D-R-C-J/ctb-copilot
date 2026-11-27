def validate_citations(gemini_cited_ids: list[str], ids_sent_to_gemini: list[str]) -> bool:
    supplied_ids_set = set(ids_sent_to_gemini)
    for i in gemini_cited_ids:
        if i not in supplied_ids_set:
            return False
    return True