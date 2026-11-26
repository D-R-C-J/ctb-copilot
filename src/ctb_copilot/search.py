import re
import string

def rank_lexical(query: str, passages: list[dict], limit: int = 5) -> list[dict]:
    pattern = f"[{re.escape(string.punctuation)}\\s]+"
    query_set = set(q.lower().strip() for q in re.split(pattern, query) if q.lower().strip())
    rank: list[dict] = []
    for passage in passages:
        count = 0
        passage_set = set(p.lower().strip() for p in re.split(pattern, passage["text"]) if p.lower().split())
        for q in query_set:
            if q in passage_set:
                count += 1
        if count > 0:
            rank.append({"id": passage["id"], "score": count})
    rank.sort(key=lambda item: item["score"], reverse=True)
    return rank[:limit]