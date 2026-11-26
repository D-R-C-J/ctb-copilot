import re
import string
from .index import build_index

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

def rank_semantic(query: str, index, model, limit: int=3) -> list[dict]:
    prepended_query: str = f"query: {query}"
    embedded_query = model.encode(prepended_query, normalize_embeddings=True)
    rank: list = []
    for i, k in index.items():
        if len(embedded_query) != len(k):
                raise ValueError("Query's vector numbers quantity different than index's")
        similarity = embedded_query @ k
        rank.append({"id":i, "score": similarity})
    rank.sort(key=lambda item: item["score"], reverse=True)
    return rank[:limit]

def rank_hybrid(lexical: list[dict], semantic: list[dict], limit: int = 3) -> list[dict]:
    totals: dict = {}
    records: list = []
    for ranks in (lexical, semantic):
        for position, hit in enumerate(ranks, 1):
            if totals.get(hit["id"], 0) == 0:
                totals[hit["id"]] = 0
            totals[hit["id"]] += 1 / position
    sorted_totals_pairs = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    for i, k in sorted_totals_pairs:
        records.append({"id": i, "score": k})
    return records[:limit]

            


    # takes semantic ranking, passages and limit and return an ordered list of passage ids
def expand_linked_evidence(semantic_ranking, passages, limit) -> list:
    if not semantic_ranking or limit <= 0: return []
    passages_by_id = {p["id"]: p for p in passages}
    first_id = semantic_ranking[0]["id"]
    ordered_list = [first_id]
    seen = {first_id}

    first_passage = passages_by_id.get(first_id)
    if first_passage is not None and first_passage["document_type"] == "CTB":
        matched_mbft = [p["id"] for p in passages if p["document_type"] == "MBFT" and p.get("ctb_passage_id") == first_id]

        if len(matched_mbft) == 1 and len(ordered_list) < limit:
            ordered_list.append(matched_mbft[0])
            seen.add(matched_mbft[0])
    for item in semantic_ranking[1:]:
        if len(ordered_list) >= limit:
            break
        passage_id = item["id"]
        if passage_id not in seen:
            ordered_list.append(passage_id)
            seen.add(passage_id)

    return ordered_list
