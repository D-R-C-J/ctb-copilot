import argparse
from .search import rank_lexical, rank_semantic, expand_linked_evidence
from .index import build_index
from sentence_transformers import SentenceTransformer

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctb-copilot",
        description="Searches CTB infractions articles and MBFT records."
    )
    subparser = parser.add_subparsers(dest="command", required=True)
    search_parser = subparser.add_parser(
        "search",
        help="Search CTB and MBFT.",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Traffic scenario, legal keywords etc."
    )
    search_parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["lexical", "semantic", "linked"],
        default="lexical",
        help="Search method to use: lexical, semantic or linked (default: lexical)"
    )
    return parser

def run_cli(passages: list[dict], argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    match args.method:
        case "lexical":
            results = rank_lexical(args.query, passages, 3)
        case "semantic":
            model = SentenceTransformer("intfloat/multilingual-e5-small")
            idx = build_index(passages, model)
            results = rank_semantic(args.query, idx, model, 3)
        case "linked":
            model = SentenceTransformer("intfloat/multilingual-e5-small")
            idx = build_index(passages, model)
            rs = rank_semantic(args.query, idx, model, 3)
            scores_by_id = {item["id"]: item["score"] for item in rs}
            ordered_ids = expand_linked_evidence(rs, passages, 3)
            results = [{"id": pid, "score": scores_by_id.get(pid)} for pid in ordered_ids]
    
    final = []
    if not results:
        no_result = "Nenhum resultado encontrado."
        print(no_result)
        return
    locators_by_id = {p["id"]: p.get("locator") for p in passages}
    text_by_id = {p["id"]: p.get("text") for p in passages}
    for r in results:
        final.append({
            "id": r["id"],
            "score": r["score"],
            "locator": locators_by_id.get(r["id"]),
            "text": text_by_id.get(r["id"]),
        })
    for f in final:
        print(f"{f}\n")
    return final
    

