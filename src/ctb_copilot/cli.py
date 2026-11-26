import argparse
from .search import rank_lexical

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
    return parser

def run_cli(passages: list[dict]):
    args = build_parser().parse_args()
    results = rank_lexical(args.query, passages, 3)
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
    

