import argparse
from .search import rank_lexical, rank_semantic, expand_linked_evidence, rank_hybrid
from .index import build_index
from .gemini import generate_assessment
from .assessment import validate_citations
from google import genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

### PARSER HELPERs ###

def add_query(parser):
    parser.add_argument(
        "query",
        type=str,
        help="Traffic scenario, legal keywords etc."
    )

def add_method(parser, default_method: str = "lexical"):
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["lexical", "semantic", "hybrid", "linked"],
        default=default_method,
        help=f"Search method to use: lexical, semantic, hybrid or linked (default: {default_method})"
    )

def add_limit(parser, default_limit: int = 3):
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=default_limit,
        help=f"Limit results for the search (default: {default_limit})."
    )

###########################################################################

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
    ask_parser = subparser.add_parser(
        "ask",
        help="Use Gemini API for the search."
    )
    add_query(search_parser)
    add_query(ask_parser)
    add_method(search_parser)
    add_method(ask_parser, "semantic")
    add_limit(search_parser)
    add_limit(ask_parser, 5)
    return parser

def run_cli(passages: list[dict], argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    match args.method:
        case "lexical":
            results = rank_lexical(args.query, passages, args.limit)
        case "semantic":
            model = SentenceTransformer("intfloat/multilingual-e5-small")
            idx = build_index(passages, model)
            results = rank_semantic(args.query, idx, model, args.limit)
        case "hybrid":
            pool_size = max(7, args.limit)
            lex = rank_lexical(args.query, passages, pool_size)
            model = SentenceTransformer("intfloat/multilingual-e5-small")
            idx = build_index(passages, model)
            sem = rank_semantic(args.query, idx, model, pool_size)
            results = rank_hybrid(lex, sem, args.limit)
        case "linked":
            model = SentenceTransformer("intfloat/multilingual-e5-small")
            idx = build_index(passages, model)
            rs = rank_semantic(args.query, idx, model, 3)
            scores_by_id = {item["id"]: item["score"] for item in rs}
            ordered_ids = expand_linked_evidence(rs, passages, args.limit)
            results = [{"id": pid, "score": scores_by_id.get(pid)} for pid in ordered_ids]
    
    final = []
    if not results:
        no_result = "Nenhum resultado encontrado."
        print(no_result)
        return
    locators_by_id = {p["id"]: p.get("locator") for p in passages}
    urls_by_id = {p["id"]: p.get("url") for p in passages}
    text_by_id = {p["id"]: p.get("text") for p in passages}
    for r in results:
        final.append({
            "id": r["id"],
            "score": r["score"],
            "locator": locators_by_id.get(r["id"]),
            "text": text_by_id.get(r["id"]),
        })

    match args.command:
        case "search":
            for f in final:
                score_display = ("incluída pelo vínculo com o CTB" if f["score"] is None else f["score"]) 
                displayed = {**f, "score": score_display}
                print(f"{displayed}\n")
            return final
        case "ask":
            load_dotenv()
            api_key = os.environ["GEMINI_API_KEY"]
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
            client = genai.Client(api_key=api_key)
            assessment = generate_assessment(args.query, final, client, os.environ["GEMINI_MODEL_NAME"])

            retrieved_ids = [i["id"] for i in final]
            assessment["retrieved_ids"] = retrieved_ids
           
            usage = assessment.get("usage", {})
            tokens_line = (
                f"Tokens -> entrada: {usage.get('input_tokens', 0)} | "
                f"saída: {usage.get('output_tokens', 0)} | "
                f"total: {usage.get('total_tokens', 0)}"
            )
            
            match assessment["outcome"]:
                case "ask_clarification":
                    print(
                        f"Pedir mais detalhes: {assessment["question"]}\n"
                        f"{tokens_line}"
                    )
                    return assessment
                case "insufficient_evidence":
                    print(
                        f"Evidências insuficientes: {assessment["explanation"]}\n"
                        f"{tokens_line}"
                    )
                    return assessment
                case "candidate":
                    ctb_id = assessment["ctb_passage_id"]
                    mbft_id = assessment["mbft_passage_id"]
                    cited_ids = [ctb_id, mbft_id]
                    
                    if not validate_citations(cited_ids, retrieved_ids):
                        raise ValueError("The assessment could not be verified against retrieved evidence.")

                    ctb_locator = locators_by_id.get(ctb_id)
                    mbft_locator = locators_by_id.get(mbft_id)
                    ctb_url = urls_by_id.get(ctb_id)
                    mbft_url = urls_by_id.get(mbft_id)

                    print(
                        f"{assessment['explanation']}\n"
                        f"Fontes: {ctb_locator} ({ctb_id}) ({ctb_url}) | {mbft_locator} ({mbft_id}) ({mbft_url})\n"
                        f"{tokens_line}"
                    )

                    return {
                        **assessment,
                        "ctb_locator": ctb_locator,
                        "ctb_url": ctb_url,
                        "mbft_locator": mbft_locator,
                        "mbft_url": mbft_url     
                    }
