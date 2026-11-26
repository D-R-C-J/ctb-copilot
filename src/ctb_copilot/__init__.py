from .corpus import load_passages
from .cli import run_cli
from .index import build_index
import pathlib
from sentence_transformers import SentenceTransformer

def main() -> None:
    print("Running load_passages on src/ctb_copilot/data/passages.jsonl")
    path = pathlib.Path(__file__).parent / "data" / "passages.jsonl"
    passages = load_passages(path)
    print(f"Loaded {len(passages)} passages successfully.")
    rank = run_cli(passages)
    #model = SentenceTransformer("intfloat/multilingual-e5-small")
    #build_index(passages, model)

