from .corpus import load_passages
from .cli import run_cli
import pathlib
def main() -> None:
    print("Running load_passages on src/ctb_copilot/data/cases/dev.jsonl")
    path = pathlib.Path(__file__).parent / "data" / "passages.jsonl"
    passages = load_passages(path)
    print(f"Loaded {len(passages)} passages successfully.")
    rank = run_cli(passages)
    

