"""Command-line entry point for retrieving chunks from the local index."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.rag_app.config import RETRIEVAL_SETTINGS
from src.rag_app.embedding import EmbeddingError
from src.rag_app.retriever import (
    BASELINE_RETRIEVAL_OPTIONS,
    production_retrieval_options,
    retrieve,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "index.json"


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve the chunks most relevant to a question from the local RAG index."
    )
    parser.add_argument("question", help="the question to retrieve for")
    parser.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL_SETTINGS.final_top_k,
        help=f"number of results to return (default: {RETRIEVAL_SETTINGS.final_top_k})",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Chroma index directory",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="use the old vector-only retrieval: no BM25 fusion, reranking, or neighbour expansion",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run retrieval and print the ranked results as JSON."""
    args = _parse_arguments(arguments)

    try:
        options = (
            dict(BASELINE_RETRIEVAL_OPTIONS)
            if args.baseline
            else production_retrieval_options(args.top_k)
        )
        results = retrieve(args.question, args.index, top_k=args.top_k, **options)
    except (OSError, TypeError, ValueError, EmbeddingError) as exc:
        print(f"Retrieval failed: {exc}")
        return 1

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
