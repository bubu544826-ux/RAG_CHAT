"""Command-line entry point for retrieving chunks from the local index."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.rag_app.embedding import EmbeddingError
from src.rag_app.retriever import DEFAULT_TOP_K, retrieve


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "index.json"


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从本地 RAG index 中检索与问题最相关的 chunks。"
    )
    parser.add_argument("question", help="要检索的问题")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"返回结果数量（默认：{DEFAULT_TOP_K}）",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="index.json 路径",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run retrieval and print the ranked results as JSON."""
    args = _parse_arguments(arguments)

    try:
        results = retrieve(args.question, args.index, top_k=args.top_k)
    except (OSError, TypeError, ValueError, EmbeddingError) as exc:
        print(f"检索失败：{exc}")
        return 1

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
