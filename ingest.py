"""Command-line entry point for building the local RAG index."""

from pathlib import Path

from src.rag_app.embedding import EmbeddingError
from src.rag_app.indexer import build_index


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIRECTORY = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "index.json"


def main() -> int:
    """Build the index and print counts for a quick acceptance check."""
    try:
        summary = build_index(INPUT_DIRECTORY, OUTPUT_PATH)
    except (OSError, TypeError, ValueError, EmbeddingError) as exc:
        print(f"生成 index 失败：{exc}")
        return 1

    print(f"文件数量：{summary['file_count']}")
    print(f"chunk 数量：{summary['chunk_count']}")
    print(f"embedding 数量：{summary['embedding_count']}")
    print(f"index 文件：{OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
