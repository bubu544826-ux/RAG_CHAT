"""Command-line entry point for building the local RAG index."""

from functools import partial
from pathlib import Path

from src.rag_app.embedding import EmbeddingError, embed_texts
from src.rag_app.indexer import build_index


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIRECTORY = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "index.json"


def main() -> int:
    """Build the index and print counts for a quick acceptance check."""
    try:
        summary = build_index(
            INPUT_DIRECTORY,
            OUTPUT_PATH,
            # Indexing a large corpus takes minutes, so show batch progress.
            embedding_function=partial(embed_texts, show_progress=True),
        )
    except (OSError, TypeError, ValueError, EmbeddingError) as exc:
        print(f"Failed to build the index: {exc}")
        return 1

    print(f"files: {summary['file_count']}")
    print(f"chunks: {summary['chunk_count']}")
    print(f"embeddings: {summary['embedding_count']}")
    print(f"Chroma index directory: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
