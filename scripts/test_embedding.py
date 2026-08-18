"""Interactively inspect an embedding without printing the full vector."""

from src.rag_app.embedding import EmbeddingError, embed_text


def main() -> int:
    text = input("Enter a sentence: ")

    try:
        vector = embed_text(text)
    except (TypeError, ValueError, EmbeddingError) as exc:
        print(f"Failed to generate the embedding: {exc}")
        return 1

    print(f"vector type: {type(vector).__name__}")
    print(f"vector dimension: {len(vector)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
