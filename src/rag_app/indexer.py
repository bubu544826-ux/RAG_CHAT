"""Build a small local JSON index from source documents."""

import json
from collections.abc import Callable
from pathlib import Path

from .chunker import chunk_document
from .document_loader import load_documents
from .embedding import embed_text


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def build_index(
    input_directory: str | Path,
    output_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_function: Callable[[str], list[float]] = embed_text,
) -> dict[str, int]:
    """Run loader, chunker and embedding, then write records as JSON."""
    documents = load_documents(input_directory)
    records: list[dict[str, object]] = []

    for document in documents:
        chunks = chunk_document(document, chunk_size=chunk_size, overlap=overlap)

        for chunk in chunks:
            # Empty chunks cannot produce useful embeddings.
            if not chunk["text"].strip():
                continue

            records.append(
                {
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "embedding": embedding_function(chunk["text"]),
                }
            )

    index_path = Path(output_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "file_count": len(documents),
        "chunk_count": len(records),
        "embedding_count": len(records),
    }
