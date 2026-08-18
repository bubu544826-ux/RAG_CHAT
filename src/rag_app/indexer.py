"""Build a small local Chroma index from source documents."""

from collections.abc import Callable
from pathlib import Path

import chromadb

from .chunker import chunk_document
from .document_loader import load_documents
from .embedding import embed_texts


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
COLLECTION_NAME = "rag_chunks"


def _prepare_chroma_directory(index_path: Path) -> None:
    """Replace a legacy JSON index with a recoverable backup directory layout."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.is_file():
        backup_path = index_path.with_name(f"{index_path.name}.legacy")
        backup_number = 1
        while backup_path.exists():
            backup_path = index_path.with_name(
                f"{index_path.name}.legacy.{backup_number}"
            )
            backup_number += 1
        index_path.replace(backup_path)

    index_path.mkdir(parents=True, exist_ok=True)


def build_index(
    input_directory: str | Path,
    output_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_function: Callable[[list[str]], list[list[float]]] = embed_texts,
) -> dict[str, int]:
    """Run loader, chunker and embedding, then persist records in Chroma."""
    documents = load_documents(input_directory)
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, str]] = []

    for document in documents:
        chunks = chunk_document(document, chunk_size=chunk_size, overlap=overlap)

        for chunk in chunks:
            # Empty chunks cannot produce useful embeddings.
            if not chunk["text"].strip():
                continue

            ids.append(chunk["chunk_id"])
            texts.append(chunk["text"])
            metadatas.append(
                {"source": chunk["source"], "chunk_id": chunk["chunk_id"]}
            )

    # Embedding every chunk in one batched call keeps the model busy instead of
    # paying per-chunk overhead once for each of them.
    embeddings: list[list[float]] = []
    if texts:
        embeddings = embedding_function(texts)
        if len(embeddings) != len(texts):
            raise ValueError("The number of embeddings does not match the number of chunks.")

    index_path = Path(output_path)
    _prepare_chroma_directory(index_path)
    client = chromadb.PersistentClient(path=str(index_path))

    collection_names = {collection.name for collection in client.list_collections()}
    if COLLECTION_NAME in collection_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    if ids:
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    return {
        "file_count": len(documents),
        "chunk_count": len(ids),
        "embedding_count": len(embeddings),
    }
