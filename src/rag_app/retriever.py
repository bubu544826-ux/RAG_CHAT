"""Retrieve the most relevant chunks from a local Chroma index."""

from collections.abc import Callable
from pathlib import Path

import chromadb

from .embedding import embed_text
from .indexer import COLLECTION_NAME


DEFAULT_TOP_K = 3


def retrieve(
    question: str,
    index_path: str | Path,
    top_k: int = DEFAULT_TOP_K,
    embedding_function: Callable[[str], list[float]] = embed_text,
) -> list[dict[str, object]]:
    """Return the ``top_k`` index chunks most similar to the question."""
    if not isinstance(question, str):
        raise TypeError("question 必须是字符串。")
    if not question.strip():
        raise ValueError("question 不能为空。")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k 必须是整数。")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0。")

    chroma_path = Path(index_path)
    if chroma_path.is_file():
        raise ValueError("检测到旧 JSON index，请先运行 python ingest.py 迁移到 Chroma。")

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )
    result_count = min(top_k, collection.count())
    if result_count == 0:
        return []

    query_embedding = embedding_function(question)
    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )
    documents = query_result["documents"]
    metadatas = query_result["metadatas"]
    distances = query_result["distances"]
    if documents is None or metadatas is None or distances is None:
        raise ValueError("Chroma 查询结果缺少 documents、metadatas 或 distances。")

    results: list[dict[str, object]] = []
    for text, metadata, distance in zip(
        documents[0],
        metadatas[0],
        distances[0],
    ):
        if text is None or metadata is None:
            raise ValueError("Chroma 查询结果包含空文档或空 metadata。")
        source = metadata.get("source")
        chunk_id = metadata.get("chunk_id")
        if not isinstance(source, str) or not isinstance(chunk_id, str):
            raise ValueError("Chroma metadata 缺少 source 或 chunk_id。")

        results.append(
            {
                "text": text,
                "source": source,
                "chunk_id": chunk_id,
                # Chroma cosine distance is 1 - cosine similarity.
                "score": 1.0 - float(distance),
            }
        )

    return results
