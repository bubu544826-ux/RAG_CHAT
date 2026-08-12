"""Retrieve the most relevant chunks from a local JSON index."""

import json
from collections.abc import Callable
from pathlib import Path

from .embedding import embed_text
from .similarity import cosine_similarity


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

    records = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("index 必须是 JSON 数组。")

    query_embedding = embedding_function(question)
    results: list[dict[str, object]] = []

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("index 中的每条记录必须是 JSON 对象。")

        required_fields = ("text", "source", "chunk_id", "embedding")
        missing_fields = [field for field in required_fields if field not in record]
        if missing_fields:
            raise ValueError(f"index 记录缺少字段：{', '.join(missing_fields)}。")

        score = cosine_similarity(query_embedding, record["embedding"])
        results.append(
            {
                "text": record["text"],
                "source": record["source"],
                "chunk_id": record["chunk_id"],
                "score": score,
            }
        )

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_k]
