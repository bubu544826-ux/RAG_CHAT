"""Create text embeddings with the configured Hugging Face model."""

from functools import lru_cache
from typing import Any

from .config import EMBEDDING_MODEL_NAME


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot create a vector."""


@lru_cache(maxsize=1)
def _load_model() -> Any:
    """Load the model once and reuse it for later embedding calls."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "缺少 sentence-transformers，请先安装 requirements.txt 中的依赖。"
        ) from exc

    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as exc:
        raise EmbeddingError(
            f"无法加载 embedding 模型 {EMBEDDING_MODEL_NAME!r}。"
        ) from exc


def embed_text(text: str) -> list[float]:
    """Convert one non-empty text string into a normalized embedding vector."""
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串。")

    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("text 不能为空。")

    try:
        model = _load_model()
        embedding = model.encode(
            cleaned_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError("生成文本向量失败。") from exc

    try:
        vector = embedding.tolist()
    except AttributeError as exc:
        raise EmbeddingError("模型返回了无效的 embedding vector。") from exc

    if (
        not isinstance(vector, list)
        or not vector
        or any(isinstance(value, list) for value in vector)
    ):
        raise EmbeddingError("模型返回了无效的 embedding vector。")

    try:
        return [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise EmbeddingError("embedding vector 包含非数值元素。") from exc
