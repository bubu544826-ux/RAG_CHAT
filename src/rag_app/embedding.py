"""Create text embeddings with the configured Hugging Face model."""

from functools import lru_cache
from typing import Any

from .config import EMBEDDING_MODEL_NAME


DEFAULT_BATCH_SIZE = 32


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

    return _coerce_vector(vector)


def _coerce_vector(vector: Any) -> list[float]:
    """Validate one model output row and convert it to a plain float list."""
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


def embed_texts(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = False,
) -> list[list[float]]:
    """Embed many texts in batches.

    One `model.encode` call per batch is much faster than one call per text,
    because the model runs a single forward pass over the whole batch instead
    of paying tokenizer and dispatch overhead for every chunk.
    """
    if not isinstance(texts, list):
        raise TypeError("texts 必须是字符串列表。")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")

    cleaned_texts = []
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("texts 必须是字符串列表。")

        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("text 不能为空。")

        cleaned_texts.append(cleaned_text)

    if not cleaned_texts:
        return []

    try:
        model = _load_model()
        embeddings = model.encode(
            cleaned_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError("生成文本向量失败。") from exc

    try:
        vectors = embeddings.tolist()
    except AttributeError as exc:
        raise EmbeddingError("模型返回了无效的 embedding vector。") from exc

    if not isinstance(vectors, list) or len(vectors) != len(cleaned_texts):
        raise EmbeddingError("模型返回的 embedding 数量与输入不一致。")

    return [_coerce_vector(vector) for vector in vectors]
