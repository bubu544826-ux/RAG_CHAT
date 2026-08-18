"""Calculate similarity between embedding vectors."""

from collections.abc import Sequence

import numpy as np


def cosine_similarity(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """Return the cosine similarity between two embedding vectors."""
    embedding_a = np.asarray(vector_a, dtype=float)
    embedding_b = np.asarray(vector_b, dtype=float)

    if embedding_a.ndim != 1 or embedding_b.ndim != 1:
        raise ValueError("Embedding vectors must be one-dimensional.")
    if embedding_a.shape != embedding_b.shape:
        raise ValueError("Both embedding vectors must have the same dimension.")
    if not np.all(np.isfinite(embedding_a)) or not np.all(
        np.isfinite(embedding_b)
    ):
        raise ValueError("Embedding vectors must contain only finite values.")

    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Embedding vectors must not be zero vectors.")

    return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))
