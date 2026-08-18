"""Application configuration values."""

import os
from dataclasses import dataclass


DEFAULT_EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_ANTHROPIC_MODEL_NAME = "claude-sonnet-5"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

EMBEDDING_MODEL_NAME = (
    os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME).strip()
    or DEFAULT_EMBEDDING_MODEL_NAME
)

ANTHROPIC_MODEL_NAME = (
    os.getenv("ANTHROPIC_MODEL_NAME", DEFAULT_ANTHROPIC_MODEL_NAME).strip()
    or DEFAULT_ANTHROPIC_MODEL_NAME
)

LOG_LEVEL = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).strip().upper()
if LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    LOG_LEVEL = DEFAULT_LOG_LEVEL


def _environment_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with an explicit safe default."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _environment_positive_int(name: str, default: int) -> int:
    """Read a positive integer, falling back when configuration is invalid."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class RetrievalSettings:
    """Tunable settings for the two-stage retrieval pipeline."""

    strategy: str = "hybrid"
    vector_top_k: int = 30
    lexical_top_k: int = 30
    rrf_k: int = 60
    reranker_enabled: bool = True
    reranker_model_name: str = DEFAULT_RERANKER_MODEL_NAME
    candidate_k: int = 30
    final_top_k: int = 5
    neighbour_expansion_enabled: bool = True
    neighbour_radius: int = 1
    query_rewrite_enabled: bool = False
    query_rewrite_mode: str = "multi_query"
    max_queries: int = 3


RETRIEVAL_SETTINGS = RetrievalSettings(
    strategy=os.getenv("RETRIEVAL_STRATEGY", "hybrid").strip().lower(),
    vector_top_k=_environment_positive_int("VECTOR_TOP_K", 30),
    lexical_top_k=_environment_positive_int("LEXICAL_TOP_K", 30),
    rrf_k=_environment_positive_int("RRF_K", 60),
    reranker_enabled=_environment_bool("RERANKER_ENABLED", True),
    reranker_model_name=(
        os.getenv("RERANKER_MODEL_NAME", DEFAULT_RERANKER_MODEL_NAME).strip()
        or DEFAULT_RERANKER_MODEL_NAME
    ),
    candidate_k=_environment_positive_int("RETRIEVAL_CANDIDATE_K", 30),
    final_top_k=_environment_positive_int("FINAL_TOP_K", 5),
    neighbour_expansion_enabled=_environment_bool("NEIGHBOUR_EXPANSION_ENABLED", True),
    neighbour_radius=_environment_positive_int("NEIGHBOUR_RADIUS", 1),
    query_rewrite_enabled=_environment_bool("QUERY_REWRITE_ENABLED", False),
    query_rewrite_mode=(
        os.getenv("QUERY_REWRITE_MODE", "multi_query").strip().lower()
    ),
    max_queries=_environment_positive_int("MAX_QUERIES", 3),
)
