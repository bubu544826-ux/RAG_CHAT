"""Coordinate retrieval, prompt building, and answer generation."""

import json
import logging
import os
import re
import time
from pathlib import Path

from .config import LOG_LEVEL
from .generator import Generator
from .prompt_builder import build_prompt
from .retriever import DEFAULT_TOP_K, retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "index.json"

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL))
if not logger.handlers:
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(log_handler)
logger.propagate = False

API_KEY_PATTERN = re.compile(
    r"(?i)(?:[A-Z0-9_]*API[_ -]?KEY\s*[:=]\s*)\S+|"
    r"\bsk-(?:ant-)?[A-Za-z0-9_-]{8,}\b"
)


def _redact_api_keys(value: object) -> str:
    """Return log-safe text with known and recognizable API keys removed."""
    text = str(value)
    known_api_keys = [
        secret
        for name, secret in os.environ.items()
        if name.upper().endswith("API_KEY") and secret
    ]
    for secret in sorted(known_api_keys, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return API_KEY_PATTERN.sub("[REDACTED]", text)


def _log_request(
    question: object,
    retrieved_chunks: list[dict[str, object]],
    started_at: float,
    error: Exception | None = None,
) -> None:
    """Write one JSON log containing only request-level metadata."""
    log_record = {
        "question": _redact_api_keys(question),
        "retrieved_chunk_ids": [
            chunk.get("chunk_id")
            for chunk in retrieved_chunks
            if isinstance(chunk, dict)
        ],
        "similarity_scores": [
            chunk.get("score")
            for chunk in retrieved_chunks
            if isinstance(chunk, dict)
        ],
        "request_latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "success": error is None,
        "error": None if error is None else _redact_api_keys(error),
    }
    logger.info(json.dumps(log_record, ensure_ascii=False, default=str))


class RAGService:
    """Run the three steps of the RAG question-answering flow."""

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        top_k: int = DEFAULT_TOP_K,
        generator: Generator | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.top_k = top_k
        self.generator = generator if generator is not None else Generator()

    def retrieve(self, question: str) -> list[dict[str, object]]:
        """Retrieve relevant chunks for one question."""
        return retrieve(question, self.index_path, top_k=self.top_k)

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: list[dict[str, object]],
    ) -> str:
        """Build the LLM prompt from the question and retrieved chunks."""
        return build_prompt(question, retrieved_chunks)

    def generate(self, prompt: str) -> str:
        """Generate the final answer from the built prompt."""
        return self.generator.generate(prompt)

    def ask(self, question: str) -> dict[str, object]:
        """Return an answer and its retrieved source information."""
        started_at = time.perf_counter()
        retrieved_chunks: list[dict[str, object]] = []

        try:
            retrieved_chunks = self.retrieve(question)
            prompt = self.build_prompt(question, retrieved_chunks)
            answer = self.generate(prompt)
            sources = [
                {
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "score": chunk["score"],
                }
                for chunk in retrieved_chunks
            ]
        except Exception as exc:
            _log_request(question, retrieved_chunks, started_at, error=exc)
            raise

        _log_request(question, retrieved_chunks, started_at)
        return {"answer": answer, "sources": sources}
