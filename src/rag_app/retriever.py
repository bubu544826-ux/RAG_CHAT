"""Retrieve, fuse, and rerank chunks from a local Chroma index."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb

from .config import DEFAULT_RERANKER_MODEL_NAME
from .embedding import embed_text
from .indexer import COLLECTION_NAME


DEFAULT_TOP_K = 3
SUPPORTED_STRATEGIES = {"vector_only", "lexical_only", "hybrid"}
SUPPORTED_REWRITE_MODES = {"single", "multi_query"}

logger = logging.getLogger(__name__)

EmbeddingFunction = Callable[[str], list[float]]
RewriteFunction = Callable[[str, int], list[str]]
RerankFunction = Callable[[str, list[dict[str, object]]], Sequence[float]]

TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]"
)
PRECISE_QUERY_PATTERN = re.compile(
    r"(?:\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b)|"
    r"(?:\b[\w.-]+\.(?:py|js|ts|json|ya?ml|toml|ini|md|txt|log)\b)|"
    r"(?:\b[0-9a-f]{8}-[0-9a-f-]{27,}\b)|(?:https?://\S+)",
    re.IGNORECASE,
)
CONSTRAINT_PATTERNS = (
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
    re.compile(r"\bv?\d+(?:\.\d+){0,3}\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\^\d+)?(?:-[A-Za-z][A-Za-z0-9]*)?\b"),
    re.compile(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b"),
    re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b"),
    re.compile(r"\b[\w.-]+\.(?:py|js|ts|json|ya?ml|toml|ini|md|txt|log)\b"),
)
NEGATIONS = {"no", "not", "never", "without", "except", "cannot", "can't", "不", "没有", "不能"}
QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "be", "because", "can", "do", "does", "for",
    "from", "how", "if", "in", "is", "it", "of", "on", "or", "should", "that",
    "the", "this", "to", "what", "when", "where", "which", "why", "with", "would",
}


def _validate_question_and_top_k(question: str, top_k: int) -> str:
    if not isinstance(question, str):
        raise TypeError("question 必须是字符串。")
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("question 不能为空。")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k 必须是整数。")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0。")
    return cleaned_question


def _open_collection(index_path: str | Path) -> Any:
    chroma_path = Path(index_path)
    if chroma_path.is_file():
        raise ValueError("检测到旧 JSON index，请先运行 python ingest.py 迁移到 Chroma。")
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_collection(name=COLLECTION_NAME, embedding_function=None)


def _result(
    text: str,
    metadata: dict[str, Any],
    score: float,
    source_name: str,
    rank: int,
) -> dict[str, object]:
    source = metadata.get("source")
    chunk_id = metadata.get("chunk_id")
    if not isinstance(source, str) or not isinstance(chunk_id, str):
        raise ValueError("Chroma metadata 缺少 source 或 chunk_id。")
    document_id = metadata.get("document_id", source)
    return {
        "text": text,
        "source": source,
        "document_id": str(document_id),
        "chunk_id": chunk_id,
        "score": float(score),
        "retrieval_score": float(score),
        "original_rank": rank,
        "final_rank": rank,
        "retrieval_source": [source_name],
    }


def vector_search(
    question: str,
    index_path: str | Path,
    top_k: int,
    embedding_function: EmbeddingFunction = embed_text,
) -> list[dict[str, object]]:
    """Run semantic search against Chroma and retain rank metadata."""
    cleaned_question = _validate_question_and_top_k(question, top_k)
    collection = _open_collection(index_path)
    result_count = min(top_k, collection.count())
    if result_count == 0:
        return []

    query_result = collection.query(
        query_embeddings=[embedding_function(cleaned_question)],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )
    documents = query_result["documents"]
    metadatas = query_result["metadatas"]
    distances = query_result["distances"]
    if documents is None or metadatas is None or distances is None:
        raise ValueError("Chroma 查询结果缺少 documents、metadatas 或 distances。")

    results: list[dict[str, object]] = []
    for rank, (text, metadata, distance) in enumerate(
        zip(documents[0], metadatas[0], distances[0]), start=1
    ):
        if text is None or metadata is None:
            raise ValueError("Chroma 查询结果包含空文档或空 metadata。")
        results.append(_result(text, metadata, 1.0 - float(distance), "vector", rank))
    return results


def tokenize(text: str) -> list[str]:
    """Tokenize prose and identifiers for the small local BM25 implementation."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _bm25_scores(query: str, documents: list[str]) -> list[float]:
    """Calculate BM25 scores without hiding the ranking formula in a framework."""
    tokenized_documents = [tokenize(document) for document in documents]
    query_terms = list(dict.fromkeys(tokenize(query)))
    if not query_terms or not documents:
        return [0.0] * len(documents)

    document_count = len(documents)
    average_length = sum(map(len, tokenized_documents)) / document_count or 1.0
    document_frequency = Counter(
        term for tokens in tokenized_documents for term in set(tokens)
    )
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for tokens in tokenized_documents:
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            frequency_in_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0
                + (document_count - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * len(tokens) / average_length
            )
            score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
        scores.append(score)
    return scores


def lexical_search(
    question: str,
    index_path: str | Path,
    top_k: int,
) -> list[dict[str, object]]:
    """Rank all local chunks with BM25 keyword search."""
    cleaned_question = _validate_question_and_top_k(question, top_k)
    collection = _open_collection(index_path)
    stored = collection.get(include=["documents", "metadatas"])
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    if len(documents) != len(metadatas):
        raise ValueError("Chroma 查询结果中的 documents 与 metadatas 数量不一致。")

    ranked = sorted(
        enumerate(_bm25_scores(cleaned_question, documents)),
        key=lambda item: (-item[1], str(stored["ids"][item[0]])),
    )
    results: list[dict[str, object]] = []
    for index, score in ranked:
        if score <= 0 or len(results) >= top_k:
            break
        text = documents[index]
        metadata = metadatas[index]
        if text is None or metadata is None:
            raise ValueError("Chroma 查询结果包含空文档或空 metadata。")
        results.append(_result(text, metadata, score, "lexical", len(results) + 1))
    return results


def reciprocal_rank_fusion(
    rankings: Sequence[tuple[str, list[dict[str, object]]]],
    rrf_k: int = 60,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Fuse ranked lists using RRF and deduplicate chunks by ``chunk_id``."""
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k <= 0:
        raise ValueError("rrf_k 必须是正整数。")
    fused_scores: dict[str, float] = {}
    first_result: dict[str, dict[str, object]] = {}
    ranks_by_chunk: dict[str, list[int]] = {}
    sources_by_chunk: dict[str, set[str]] = {}

    for source_name, results in rankings:
        seen_in_ranking: set[str] = set()
        for rank, item in enumerate(results, start=1):
            chunk_id = item.get("chunk_id")
            if not isinstance(chunk_id, str) or chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            first_result.setdefault(chunk_id, dict(item))
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
                rrf_k + rank
            )
            ranks_by_chunk.setdefault(chunk_id, []).append(rank)
            sources_by_chunk.setdefault(chunk_id, set()).add(source_name)

    ordered_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            -fused_scores[chunk_id],
            min(ranks_by_chunk[chunk_id]),
            chunk_id,
        ),
    )
    if limit is not None:
        ordered_ids = ordered_ids[:limit]

    fused: list[dict[str, object]] = []
    for final_rank, chunk_id in enumerate(ordered_ids, start=1):
        item = first_result[chunk_id]
        item.update(
            {
                "score": fused_scores[chunk_id],
                "retrieval_score": fused_scores[chunk_id],
                "original_rank": min(ranks_by_chunk[chunk_id]),
                "final_rank": final_rank,
                "retrieval_source": sorted(sources_by_chunk[chunk_id]),
            }
        )
        fused.append(item)
    return fused


def _ranked_fallback(
    results: list[dict[str, object]], source_name: str, limit: int
) -> list[dict[str, object]]:
    fallback: list[dict[str, object]] = []
    for rank, original in enumerate(results[:limit], start=1):
        item = dict(original)
        item["original_rank"] = rank
        item["final_rank"] = rank
        item["retrieval_source"] = [source_name]
        fallback.append(item)
    return fallback


def retrieve_candidates(
    question: str,
    index_path: str | Path,
    *,
    strategy: str,
    vector_top_k: int,
    lexical_top_k: int,
    candidate_k: int,
    rrf_k: int,
    embedding_function: EmbeddingFunction,
) -> list[dict[str, object]]:
    """Run one retrieval query with backend-level graceful fallbacks."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"不支持的 retrieval strategy：{strategy}")
    if strategy == "vector_only":
        return vector_search(question, index_path, vector_top_k, embedding_function)[
            :candidate_k
        ]
    if strategy == "lexical_only":
        return lexical_search(question, index_path, lexical_top_k)[:candidate_k]

    vector_results: list[dict[str, object]] | None = None
    lexical_results: list[dict[str, object]] | None = None
    vector_error: Exception | None = None
    lexical_error: Exception | None = None
    try:
        vector_results = vector_search(
            question, index_path, vector_top_k, embedding_function
        )
    except Exception as exc:  # Optional backend: lexical search can still answer.
        vector_error = exc
        logger.warning("Vector search failed; using lexical fallback: %s", exc)
    try:
        lexical_results = lexical_search(question, index_path, lexical_top_k)
    except Exception as exc:  # Optional backend: vector search can still answer.
        lexical_error = exc
        logger.warning("BM25 search failed; using vector fallback: %s", exc)

    if vector_results is not None and lexical_results is not None:
        return reciprocal_rank_fusion(
            [("vector", vector_results), ("lexical", lexical_results)],
            rrf_k=rrf_k,
            limit=candidate_k,
        )
    if vector_results is not None:
        return _ranked_fallback(vector_results, "vector", candidate_k)
    if lexical_results is not None:
        return _ranked_fallback(lexical_results, "lexical", candidate_k)
    raise RuntimeError(
        f"向量检索和 BM25 检索均失败：vector={vector_error}; lexical={lexical_error}"
    )


def is_precise_query(question: str) -> bool:
    """Return whether rewriting could damage a code-, file-, or ID-like query."""
    return PRECISE_QUERY_PATTERN.search(question) is not None


def extract_query_constraints(question: str) -> set[str]:
    """Extract values that every accepted rewrite must preserve verbatim."""
    constraints: set[str] = set()
    for pattern in CONSTRAINT_PATTERNS:
        constraints.update(pattern.findall(question))
    for proper_noun in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question):
        if proper_noun.lower() not in QUERY_STOP_WORDS:
            constraints.add(proper_noun)
    question_lower = question.lower()
    constraints.update(
        negation
        for negation in NEGATIONS
        if re.search(rf"(?<!\w){re.escape(negation)}(?!\w)", question_lower)
    )
    return constraints


def preserves_query_constraints(original: str, rewritten: str) -> bool:
    """Reject a rewrite when it drops dates, IDs, numbers, products, or negation."""
    rewritten_lower = rewritten.lower()
    return all(value.lower() in rewritten_lower for value in extract_query_constraints(original))


def rule_based_rewrite(question: str, max_queries: int) -> list[str]:
    """Create transparent keyword-oriented query variants with no LLM dependency."""
    tokens = TOKEN_PATTERN.findall(question)
    keywords = [token for token in tokens if token.lower() not in QUERY_STOP_WORDS]
    keyword_query = " ".join(keywords).strip()
    candidates = [keyword_query, f"{keyword_query} specification requirements".strip()]
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate.lower() != question.strip().lower() and candidate not in unique:
            unique.append(candidate)
    return unique[:max_queries]


def generate_search_queries(
    question: str,
    *,
    enabled: bool,
    mode: str,
    max_queries: int,
    rewrite_function: RewriteFunction | None = None,
) -> list[str]:
    """Return safe search queries, falling back to the original on any failure."""
    cleaned_question = _validate_question_and_top_k(question, 1)
    if not enabled or is_precise_query(cleaned_question):
        return [cleaned_question]
    if mode not in SUPPORTED_REWRITE_MODES:
        raise ValueError(f"不支持的 query rewrite mode：{mode}")
    if not isinstance(max_queries, int) or isinstance(max_queries, bool) or max_queries <= 0:
        raise ValueError("max_queries 必须是正整数。")

    rewriter = rewrite_function or rule_based_rewrite
    try:
        rewritten_queries = rewriter(cleaned_question, max_queries)
        if not isinstance(rewritten_queries, list):
            raise TypeError("Query rewriter 必须返回字符串列表。")
    except Exception as exc:
        logger.warning("Query rewrite failed; using original query: %s", exc)
        return [cleaned_question]

    safe_rewrites: list[str] = []
    for rewritten in rewritten_queries:
        if not isinstance(rewritten, str):
            continue
        cleaned_rewrite = rewritten.strip()
        if (
            cleaned_rewrite
            and cleaned_rewrite.lower() != cleaned_question.lower()
            and preserves_query_constraints(cleaned_question, cleaned_rewrite)
            and cleaned_rewrite not in safe_rewrites
        ):
            safe_rewrites.append(cleaned_rewrite)

    if mode == "single":
        return [safe_rewrites[0]] if safe_rewrites else [cleaned_question]
    return ([cleaned_question] + safe_rewrites)[:max_queries]


def merge_multi_query_results(
    query_results: Sequence[tuple[str, list[dict[str, object]]]],
    *,
    rrf_k: int,
    limit: int,
) -> list[dict[str, object]]:
    """RRF-merge query variants without evicting original-query candidates.

    Rewrites are optional recall aids, so a fixed-size pool must not lose a
    candidate merely because several similar rewrites agree on another chunk.
    Rewrites fill unused slots and can reorder the pool, while the original
    query's candidates are retained as the safe baseline.
    """
    fused = reciprocal_rank_fusion(query_results, rrf_k=rrf_k)
    selected = fused[:limit]
    original_results = query_results[0][1][:limit] if query_results else []
    original_ids = {str(item["chunk_id"]) for item in original_results}
    selected_ids = {str(item["chunk_id"]) for item in selected}
    for original in original_results:
        chunk_id = str(original["chunk_id"])
        if chunk_id in selected_ids:
            continue
        replace_index = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if str(selected[index]["chunk_id"]) not in original_ids
            ),
            None,
        )
        if replace_index is None:
            if len(selected) < limit:
                selected.append(dict(original))
            else:
                continue
        else:
            selected_ids.discard(str(selected[replace_index]["chunk_id"]))
            selected[replace_index] = dict(original)
        selected_ids.add(chunk_id)
    fused = selected
    results_by_id = {
        str(item["chunk_id"]): item
        for _query, results in query_results
        for item in results
    }
    queries_by_id: dict[str, list[str]] = {}
    sources_by_id: dict[str, set[str]] = {}
    for query, results in query_results:
        for item in results:
            chunk_id = str(item["chunk_id"])
            queries_by_id.setdefault(chunk_id, []).append(query)
            raw_sources = item.get("retrieval_source", [])
            if isinstance(raw_sources, list):
                sources_by_id.setdefault(chunk_id, set()).update(map(str, raw_sources))
    for item in fused:
        chunk_id = str(item["chunk_id"])
        item["retrieval_source"] = sorted(sources_by_id.get(chunk_id, set()))
        item["search_queries"] = list(dict.fromkeys(queries_by_id.get(chunk_id, [])))
        # Keep fields from a backend result if the generic RRF copy omitted one.
        item.setdefault("document_id", results_by_id[chunk_id].get("document_id"))
    for final_rank, item in enumerate(fused, start=1):
        item["final_rank"] = final_rank
    return fused


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def cross_encoder_scores(
    question: str,
    candidates: list[dict[str, object]],
    model_name: str = DEFAULT_RERANKER_MODEL_NAME,
) -> Sequence[float]:
    """Score query/chunk pairs with a compact sentence-transformers reranker."""
    model = _load_cross_encoder(model_name)
    pairs = [(question, str(candidate["text"])) for candidate in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    return scores.tolist() if hasattr(scores, "tolist") else list(scores)


def rerank_candidates(
    question: str,
    candidates: list[dict[str, object]],
    *,
    rerank_function: RerankFunction,
    final_k: int,
) -> list[dict[str, object]]:
    """Apply second-stage scores with deterministic stable tie handling."""
    if not candidates:
        return []
    scores = list(rerank_function(question, candidates))
    if len(scores) != len(candidates):
        raise ValueError("Reranker score 数量与候选数量不一致。")
    scored: list[tuple[int, float, dict[str, object]]] = []
    for index, (candidate, raw_score) in enumerate(zip(candidates, scores)):
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("Reranker 返回了非有限分数。")
        scored.append((index, score, candidate))
    scored.sort(key=lambda item: (-item[1], item[0]))

    reranked: list[dict[str, object]] = []
    for final_rank, (_index, rerank_score, original) in enumerate(
        scored[:final_k], start=1
    ):
        item = dict(original)
        item["rerank_score"] = rerank_score
        item["score"] = rerank_score
        item["final_rank"] = final_rank
        reranked.append(item)
    return reranked


def retrieve(
    question: str,
    index_path: str | Path,
    top_k: int = DEFAULT_TOP_K,
    embedding_function: EmbeddingFunction = embed_text,
    *,
    strategy: str = "vector_only",
    vector_top_k: int | None = None,
    lexical_top_k: int = 30,
    candidate_k: int | None = None,
    rrf_k: int = 60,
    reranker_enabled: bool = False,
    reranker_model_name: str = DEFAULT_RERANKER_MODEL_NAME,
    query_rewrite_enabled: bool = False,
    query_rewrite_mode: str = "multi_query",
    max_queries: int = 3,
    rewrite_function: RewriteFunction | None = None,
    rerank_function: RerankFunction | None = None,
) -> list[dict[str, object]]:
    """Run configurable vector/BM25 retrieval, rewriting, fusion, and reranking.

    Keyword-only arguments default to the original vector-only behavior so older
    direct callers remain compatible. ``RAGService`` supplies the application
    settings for the full pipeline.
    """
    cleaned_question = _validate_question_and_top_k(question, top_k)
    vector_k = vector_top_k if vector_top_k is not None else top_k
    pool_size = candidate_k if candidate_k is not None else max(top_k, vector_k, lexical_top_k)
    for name, value in {
        "vector_top_k": vector_k,
        "lexical_top_k": lexical_top_k,
        "candidate_k": pool_size,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} 必须是正整数。")
    if pool_size < top_k:
        raise ValueError("candidate_k 不能小于最终 top_k。")

    search_queries = generate_search_queries(
        cleaned_question,
        enabled=query_rewrite_enabled,
        mode=query_rewrite_mode,
        max_queries=max_queries,
        rewrite_function=rewrite_function,
    )
    query_results: list[tuple[str, list[dict[str, object]]]] = []
    retrieval_errors: list[Exception] = []
    for search_query in search_queries:
        try:
            candidates = retrieve_candidates(
                search_query,
                index_path,
                strategy=strategy,
                vector_top_k=vector_k,
                lexical_top_k=lexical_top_k,
                candidate_k=pool_size,
                rrf_k=rrf_k,
                embedding_function=embedding_function,
            )
            query_results.append((search_query, candidates))
        except Exception as exc:
            retrieval_errors.append(exc)
            logger.warning("Retrieval failed for query variant %r: %s", search_query, exc)

    if not query_results:
        if retrieval_errors:
            raise retrieval_errors[0]
        return []
    if len(query_results) == 1:
        candidates = query_results[0][1][:pool_size]
    else:
        candidates = merge_multi_query_results(
            query_results, rrf_k=rrf_k, limit=pool_size
        )

    if not reranker_enabled:
        return candidates[:top_k]
    scorer = rerank_function or (
        lambda query, items: cross_encoder_scores(
            query, items, model_name=reranker_model_name
        )
    )
    try:
        return rerank_candidates(
            cleaned_question, candidates, rerank_function=scorer, final_k=top_k
        )
    except Exception as exc:
        logger.warning("Reranker failed; using fused retrieval ranking: %s", exc)
        return _ranked_fallback(candidates, "fused", top_k)
