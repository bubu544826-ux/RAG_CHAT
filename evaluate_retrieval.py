"""Evaluate ranked chunk retrieval against explicit relevance labels."""

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from src.rag_app.embedding import EmbeddingError
from src.rag_app.retriever import (
    BASELINE_RETRIEVAL_OPTIONS,
    production_retrieval_options,
    retrieve,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "index.json"
DEFAULT_TEST_SET_PATH = PROJECT_ROOT / "RAG_test.json"
# Keep the old name available for callers that imported it.
DEFAULT_QUESTIONS_PATH = DEFAULT_TEST_SET_PATH
DEFAULT_REPORT_PATH = PROJECT_ROOT / "retrieval_evaluation_report.json"
EVALUATION_CUTOFFS = (1, 3, 5, 10)


def _validate_evaluation_cases(cases: object) -> list[dict[str, object]]:
    """Validate the fields needed for label-based retrieval evaluation."""
    if not isinstance(cases, list) or not cases:
        raise ValueError("The evaluation test set must be a non-empty JSON array.")

    seen_case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every evaluation case must be a JSON object.")

        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Every evaluation case must contain a non-empty id.")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        seen_case_ids.add(case_id)

        question = case.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Evaluation case {case_id} must contain a non-empty question.")

        relevant_chunk_ids = case.get("relevant_chunk_ids")
        if not isinstance(relevant_chunk_ids, list) or not relevant_chunk_ids:
            raise ValueError(
                f"Evaluation case {case_id} must contain a non-empty relevant_chunk_ids list."
            )
        if any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in relevant_chunk_ids
        ):
            raise ValueError(
                f"relevant_chunk_ids of evaluation case {case_id} must be non-empty strings."
            )
        if len(relevant_chunk_ids) != len(set(relevant_chunk_ids)):
            raise ValueError(
                f"relevant_chunk_ids of evaluation case {case_id} must not repeat."
            )

    return cases


def load_evaluation_cases(path: str | Path) -> list[dict[str, object]]:
    """Load and validate a chunk-labeled evaluation test set."""
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    return _validate_evaluation_cases(cases)


def load_evaluation_questions(path: str | Path) -> list[dict[str, object]]:
    """Backward-compatible alias for :func:`load_evaluation_cases`."""
    return load_evaluation_cases(path)


def _retrieved_chunk_ids(results: object) -> list[str]:
    """Extract unique chunk IDs from one ranked retrieval result list."""
    if not isinstance(results, list):
        raise ValueError("The retriever must return a list of results.")

    chunk_ids: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Every retriever result must be an object.")
        chunk_id = result.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Every retriever result must contain a non-empty chunk_id.")
        if chunk_id in chunk_ids:
            raise ValueError(f"The retriever returned a duplicate chunk_id: {chunk_id}")
        chunk_ids.append(chunk_id)

    return chunk_ids


def precision_ceiling(cases: list[dict[str, object]], top_k: int) -> float:
    """Highest Precision@k these labels allow, as a percentage.

    ``Precision@k`` divides by the raw cutoff, so a case with one relevant
    chunk caps at ``1/k``. Printing the ceiling keeps the metric readable.
    """
    total = sum(
        min(len(case["relevant_chunk_ids"]), top_k) for case in cases  # type: ignore[arg-type]
    )
    return total / (len(cases) * top_k) * 100


def evaluate_retrieval(
    cases: list[dict[str, object]],
    index_path: str | Path,
    top_k: int = 5,
    retrieve_function: Callable[..., list[dict[str, object]]] = retrieve,
    retrieval_options: dict[str, object] | None = None,
) -> dict[str, float]:
    """Macro-average Recall, Precision, MRR, and binary NDCG at ``top_k``."""
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    _validate_evaluation_cases(cases)
    options = (
        production_retrieval_options(top_k)
        if retrieval_options is None
        else dict(retrieval_options)
    )

    recall_total = 0.0
    precision_total = 0.0
    reciprocal_rank_total = 0.0
    ndcg_total = 0.0

    for case in cases:
        question = case["question"]
        relevant_ids = set(case["relevant_chunk_ids"])
        results = retrieve_function(question, index_path, top_k=top_k, **options)
        ranked_ids = _retrieved_chunk_ids(results)[:top_k]

        relevant_ranks = [
            rank
            for rank, chunk_id in enumerate(ranked_ids, start=1)
            if chunk_id in relevant_ids
        ]
        hit_count = len(relevant_ranks)

        recall_total += hit_count / len(relevant_ids)
        precision_total += hit_count / top_k
        if relevant_ranks:
            reciprocal_rank_total += 1.0 / relevant_ranks[0]

        discounted_gain = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
        ideal_hit_count = min(len(relevant_ids), top_k)
        ideal_gain = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_hit_count + 1)
        )
        ndcg_total += discounted_gain / ideal_gain

    case_count = len(cases)
    return {
        f"Recall@{top_k}": recall_total / case_count * 100,
        f"Precision@{top_k}": precision_total / case_count * 100,
        f"MRR@{top_k}": reciprocal_rank_total / case_count * 100,
        f"NDCG@{top_k}": ndcg_total / case_count * 100,
    }


def calculate_metrics_at_cutoffs(
    cases: list[dict[str, object]],
    rankings: list[list[str]],
    cutoffs: Sequence[int] = EVALUATION_CUTOFFS,
) -> dict[str, float]:
    """Calculate all requested metrics from one shared set of Top-10 rankings."""
    _validate_evaluation_cases(cases)
    if len(rankings) != len(cases):
        raise ValueError("The number of rankings must match the number of evaluation cases.")

    metrics: dict[str, float] = {}
    for top_k in cutoffs:
        if top_k <= 0:
            raise ValueError("The evaluation cutoff must be greater than 0.")
        recall_total = precision_total = reciprocal_rank_total = ndcg_total = 0.0
        for case, ranked_ids in zip(cases, rankings):
            relevant_ids = set(case["relevant_chunk_ids"])
            relevant_ranks = [
                rank
                for rank, chunk_id in enumerate(ranked_ids[:top_k], start=1)
                if chunk_id in relevant_ids
            ]
            recall_total += len(relevant_ranks) / len(relevant_ids)
            precision_total += len(relevant_ranks) / top_k
            if relevant_ranks:
                reciprocal_rank_total += 1.0 / relevant_ranks[0]
            discounted_gain = sum(
                1.0 / math.log2(rank + 1) for rank in relevant_ranks
            )
            ideal_gain = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, min(len(relevant_ids), top_k) + 1)
            )
            ndcg_total += discounted_gain / ideal_gain

        case_count = len(cases)
        metrics.update(
            {
                f"Recall@{top_k}": recall_total / case_count * 100,
                f"Precision@{top_k}": precision_total / case_count * 100,
                f"MRR@{top_k}": reciprocal_rank_total / case_count * 100,
                f"NDCG@{top_k}": ndcg_total / case_count * 100,
            }
        )
    return metrics


RERANKER_MODELS_TO_COMPARE = (
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L12-v2",
    "BAAI/bge-reranker-base",
)


def evaluate_pipeline_comparison(
    cases: list[dict[str, object]],
    index_path: str | Path,
    reranker_models: Sequence[str] = (),
) -> dict[str, dict[str, object]]:
    """Run vector, hybrid, reranked, expanded, and rewritten experiments."""
    fusion = {
        "strategy": "hybrid",
        "vector_top_k": 30,
        "lexical_top_k": 30,
        "candidate_k": 30,
        "rrf_k": 60,
    }
    experiments: dict[str, dict[str, object]] = {
        "vector_baseline": {
            "strategy": "vector_only",
            "vector_top_k": 30,
            "candidate_k": 30,
        },
        "hybrid": dict(fusion),
        "hybrid_reranker": {**fusion, "reranker_enabled": True},
        "hybrid_reranker_neighbours": {
            **fusion,
            "reranker_enabled": True,
            "neighbour_expansion": True,
            "neighbour_radius": 1,
        },
        "hybrid_reranker_neighbours_r2": {
            **fusion,
            "reranker_enabled": True,
            "neighbour_expansion": True,
            "neighbour_radius": 2,
        },
        "full_pipeline": {
            **fusion,
            "reranker_enabled": True,
            "neighbour_expansion": True,
            "neighbour_radius": 1,
            "query_rewrite_enabled": True,
            "query_rewrite_mode": "multi_query",
            "max_queries": 3,
        },
    }
    for model_name in reranker_models:
        label = model_name.rsplit("/", 1)[-1]
        experiments[f"reranker_{label}"] = {
            **fusion,
            "reranker_enabled": True,
            "reranker_model_name": model_name,
            "neighbour_expansion": True,
            "neighbour_radius": 1,
        }
    report: dict[str, dict[str, object]] = {}
    for experiment_name, options in experiments.items():
        rankings: list[list[str]] = []
        latencies: list[float] = []
        for case in cases:
            started_at = time.perf_counter()
            results = retrieve(
                str(case["question"]), index_path, top_k=max(EVALUATION_CUTOFFS), **options
            )
            latencies.append((time.perf_counter() - started_at) * 1000)
            rankings.append(_retrieved_chunk_ids(results))
        sorted_latencies = sorted(latencies)
        report[experiment_name] = {
            "metrics": calculate_metrics_at_cutoffs(cases, rankings),
            "latency_ms": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p95": sorted_latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)],
            },
            "configuration": options,
        }
    return report


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the retriever against explicit relevant-chunk labels.")
    parser.add_argument(
        "--test-set",
        "--questions",
        dest="test_set",
        type=Path,
        default=DEFAULT_TEST_SET_PATH,
        help="path to the evaluation JSON file labeled with relevant_chunk_ids",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Chroma index directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="number of ranked results to evaluate (default: 5)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="evaluate only the old vector-only baseline, for comparison with historical numbers",
    )
    parser.add_argument(
        "--reranker-model",
        dest="reranker_model",
        default=None,
        help="override the CrossEncoder reranker model name",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run the vector/hybrid/reranker/neighbour/rewrite comparison experiments",
    )
    parser.add_argument(
        "--compare-rerankers",
        action="store_true",
        help="also evaluate several reranker models in the comparison (the first run downloads them)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="path of the comparison experiment JSON report",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run ranked retrieval evaluation and print percentage metrics."""
    args = _parse_arguments(arguments)

    try:
        cases = load_evaluation_cases(args.test_set)
        if args.compare:
            report = evaluate_pipeline_comparison(
                cases,
                args.index,
                reranker_models=(
                    RERANKER_MODELS_TO_COMPARE if args.compare_rerankers else ()
                ),
            )
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"questions: {len(cases)}")
            for pipeline_name, result in report.items():
                metrics = result["metrics"]
                latency = result["latency_ms"]
                print(
                    f"{pipeline_name}: Recall@5={metrics['Recall@5']:.2f}% "
                    f"Recall@10={metrics['Recall@10']:.2f}% "
                    f"Precision@5={metrics['Precision@5']:.2f}% "
                    f"MRR@10={metrics['MRR@10']:.2f}% "
                    f"NDCG@5={metrics['NDCG@5']:.2f}% "
                    f"median_latency={latency['median']:.2f}ms"
                )
            print(f"report: {args.output}")
            return 0

        if args.baseline:
            options = dict(BASELINE_RETRIEVAL_OPTIONS)
            pipeline_label = "vector-only baseline"
        else:
            options = production_retrieval_options(args.top_k)
            pipeline_label = "production retrieval pipeline (same as RAGService)"
        if args.reranker_model:
            options["reranker_model_name"] = args.reranker_model
        metrics = evaluate_retrieval(
            cases, args.index, top_k=args.top_k, retrieval_options=options
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError, EmbeddingError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1

    print(f"questions: {len(cases)}")
    print(f"retrieval pipeline: {pipeline_label}")
    ceiling = precision_ceiling(cases, args.top_k)
    for metric_name, value in metrics.items():
        if metric_name.startswith("Precision@"):
            # Precision@k divides by k, so these labels cap it below 100%.
            print(f"{metric_name}: {value:.2f}% (ceiling on this test set: {ceiling:.2f}%)")
        else:
            print(f"{metric_name}: {value:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
