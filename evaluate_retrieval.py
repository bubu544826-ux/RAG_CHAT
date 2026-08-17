"""Evaluate ranked chunk retrieval against explicit relevance labels."""

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from src.rag_app.embedding import EmbeddingError
from src.rag_app.retriever import retrieve


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
        raise ValueError("评估测试集必须是非空 JSON 数组。")

    seen_case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("每个评估样本必须是 JSON 对象。")

        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("每个评估样本都必须包含非空 id。")
        if case_id in seen_case_ids:
            raise ValueError(f"评估样本 id 不能重复：{case_id}")
        seen_case_ids.add(case_id)

        question = case.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"评估样本 {case_id} 必须包含非空 question。")

        relevant_chunk_ids = case.get("relevant_chunk_ids")
        if not isinstance(relevant_chunk_ids, list) or not relevant_chunk_ids:
            raise ValueError(
                f"评估样本 {case_id} 必须包含非空 relevant_chunk_ids 列表。"
            )
        if any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in relevant_chunk_ids
        ):
            raise ValueError(
                f"评估样本 {case_id} 的 relevant_chunk_ids 必须是非空字符串。"
            )
        if len(relevant_chunk_ids) != len(set(relevant_chunk_ids)):
            raise ValueError(
                f"评估样本 {case_id} 的 relevant_chunk_ids 不能重复。"
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
        raise ValueError("Retriever 返回结果必须是列表。")

    chunk_ids: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Retriever 的每条结果必须是对象。")
        chunk_id = result.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Retriever 的每条结果都必须包含非空 chunk_id。")
        if chunk_id in chunk_ids:
            raise ValueError(f"Retriever 返回了重复 chunk_id：{chunk_id}")
        chunk_ids.append(chunk_id)

    return chunk_ids


def evaluate_retrieval(
    cases: list[dict[str, object]],
    index_path: str | Path,
    top_k: int = 3,
    retrieve_function: Callable[..., list[dict[str, object]]] = retrieve,
) -> dict[str, float]:
    """Macro-average Recall, Precision, MRR, and binary NDCG at ``top_k``."""
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k 必须是整数。")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0。")
    _validate_evaluation_cases(cases)

    recall_total = 0.0
    precision_total = 0.0
    reciprocal_rank_total = 0.0
    ndcg_total = 0.0

    for case in cases:
        question = case["question"]
        relevant_ids = set(case["relevant_chunk_ids"])
        results = retrieve_function(question, index_path, top_k=top_k)
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
        raise ValueError("排名数量必须与评估样本数量一致。")

    metrics: dict[str, float] = {}
    for top_k in cutoffs:
        if top_k <= 0:
            raise ValueError("评估 cutoff 必须大于 0。")
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


def evaluate_pipeline_comparison(
    cases: list[dict[str, object]],
    index_path: str | Path,
) -> dict[str, dict[str, object]]:
    """Run vector, hybrid, reranked, and rewritten retrieval experiments."""
    experiments = {
        "vector_baseline": {
            "strategy": "vector_only",
            "vector_top_k": 30,
            "candidate_k": 30,
        },
        "hybrid": {
            "strategy": "hybrid",
            "vector_top_k": 30,
            "lexical_top_k": 30,
            "candidate_k": 30,
            "rrf_k": 60,
        },
        "hybrid_reranker": {
            "strategy": "hybrid",
            "vector_top_k": 30,
            "lexical_top_k": 30,
            "candidate_k": 30,
            "rrf_k": 60,
            "reranker_enabled": True,
        },
        "full_pipeline": {
            "strategy": "hybrid",
            "vector_top_k": 30,
            "lexical_top_k": 30,
            "candidate_k": 30,
            "rrf_k": 60,
            "reranker_enabled": True,
            "query_rewrite_enabled": True,
            "query_rewrite_mode": "multi_query",
            "max_queries": 3,
        },
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
    parser = argparse.ArgumentParser(description="使用显式相关 chunk 标注评估 Retriever。")
    parser.add_argument(
        "--test-set",
        "--questions",
        dest="test_set",
        type=Path,
        default=DEFAULT_TEST_SET_PATH,
        help="带 relevant_chunk_ids 标注的评估 JSON 文件路径",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Chroma index 目录",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="参与评估的排名结果数量（默认：3）",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="运行 vector/hybrid/reranker/rewrite 四组对照实验",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="四组实验 JSON 报告路径",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run ranked retrieval evaluation and print percentage metrics."""
    args = _parse_arguments(arguments)

    try:
        cases = load_evaluation_cases(args.test_set)
        if args.compare:
            report = evaluate_pipeline_comparison(cases, args.index)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"问题数量：{len(cases)}")
            for pipeline_name, result in report.items():
                metrics = result["metrics"]
                latency = result["latency_ms"]
                print(
                    f"{pipeline_name}: Recall@5={metrics['Recall@5']:.2f}% "
                    f"Recall@10={metrics['Recall@10']:.2f}% "
                    f"Precision@5={metrics['Precision@5']:.2f}% "
                    f"MRR@10={metrics['MRR@10']:.2f}% "
                    f"NDCG@5={metrics['NDCG@5']:.2f}% "
                    f"mean_latency={latency['mean']:.2f}ms"
                )
            print(f"报告：{args.output}")
            return 0
        metrics = evaluate_retrieval(cases, args.index, top_k=args.top_k)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, EmbeddingError) as exc:
        print(f"评估失败：{exc}")
        return 1

    print(f"问题数量：{len(cases)}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
