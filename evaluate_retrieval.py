"""Run a minimal source-level evaluation for the local retriever."""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from src.rag_app.embedding import EmbeddingError
from src.rag_app.retriever import retrieve


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "index.json"
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "evaluation_questions.json"


def load_evaluation_questions(path: str | Path) -> list[dict[str, str]]:
    """Load question and expected source pairs from a JSON file."""
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("评估问题必须是非空 JSON 数组。")

    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("每个评估问题必须是 JSON 对象。")
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError("每个评估问题都必须包含非空 question。")
        if (
            not isinstance(item.get("expected_source"), str)
            or not item["expected_source"].strip()
        ):
            raise ValueError("每个评估问题都必须包含非空 expected_source。")

    return questions


def evaluate_retrieval(
    questions: list[dict[str, str]],
    index_path: str | Path,
    retrieve_function: Callable[..., list[dict[str, object]]] = retrieve,
) -> dict[str, float]:
    """Calculate source-level Recall@1 and Recall@3."""
    if not questions:
        raise ValueError("评估问题不能为空。")

    hits_at_1 = 0
    hits_at_3 = 0

    for item in questions:
        results = retrieve_function(item["question"], index_path, top_k=3)
        retrieved_sources = [result["source"] for result in results]
        expected_source = item["expected_source"]

        if expected_source in retrieved_sources[:1]:
            hits_at_1 += 1
        if expected_source in retrieved_sources[:3]:
            hits_at_3 += 1

    question_count = len(questions)
    return {
        "Recall@1": hits_at_1 / question_count,
        "Recall@3": hits_at_3 / question_count,
    }


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 Retriever 的 source 召回率。")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="评估问题 JSON 文件路径",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="index.json 路径",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run evaluation and print the two recall metrics."""
    args = _parse_arguments(arguments)

    try:
        questions = load_evaluation_questions(args.questions)
        metrics = evaluate_retrieval(questions, args.index)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, EmbeddingError) as exc:
        print(f"评估失败：{exc}")
        return 1

    print(f"问题数量：{len(questions)}")
    print(f"Recall@1: {metrics['Recall@1']:.2%}")
    print(f"Recall@3: {metrics['Recall@3']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
