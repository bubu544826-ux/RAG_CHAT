"""Run a small end-to-end evaluation for the RAG pipeline."""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = PROJECT_ROOT / "evaluation_cases.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "evaluation_report.json"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "index.json"
DEFAULT_TOP_K = 3
DEFAULT_REFUSAL_PHRASES = (
    "无法回答",
    "没有足够信息",
    "未提供相关信息",
    "不能确定",
    "无法确定",
)


def load_evaluation_cases(path: str | Path) -> list[dict[str, object]]:
    """Load and validate answerable and no-answer evaluation cases."""
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("评估数据必须是非空 JSON 数组。")

    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("每条评估数据必须是 JSON 对象。")
        if set(item) != {
            "question",
            "expected_source",
            "expected_answer_keywords",
        }:
            raise ValueError(
                "每条评估数据必须且只能包含 question、expected_source 和 "
                "expected_answer_keywords。"
            )

        question = item["question"]
        expected_source = item["expected_source"]
        expected_keywords = item["expected_answer_keywords"]

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 必须是非空字符串。")
        if expected_source is not None and (
            not isinstance(expected_source, str) or not expected_source.strip()
        ):
            raise ValueError("expected_source 必须是非空字符串或 null。")
        if not isinstance(expected_keywords, list) or any(
            not isinstance(keyword, str) or not keyword.strip()
            for keyword in expected_keywords
        ):
            raise ValueError("expected_answer_keywords 必须是非空字符串组成的数组。")
        if expected_source is None and expected_keywords:
            raise ValueError("无答案问题的 expected_answer_keywords 必须为空数组。")
        if expected_source is not None and not expected_keywords:
            raise ValueError("有答案问题必须提供至少一个 expected_answer_keywords。")

    return cases


def _normalize_text(text: str) -> str:
    """Make simple keyword checks insensitive to case and whitespace."""
    return "".join(text.casefold().split())


def answer_contains_keywords(answer: str, keywords: list[str]) -> tuple[bool, list[str]]:
    """Return whether every expected keyword occurs in the answer."""
    normalized_answer = _normalize_text(answer)
    matched_keywords = [
        keyword
        for keyword in keywords
        if _normalize_text(keyword) in normalized_answer
    ]
    return len(matched_keywords) == len(keywords), matched_keywords


def answer_refuses_unknown(
    answer: str,
    refusal_phrases: Sequence[str] = DEFAULT_REFUSAL_PHRASES,
) -> bool:
    """Detect an explicit refusal using a small, readable phrase list."""
    normalized_answer = _normalize_text(answer)
    return any(
        _normalize_text(phrase) in normalized_answer for phrase in refusal_phrases
    )


def evaluate_rag(
    cases: list[dict[str, object]],
    ask_function: Callable[[str], dict[str, object]],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, object]:
    """Evaluate retrieval, answer keywords, and no-answer refusals."""
    if not cases:
        raise ValueError("评估数据不能为空。")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k 必须是整数。")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0。")

    retrieval_hits = 0
    keyword_passes = 0
    refusal_passes = 0
    answerable_count = 0
    no_answer_count = 0
    case_results: list[dict[str, object]] = []

    for item in cases:
        question = item["question"]
        expected_source = item["expected_source"]
        expected_keywords = item["expected_answer_keywords"]
        if not isinstance(question, str) or not isinstance(expected_keywords, list):
            raise ValueError("评估数据格式无效，请先使用 load_evaluation_cases 加载。")

        response = ask_function(question)
        if not isinstance(response, dict):
            raise ValueError("ask_function 必须返回字典。")
        answer = response.get("answer")
        sources = response.get("sources")
        if not isinstance(answer, str) or not isinstance(sources, list):
            raise ValueError("问答结果必须包含字符串 answer 和数组 sources。")

        retrieved_sources = []
        for source_item in sources[:top_k]:
            if not isinstance(source_item, dict) or not isinstance(
                source_item.get("source"), str
            ):
                raise ValueError("sources 中的每项都必须包含字符串 source。")
            retrieved_sources.append(source_item["source"])

        result: dict[str, object] = {
            "question": question,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "answer": answer,
        }

        if expected_source is not None:
            answerable_count += 1
            retrieval_hit = expected_source in retrieved_sources
            keyword_pass, matched_keywords = answer_contains_keywords(
                answer,
                expected_keywords,
            )
            retrieval_hits += int(retrieval_hit)
            keyword_passes += int(keyword_pass)
            result.update(
                {
                    "retrieval_hit": retrieval_hit,
                    "expected_answer_keywords": expected_keywords,
                    "matched_answer_keywords": matched_keywords,
                    "answer_keyword_pass": keyword_pass,
                }
            )
        else:
            no_answer_count += 1
            refusal_pass = answer_refuses_unknown(answer)
            refusal_passes += int(refusal_pass)
            result["no_answer_refusal_pass"] = refusal_pass

        case_results.append(result)

    return {
        "summary": {
            "total_cases": len(cases),
            "answerable_cases": answerable_count,
            "no_answer_cases": no_answer_count,
            f"retrieval_recall_at_{top_k}": (
                retrieval_hits / answerable_count if answerable_count else None
            ),
            "answer_keyword_pass_rate": (
                keyword_passes / answerable_count if answerable_count else None
            ),
            "no_answer_refusal_rate": (
                refusal_passes / no_answer_count if no_answer_count else None
            ),
        },
        "cases": case_results,
    }


def write_report(report: dict[str, object], path: str | Path) -> None:
    """Write the evaluation report as readable JSON."""
    Path(path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行简单的端到端 RAG evaluation。")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the RAG evaluation and save a JSON report."""
    args = _parse_arguments(arguments)

    try:
        from src.rag_app.rag_service import RAGService

        cases = load_evaluation_cases(args.cases)
        service = RAGService(index_path=args.index, top_k=args.top_k)
        report = evaluate_rag(cases, service.ask, top_k=args.top_k)
        write_report(report, args.output)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
        print(f"评估失败：{exc}")
        return 1

    summary = report["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"完整报告：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
