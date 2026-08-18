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
    "does not contain enough information",
    "not enough information",
    "no relevant information",
    "cannot answer",
    "unable to answer",
)


def load_evaluation_cases(path: str | Path) -> list[dict[str, object]]:
    """Load and validate answerable and no-answer evaluation cases."""
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("The evaluation data must be a non-empty JSON array.")

    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("Every evaluation case must be a JSON object.")
        if set(item) != {
            "question",
            "expected_source",
            "expected_answer_keywords",
        }:
            raise ValueError(
                "Every evaluation case must contain exactly question, "
                "expected_source, and expected_answer_keywords."
            )

        question = item["question"]
        expected_source = item["expected_source"]
        expected_keywords = item["expected_answer_keywords"]

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")
        if expected_source is not None and (
            not isinstance(expected_source, str) or not expected_source.strip()
        ):
            raise ValueError("expected_source must be a non-empty string or null.")
        if not isinstance(expected_keywords, list) or any(
            not isinstance(keyword, str) or not keyword.strip()
            for keyword in expected_keywords
        ):
            raise ValueError(
                "expected_answer_keywords must be an array of non-empty strings."
            )
        if expected_source is None and expected_keywords:
            raise ValueError(
                "expected_answer_keywords must be an empty array for a no-answer question."
            )
        if expected_source is not None and not expected_keywords:
            raise ValueError(
                "An answerable question needs at least one expected_answer_keywords entry."
            )

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
        raise ValueError("The evaluation data must not be empty.")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

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
            raise ValueError(
                "Invalid evaluation data format; load it with load_evaluation_cases first."
            )

        response = ask_function(question)
        if not isinstance(response, dict):
            raise ValueError("ask_function must return a dictionary.")
        answer = response.get("answer")
        sources = response.get("sources")
        if not isinstance(answer, str) or not isinstance(sources, list):
            raise ValueError("The QA result must contain a string answer and an array of sources.")

        retrieved_sources = []
        for source_item in sources[:top_k]:
            if not isinstance(source_item, dict) or not isinstance(
                source_item.get("source"), str
            ):
                raise ValueError("Every item in sources must contain a string source.")
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
    parser = argparse.ArgumentParser(description="Run a simple end-to-end RAG evaluation.")
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
        print(f"Evaluation failed: {exc}")
        return 1

    summary = report["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Full report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
