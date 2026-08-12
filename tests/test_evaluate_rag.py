"""Tests for the simple end-to-end RAG evaluation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from evaluate_rag import (
    DEFAULT_CASES_PATH,
    answer_contains_keywords,
    answer_refuses_unknown,
    evaluate_rag,
    load_evaluation_cases,
    write_report,
)


class RAGEvaluationTest(unittest.TestCase):
    def test_evaluates_three_metrics_and_case_details(self) -> None:
        cases = [
            {
                "question": "已知问题一",
                "expected_source": "a.txt",
                "expected_answer_keywords": ["事实 A", "数字 3"],
            },
            {
                "question": "已知问题二",
                "expected_source": "b.txt",
                "expected_answer_keywords": ["事实 B"],
            },
            {
                "question": "未知问题",
                "expected_source": None,
                "expected_answer_keywords": [],
            },
        ]
        ask_function = Mock(
            side_effect=[
                {
                    "answer": "答案包含事实 A 和数字3。",
                    "sources": [{"source": "a.txt"}, {"source": "x.txt"}],
                },
                {
                    "answer": "答案没有预期内容。",
                    "sources": [{"source": "x.txt"}, {"source": "b.txt"}],
                },
                {
                    "answer": "根据提供的 context 无法回答该问题。",
                    "sources": [{"source": "x.txt"}],
                },
            ]
        )

        report = evaluate_rag(cases, ask_function, top_k=1)

        self.assertEqual(
            report["summary"],
            {
                "total_cases": 3,
                "answerable_cases": 2,
                "no_answer_cases": 1,
                "retrieval_recall_at_1": 0.5,
                "answer_keyword_pass_rate": 0.5,
                "no_answer_refusal_rate": 1.0,
            },
        )
        self.assertTrue(report["cases"][0]["retrieval_hit"])
        self.assertEqual(
            report["cases"][0]["matched_answer_keywords"],
            ["事实 A", "数字 3"],
        )
        self.assertFalse(report["cases"][1]["retrieval_hit"])
        self.assertTrue(report["cases"][2]["no_answer_refusal_pass"])
        self.assertEqual(ask_function.call_count, 3)

    def test_keyword_check_ignores_case_and_whitespace(self) -> None:
        passed, matched = answer_contains_keywords(
            "Use Multi Factor Authentication and wait 30 minutes.",
            ["multi factor authentication", "30minutes"],
        )

        self.assertTrue(passed)
        self.assertEqual(matched, ["multi factor authentication", "30minutes"])

    def test_refusal_check_requires_explicit_refusal_phrase(self) -> None:
        self.assertTrue(answer_refuses_unknown("现有信息不足，无法确定答案。"))
        self.assertFalse(answer_refuses_unknown("总部食堂周末九点开门。"))

    def test_default_cases_include_answerable_and_no_answer_questions(self) -> None:
        cases = load_evaluation_cases(DEFAULT_CASES_PATH)

        self.assertEqual(len(cases), 12)
        self.assertEqual(
            sum(case["expected_source"] is None for case in cases),
            2,
        )

    def test_rejects_no_answer_case_with_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "question": "未知问题",
                            "expected_source": None,
                            "expected_answer_keywords": ["编造的事实"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_evaluation_cases(path)

    def test_writes_readable_json_report(self) -> None:
        report = {"summary": {"retrieval_recall_at_3": 1.0}, "cases": []}
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "report.json"

            write_report(report, path)

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                report,
            )


if __name__ == "__main__":
    unittest.main()
