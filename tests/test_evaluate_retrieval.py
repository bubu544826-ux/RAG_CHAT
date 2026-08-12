"""Tests for the minimal retrieval evaluation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from evaluate_retrieval import (
    DEFAULT_QUESTIONS_PATH,
    evaluate_retrieval,
    load_evaluation_questions,
)


class RetrievalEvaluationTest(unittest.TestCase):
    def test_calculates_recall_at_1_and_3(self) -> None:
        questions = [
            {"question": "问题一", "expected_source": "a.txt"},
            {"question": "问题二", "expected_source": "b.txt"},
            {"question": "问题三", "expected_source": "c.txt"},
        ]
        retrieve_function = Mock(
            side_effect=[
                [{"source": "a.txt"}, {"source": "x.txt"}],
                [{"source": "x.txt"}, {"source": "b.txt"}],
                [{"source": "x.txt"}, {"source": "y.txt"}],
            ]
        )

        metrics = evaluate_retrieval(
            questions,
            "index.json",
            retrieve_function=retrieve_function,
        )

        self.assertAlmostEqual(metrics["Recall@1"], 1 / 3)
        self.assertAlmostEqual(metrics["Recall@3"], 2 / 3)
        self.assertEqual(retrieve_function.call_count, 3)
        for item, call in zip(questions, retrieve_function.call_args_list):
            self.assertEqual(call.args, (item["question"], "index.json"))
            self.assertEqual(call.kwargs, {"top_k": 3})

    def test_default_question_set_contains_ten_valid_pairs(self) -> None:
        questions = load_evaluation_questions(DEFAULT_QUESTIONS_PATH)

        self.assertEqual(len(questions), 10)
        self.assertTrue(
            all(set(item) == {"question", "expected_source"} for item in questions)
        )

    def test_rejects_question_without_expected_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_path = Path(temporary_directory) / "questions.json"
            invalid_path.write_text(
                json.dumps([{"question": "缺少 source"}], ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_evaluation_questions(invalid_path)

    def test_rejects_empty_question_list(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_retrieval([], "index.json")


if __name__ == "__main__":
    unittest.main()
