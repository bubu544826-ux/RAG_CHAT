"""Tests for label-based ranked retrieval evaluation."""

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from evaluate_retrieval import (
    DEFAULT_TEST_SET_PATH,
    calculate_metrics_at_cutoffs,
    evaluate_pipeline_comparison,
    evaluate_retrieval,
    load_evaluation_cases,
    main,
)


def _case(
    case_id: str = "case-1",
    relevant_chunk_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": case_id,
        "question": f"Question for {case_id}",
        "relevant_chunk_ids": (
            ["chunk-a"] if relevant_chunk_ids is None else relevant_chunk_ids
        ),
        "ground_truth": "Metadata that retrieval evaluation does not use.",
    }


class RetrievalEvaluationTest(unittest.TestCase):
    def test_calculates_multiple_cutoffs_from_shared_rankings(self) -> None:
        metrics = calculate_metrics_at_cutoffs(
            [_case("case", ["relevant"])],
            [["other", "relevant"]],
            cutoffs=(1, 2),
        )

        self.assertEqual(metrics["Recall@1"], 0.0)
        self.assertEqual(metrics["Recall@2"], 100.0)
        self.assertEqual(metrics["Precision@2"], 50.0)
        self.assertEqual(metrics["MRR@2"], 50.0)

    @patch("evaluate_retrieval.retrieve", return_value=[{"chunk_id": "chunk-a"}])
    def test_pipeline_comparison_runs_all_four_experiments(
        self, retrieve_mock: Mock
    ) -> None:
        report = evaluate_pipeline_comparison([_case()], "index")

        self.assertEqual(
            set(report),
            {"vector_baseline", "hybrid", "hybrid_reranker", "full_pipeline"},
        )
        self.assertEqual(retrieve_mock.call_count, 4)
        self.assertEqual(
            report["full_pipeline"]["metrics"]["Recall@10"], 100.0
        )

    def test_calculates_exact_macro_averaged_ranked_metrics(self) -> None:
        cases = [
            _case("multiple-hits", ["chunk-a", "chunk-b"]),
            _case("first-hit", ["chunk-c"]),
            _case("miss", ["chunk-d", "chunk-e"]),
        ]
        retrieve_function = Mock(
            side_effect=[
                [
                    {"chunk_id": "other-1"},
                    {"chunk_id": "chunk-b"},
                    {"chunk_id": "chunk-a"},
                ],
                [
                    {"chunk_id": "chunk-c"},
                    {"chunk_id": "other-2"},
                    {"chunk_id": "other-3"},
                ],
                [{"chunk_id": "other-4"}, {"chunk_id": "other-5"}],
            ]
        )

        metrics = evaluate_retrieval(
            cases,
            "index",
            top_k=3,
            retrieve_function=retrieve_function,
        )

        first_case_ndcg = (
            1 / math.log2(3) + 1 / math.log2(4)
        ) / (1 / math.log2(2) + 1 / math.log2(3))
        self.assertAlmostEqual(metrics["Recall@3"], 200 / 3)
        self.assertAlmostEqual(metrics["Precision@3"], 100 / 3)
        self.assertAlmostEqual(metrics["MRR@3"], 50.0)
        self.assertAlmostEqual(metrics["NDCG@3"], (first_case_ndcg + 1) / 3 * 100)

    def test_retrieves_once_per_question_with_requested_k_and_slices_results(self) -> None:
        cases = [_case("late-hit", ["relevant"])]
        retrieve_function = Mock(
            return_value=[
                {"chunk_id": "other-1"},
                {"chunk_id": "other-2"},
                {"chunk_id": "other-3"},
                {"chunk_id": "relevant"},
            ]
        )

        metrics = evaluate_retrieval(
            cases,
            "index",
            top_k=3,
            retrieve_function=retrieve_function,
        )

        retrieve_function.assert_called_once_with(
            cases[0]["question"], "index", top_k=3
        )
        self.assertEqual(
            metrics,
            {"Recall@3": 0.0, "Precision@3": 0.0, "MRR@3": 0.0, "NDCG@3": 0.0},
        )

    def test_returns_unrounded_percentage_values(self) -> None:
        cases = [_case("hit", ["relevant"]), _case("miss", ["missing"])]
        retrieve_function = Mock(
            side_effect=[[{"chunk_id": "relevant"}], [{"chunk_id": "other"}]]
        )

        metrics = evaluate_retrieval(
            cases, "index", top_k=2, retrieve_function=retrieve_function
        )

        self.assertEqual(
            metrics,
            {
                "Recall@2": 50.0,
                "Precision@2": 25.0,
                "MRR@2": 50.0,
                "NDCG@2": 50.0,
            },
        )
        self.assertTrue(all(0 <= value <= 100 for value in metrics.values()))

    def test_rejects_invalid_top_k_and_empty_cases(self) -> None:
        for invalid_top_k in (0, -1):
            with self.subTest(top_k=invalid_top_k), self.assertRaises(ValueError):
                evaluate_retrieval([_case()], "index", top_k=invalid_top_k)
        for invalid_top_k in (True, 1.5, "3"):
            with self.subTest(top_k=invalid_top_k), self.assertRaises(TypeError):
                evaluate_retrieval([_case()], "index", top_k=invalid_top_k)
        with self.assertRaises(ValueError):
            evaluate_retrieval([], "index")

    def test_rejects_invalid_case_labels_when_called_directly(self) -> None:
        invalid_cases = [
            {"id": "missing-label", "question": "Question"},
            _case("empty-label", []),
            _case("duplicate-label", ["chunk-a", "chunk-a"]),
        ]

        for invalid_case in invalid_cases:
            with self.subTest(case=invalid_case), self.assertRaises(ValueError):
                evaluate_retrieval([invalid_case], "index")

    def test_rejects_malformed_retrieval_results(self) -> None:
        malformed_results = [
            "not a list",
            ["not an object"],
            [{}],
            [{"chunk_id": 1}],
            [{"chunk_id": "same"}, {"chunk_id": "same"}],
        ]

        for results in malformed_results:
            with self.subTest(results=results), self.assertRaises(ValueError):
                evaluate_retrieval(
                    [_case()],
                    "index",
                    retrieve_function=Mock(return_value=results),
                )

    def test_accepts_empty_retrieval_results_as_a_miss(self) -> None:
        metrics = evaluate_retrieval(
            [_case()], "index", retrieve_function=Mock(return_value=[])
        )

        self.assertEqual(
            metrics,
            {"Recall@3": 0.0, "Precision@3": 0.0, "MRR@3": 0.0, "NDCG@3": 0.0},
        )

    def test_loader_requires_unique_ids_questions_and_relevance_labels(self) -> None:
        invalid_test_sets = [
            [{"question": "Question", "relevant_chunk_ids": ["chunk-a"]}],
            [{"id": "case", "question": " ", "relevant_chunk_ids": ["chunk-a"]}],
            [{"id": "case", "question": "Question"}],
            [_case("case", [])],
            [_case("case", ["chunk-a", "chunk-a"])],
            [_case("case"), _case("case")],
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "cases.json"
            for test_set in invalid_test_sets:
                with self.subTest(test_set=test_set):
                    path.write_text(
                        json.dumps(test_set, ensure_ascii=False), encoding="utf-8"
                    )
                    with self.assertRaises(ValueError):
                        load_evaluation_cases(path)

    def test_default_test_set_contains_twenty_valid_annotated_cases(self) -> None:
        cases = load_evaluation_cases(DEFAULT_TEST_SET_PATH)

        self.assertEqual(len(cases), 20)
        self.assertEqual(len({case["id"] for case in cases}), 20)
        self.assertTrue(all(case["relevant_chunk_ids"] for case in cases))
        self.assertTrue(all("ground_truth" in case for case in cases))

    @patch("evaluate_retrieval.evaluate_retrieval")
    def test_cli_prints_all_four_percentage_metrics(self, evaluate_mock: Mock) -> None:
        evaluate_mock.return_value = {
            "Recall@5": 75.125,
            "Precision@5": 50.0,
            "MRR@5": 62.5,
            "NDCG@5": 70.25,
        }
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["--top-k", "5"])

        self.assertEqual(exit_code, 0)
        self.assertIn("问题数量：20", output.getvalue())
        self.assertIn("Recall@5: 75.12%", output.getvalue())
        self.assertIn("Precision@5: 50.00%", output.getvalue())
        self.assertIn("MRR@5: 62.50%", output.getvalue())
        self.assertIn("NDCG@5: 70.25%", output.getvalue())
        evaluate_mock.assert_called_once()
        self.assertEqual(evaluate_mock.call_args.kwargs, {"top_k": 5})


if __name__ == "__main__":
    unittest.main()
