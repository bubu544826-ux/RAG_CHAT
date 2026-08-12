"""Tests for cosine similarity."""

import unittest

from src.rag_app.similarity import cosine_similarity


class CosineSimilarityTest(unittest.TestCase):
    def test_same_vectors_have_similarity_one(self) -> None:
        score = cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

        self.assertAlmostEqual(score, 1.0)

    def test_different_vectors_have_lower_similarity(self) -> None:
        score = cosine_similarity([1.0, 0.0], [0.0, 1.0])

        self.assertAlmostEqual(score, 0.0)

    def test_rejects_vectors_with_different_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_rejects_zero_vector(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])

    def test_rejects_non_finite_values(self) -> None:
        for invalid_value in (float("nan"), float("inf")):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(ValueError):
                    cosine_similarity([invalid_value, 1.0], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
