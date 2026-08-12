"""Tests for the embedding module."""

import unittest
from unittest.mock import Mock, patch

from src.rag_app.embedding import EmbeddingError, embed_text


class FakeEmbedding:
    """Small stand-in for a NumPy embedding used by Sentence Transformers."""

    def tolist(self) -> list[float]:
        return [0.1, 0.2, 0.3]


class EmbedTextTest(unittest.TestCase):
    @patch("src.rag_app.embedding._load_model")
    def test_returns_embedding_as_float_list(self, load_model: Mock) -> None:
        model = load_model.return_value
        model.encode.return_value = FakeEmbedding()

        vector = embed_text("  RAG 是什么？  ")

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        model.encode.assert_called_once_with(
            "RAG 是什么？",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    @patch("src.rag_app.embedding._load_model")
    def test_rejects_empty_text_without_loading_model(self, load_model: Mock) -> None:
        with self.assertRaises(ValueError):
            embed_text("   ")

        load_model.assert_not_called()

    @patch("src.rag_app.embedding._load_model")
    def test_wraps_model_errors(self, load_model: Mock) -> None:
        load_model.return_value.encode.side_effect = RuntimeError("inference failed")

        with self.assertRaises(EmbeddingError) as context:
            embed_text("hello")

        self.assertIsInstance(context.exception.__cause__, RuntimeError)

    @patch("src.rag_app.embedding._load_model")
    def test_rejects_invalid_model_output(self, load_model: Mock) -> None:
        load_model.return_value.encode.return_value = object()

        with self.assertRaises(EmbeddingError):
            embed_text("hello")


if __name__ == "__main__":
    unittest.main()
