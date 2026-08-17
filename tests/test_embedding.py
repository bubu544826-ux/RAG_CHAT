"""Tests for the embedding module."""

import unittest
from unittest.mock import Mock, patch

from src.rag_app.embedding import EmbeddingError, embed_text, embed_texts


class FakeEmbedding:
    """Small stand-in for a NumPy embedding used by Sentence Transformers."""

    def tolist(self) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeEmbeddingBatch:
    """Small stand-in for a 2D NumPy array of embeddings."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return self._rows


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


class EmbedTextsTest(unittest.TestCase):
    @patch("src.rag_app.embedding._load_model")
    def test_embeds_every_text_in_one_batched_call(self, load_model: Mock) -> None:
        model = load_model.return_value
        model.encode.return_value = FakeEmbeddingBatch([[0.1, 0.2], [0.3, 0.4]])

        vectors = embed_texts(["  第一段  ", "第二段"], batch_size=8)

        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
        model.encode.assert_called_once_with(
            ["第一段", "第二段"],
            batch_size=8,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    @patch("src.rag_app.embedding._load_model")
    def test_returns_empty_list_without_loading_model(self, load_model: Mock) -> None:
        self.assertEqual(embed_texts([]), [])

        load_model.assert_not_called()

    @patch("src.rag_app.embedding._load_model")
    def test_rejects_empty_text_without_loading_model(self, load_model: Mock) -> None:
        with self.assertRaises(ValueError):
            embed_texts(["ok", "   "])

        load_model.assert_not_called()

    @patch("src.rag_app.embedding._load_model")
    def test_rejects_mismatched_embedding_count(self, load_model: Mock) -> None:
        model = load_model.return_value
        model.encode.return_value = FakeEmbeddingBatch([[0.1, 0.2]])

        with self.assertRaises(EmbeddingError):
            embed_texts(["第一段", "第二段"])

    @patch("src.rag_app.embedding._load_model")
    def test_wraps_model_errors(self, load_model: Mock) -> None:
        load_model.return_value.encode.side_effect = RuntimeError("inference failed")

        with self.assertRaises(EmbeddingError) as context:
            embed_texts(["hello"])

        self.assertIsInstance(context.exception.__cause__, RuntimeError)


if __name__ == "__main__":
    unittest.main()
