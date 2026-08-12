"""Tests for retrieving chunks from the local JSON index."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.rag_app.retriever import retrieve


class RetrieverTest(unittest.TestCase):
    def test_returns_top_k_chunks_ordered_by_descending_score(self) -> None:
        records = [
            {
                "text": "不相关",
                "source": "notes.md",
                "chunk_id": "notes.md#chunk-0",
                "embedding": [0.0, 1.0],
            },
            {
                "text": "最相关",
                "source": "rag.md",
                "chunk_id": "rag.md#chunk-0",
                "embedding": [1.0, 0.0],
            },
            {
                "text": "部分相关",
                "source": "rag.md",
                "chunk_id": "rag.md#chunk-1",
                "embedding": [1.0, 1.0],
            },
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            index_path = Path(temporary_directory) / "index.json"
            index_path.write_text(
                json.dumps(records, ensure_ascii=False), encoding="utf-8"
            )
            embedding_function = Mock(return_value=[1.0, 0.0])

            results = retrieve(
                "RAG 是什么？",
                index_path,
                top_k=2,
                embedding_function=embedding_function,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            [result["chunk_id"] for result in results],
            ["rag.md#chunk-0", "rag.md#chunk-1"],
        )
        self.assertAlmostEqual(results[0]["score"], 1.0)
        self.assertAlmostEqual(results[1]["score"], 2**-0.5)
        self.assertNotIn("embedding", results[0])
        embedding_function.assert_called_once_with("RAG 是什么？")

    def test_returns_all_chunks_when_top_k_exceeds_index_size(self) -> None:
        records = [
            {
                "text": "RAG",
                "source": "rag.txt",
                "chunk_id": "rag.txt#chunk-0",
                "embedding": [1.0],
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            index_path = Path(temporary_directory) / "index.json"
            index_path.write_text(json.dumps(records), encoding="utf-8")

            results = retrieve(
                "RAG",
                index_path,
                top_k=5,
                embedding_function=lambda question: [1.0],
            )

        self.assertEqual(len(results), 1)

    def test_rejects_non_positive_top_k(self) -> None:
        with self.assertRaises(ValueError):
            retrieve("RAG", "unused.json", top_k=0)


if __name__ == "__main__":
    unittest.main()
