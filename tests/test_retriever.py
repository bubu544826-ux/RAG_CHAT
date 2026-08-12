"""Tests for retrieving chunks from a local Chroma index."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import chromadb

from src.rag_app.indexer import COLLECTION_NAME
from src.rag_app.retriever import retrieve
from tests.chroma_test_utils import temporary_chroma_directory


class RetrieverTest(unittest.TestCase):
    def _create_collection(
        self,
        index_path: Path,
        records: list[dict[str, object]],
    ) -> None:
        client = chromadb.PersistentClient(path=str(index_path))
        collection = client.create_collection(
            COLLECTION_NAME,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
        if records:
            collection.add(
                ids=[record["chunk_id"] for record in records],
                documents=[record["text"] for record in records],
                metadatas=[
                    {
                        "source": record["source"],
                        "chunk_id": record["chunk_id"],
                    }
                    for record in records
                ],
                embeddings=[record["embedding"] for record in records],
            )

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

        with temporary_chroma_directory() as temporary_path:
            index_path = temporary_path / "index.json"
            self._create_collection(index_path, records)
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
        self.assertAlmostEqual(results[0]["score"], 1.0, places=5)
        self.assertAlmostEqual(results[1]["score"], 2**-0.5, places=5)
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

        with temporary_chroma_directory() as temporary_path:
            index_path = temporary_path / "index.json"
            self._create_collection(index_path, records)

            results = retrieve(
                "RAG",
                index_path,
                top_k=5,
                embedding_function=lambda question: [1.0],
            )

        self.assertEqual(len(results), 1)

    def test_returns_empty_list_for_empty_collection(self) -> None:
        with temporary_chroma_directory() as temporary_path:
            index_path = temporary_path / "index.json"
            self._create_collection(index_path, [])
            embedding_function = Mock(return_value=[1.0])

            results = retrieve(
                "RAG",
                index_path,
                embedding_function=embedding_function,
            )

        self.assertEqual(results, [])
        embedding_function.assert_not_called()

    def test_rejects_legacy_json_index_with_migration_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            index_path = Path(temporary_directory) / "index.json"
            index_path.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "python ingest.py"):
                retrieve(
                    "RAG",
                    index_path,
                    embedding_function=lambda question: [1.0],
                )

    def test_rejects_non_positive_top_k(self) -> None:
        with self.assertRaises(ValueError):
            retrieve("RAG", "unused.json", top_k=0)


if __name__ == "__main__":
    unittest.main()
