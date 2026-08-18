"""Tests for retrieving chunks from a local Chroma index."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import chromadb

from src.rag_app.indexer import COLLECTION_NAME
from src.rag_app.retriever import expand_with_neighbours, retrieve
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


class NeighbourExpansionTest(unittest.TestCase):
    """Cover the adjacency fix for chunks cut on fixed character boundaries."""

    def _records(self, count: int) -> list[dict[str, object]]:
        return [
            {
                "text": f"chunk body {index}",
                "source": "doc.txt",
                "chunk_id": f"doc.txt#chunk-{index}",
                "embedding": [1.0, 0.0],
            }
            for index in range(count)
        ]

    def _populate(self, index_path: Path, records: list[dict[str, object]]) -> None:
        client = chromadb.PersistentClient(path=str(index_path))
        collection = client.create_collection(
            COLLECTION_NAME,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
        collection.add(
            ids=[record["chunk_id"] for record in records],
            documents=[record["text"] for record in records],
            metadatas=[
                {"source": record["source"], "chunk_id": record["chunk_id"]}
                for record in records
            ],
            embeddings=[record["embedding"] for record in records],
        )

    def test_admits_both_neighbours_and_keeps_seed_order(self) -> None:
        with temporary_chroma_directory() as index_path:
            self._populate(index_path, self._records(5))

            expanded = expand_with_neighbours(
                [{"chunk_id": "doc.txt#chunk-2", "text": "chunk body 2"}],
                index_path,
                limit=3,
            )

        self.assertEqual(
            [item["chunk_id"] for item in expanded],
            ["doc.txt#chunk-2", "doc.txt#chunk-3", "doc.txt#chunk-1"],
        )
        self.assertEqual([item["final_rank"] for item in expanded], [1, 2, 3])
        self.assertEqual(expanded[1]["text"], "chunk body 3")

    def test_skips_ids_the_index_does_not_hold(self) -> None:
        with temporary_chroma_directory() as index_path:
            self._populate(index_path, self._records(3))

            # chunk-0 has no left neighbour, chunk-2 no right neighbour.
            first = expand_with_neighbours(
                [{"chunk_id": "doc.txt#chunk-0"}], index_path, limit=5
            )
            last = expand_with_neighbours(
                [{"chunk_id": "doc.txt#chunk-2"}], index_path, limit=5
            )

        self.assertEqual(
            [item["chunk_id"] for item in first],
            ["doc.txt#chunk-0", "doc.txt#chunk-1"],
        )
        self.assertEqual(
            [item["chunk_id"] for item in last],
            ["doc.txt#chunk-2", "doc.txt#chunk-1"],
        )

    def test_deduplicates_when_seeds_are_adjacent(self) -> None:
        with temporary_chroma_directory() as index_path:
            self._populate(index_path, self._records(5))

            expanded = expand_with_neighbours(
                [
                    {"chunk_id": "doc.txt#chunk-1", "text": "chunk body 1"},
                    {"chunk_id": "doc.txt#chunk-2", "text": "chunk body 2"},
                ],
                index_path,
                limit=10,
            )

        chunk_ids = [item["chunk_id"] for item in expanded]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertIn("doc.txt#chunk-2", chunk_ids)

    def test_radius_widens_the_admitted_window(self) -> None:
        with temporary_chroma_directory() as index_path:
            self._populate(index_path, self._records(7))

            expanded = expand_with_neighbours(
                [{"chunk_id": "doc.txt#chunk-3"}], index_path, limit=5, radius=2
            )

        self.assertEqual(
            sorted(item["chunk_id"] for item in expanded),
            [
                "doc.txt#chunk-1",
                "doc.txt#chunk-2",
                "doc.txt#chunk-3",
                "doc.txt#chunk-4",
                "doc.txt#chunk-5",
            ],
        )

    def test_ignores_chunk_ids_that_do_not_carry_a_position(self) -> None:
        with temporary_chroma_directory() as index_path:
            self._populate(index_path, self._records(2))

            expanded = expand_with_neighbours(
                [{"chunk_id": "legacy-id-without-position", "text": "kept"}],
                index_path,
                limit=3,
            )

        self.assertEqual(
            [item["chunk_id"] for item in expanded], ["legacy-id-without-position"]
        )

    def test_rejects_non_positive_radius(self) -> None:
        with self.assertRaises(ValueError):
            expand_with_neighbours(
                [{"chunk_id": "doc.txt#chunk-1"}], "unused", limit=3, radius=0
            )

    def test_retrieve_returns_top_k_with_contiguous_ranks_when_expanding(self) -> None:
        with temporary_chroma_directory() as index_path:
            self._populate(index_path, self._records(6))

            results = retrieve(
                "chunk",
                index_path,
                top_k=4,
                embedding_function=lambda question: [1.0, 0.0],
                neighbour_expansion=True,
            )

        self.assertEqual(len(results), 4)
        self.assertEqual([item["final_rank"] for item in results], [1, 2, 3, 4])
        self.assertEqual(len({item["chunk_id"] for item in results}), 4)


if __name__ == "__main__":
    unittest.main()
