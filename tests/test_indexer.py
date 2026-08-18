"""Tests for the local Chroma indexing pipeline."""

import unittest
from unittest.mock import Mock

import chromadb

from src.rag_app.indexer import COLLECTION_NAME, build_index
from tests.chroma_test_utils import temporary_chroma_directory


class BuildIndexTest(unittest.TestCase):
    def test_builds_chroma_records_for_every_chunk(self) -> None:
        with temporary_chroma_directory() as temporary_path:
            input_directory = temporary_path / "raw"
            output_path = temporary_path / "processed" / "index.json"
            input_directory.mkdir()
            (input_directory / "a.txt").write_text("abcdef", encoding="utf-8")
            (input_directory / "b.md").write_text("xyz", encoding="utf-8")
            embedding_function = Mock(
                side_effect=lambda texts: [[float(len(text)), 1.0] for text in texts]
            )

            summary = build_index(
                input_directory,
                output_path,
                chunk_size=4,
                overlap=0,
                embedding_function=embedding_function,
            )

            client = chromadb.PersistentClient(path=str(output_path))
            collection = client.get_collection(
                COLLECTION_NAME,
                embedding_function=None,
            )
            stored = collection.get(include=["documents", "metadatas", "embeddings"])

            self.assertTrue(output_path.is_dir())
            self.assertEqual(
                summary,
                {"file_count": 2, "chunk_count": 3, "embedding_count": 3},
            )
            self.assertEqual(collection.count(), 3)
            records_by_id = {
                record_id: {
                    "text": text,
                    "metadata": metadata,
                    "embedding": list(embedding),
                }
                for record_id, text, metadata, embedding in zip(
                    stored["ids"],
                    stored["documents"],
                    stored["metadatas"],
                    stored["embeddings"],
                )
            }
            self.assertEqual(records_by_id["a.txt#chunk-0"]["text"], "abcd")
            self.assertEqual(
                records_by_id["a.txt#chunk-0"]["metadata"],
                {"source": "a.txt", "chunk_id": "a.txt#chunk-0"},
            )
            self.assertEqual(
                records_by_id["a.txt#chunk-0"]["embedding"],
                [4.0, 1.0],
            )
            self.assertEqual(records_by_id["a.txt#chunk-1"]["text"], "ef")
            self.assertEqual(records_by_id["b.md#chunk-0"]["text"], "xyz")
            # All chunks are embedded in a single batched call.
            self.assertEqual(embedding_function.call_count, 1)
            self.assertEqual(
                sorted(embedding_function.call_args.args[0]),
                ["abcd", "ef", "xyz"],
            )

    def test_creates_empty_collection_for_whitespace_only_document(self) -> None:
        with temporary_chroma_directory() as temporary_path:
            input_directory = temporary_path / "raw"
            output_path = temporary_path / "processed" / "index.json"
            input_directory.mkdir()
            (input_directory / "empty.txt").write_text("   ", encoding="utf-8")
            embedding_function = Mock(return_value=[[1.0]])

            summary = build_index(
                input_directory,
                output_path,
                embedding_function=embedding_function,
            )

            client = chromadb.PersistentClient(path=str(output_path))
            collection = client.get_collection(
                COLLECTION_NAME,
                embedding_function=None,
            )
            self.assertEqual(
                summary,
                {"file_count": 1, "chunk_count": 0, "embedding_count": 0},
            )
            self.assertEqual(collection.count(), 0)
            embedding_function.assert_not_called()

    def test_rebuild_removes_chunks_that_are_no_longer_present(self) -> None:
        with temporary_chroma_directory() as temporary_path:
            input_directory = temporary_path / "raw"
            output_path = temporary_path / "processed" / "index.json"
            input_directory.mkdir()
            first_document = input_directory / "first.txt"
            first_document.write_text("first", encoding="utf-8")

            build_index(
                input_directory,
                output_path,
                embedding_function=lambda texts: [[1.0] for _ in texts],
            )
            first_document.unlink()
            (input_directory / "second.txt").write_text("second", encoding="utf-8")

            build_index(
                input_directory,
                output_path,
                embedding_function=lambda texts: [[1.0] for _ in texts],
            )

            client = chromadb.PersistentClient(path=str(output_path))
            collection = client.get_collection(
                COLLECTION_NAME,
                embedding_function=None,
            )
            self.assertEqual(collection.get()["ids"], ["second.txt#chunk-0"])

    def test_backs_up_legacy_json_index_before_creating_chroma_directory(self) -> None:
        with temporary_chroma_directory() as temporary_path:
            input_directory = temporary_path / "raw"
            output_path = temporary_path / "processed" / "index.json"
            input_directory.mkdir()
            output_path.parent.mkdir()
            output_path.write_text('[{"legacy": true}]', encoding="utf-8")

            build_index(
                input_directory,
                output_path,
                embedding_function=lambda texts: [[1.0] for _ in texts],
            )

            self.assertTrue(output_path.is_dir())
            self.assertEqual(
                output_path.with_name("index.json.legacy").read_text(encoding="utf-8"),
                '[{"legacy": true}]',
            )


if __name__ == "__main__":
    unittest.main()
