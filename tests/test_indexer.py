"""Tests for the local JSON indexing pipeline."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.rag_app.indexer import build_index


class BuildIndexTest(unittest.TestCase):
    def test_builds_json_records_for_every_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_directory = temporary_path / "raw"
            output_path = temporary_path / "processed" / "index.json"
            input_directory.mkdir()
            (input_directory / "a.txt").write_text("abcdef", encoding="utf-8")
            (input_directory / "b.md").write_text("xyz", encoding="utf-8")
            embedding_function = Mock(
                side_effect=lambda text: [float(len(text)), 1.0]
            )

            summary = build_index(
                input_directory,
                output_path,
                chunk_size=4,
                overlap=0,
                embedding_function=embedding_function,
            )

            records = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary,
                {"file_count": 2, "chunk_count": 3, "embedding_count": 3},
            )
            self.assertEqual(
                records,
                [
                    {
                        "source": "a.txt",
                        "chunk_id": "a.txt#chunk-0",
                        "text": "abcd",
                        "embedding": [4.0, 1.0],
                    },
                    {
                        "source": "a.txt",
                        "chunk_id": "a.txt#chunk-1",
                        "text": "ef",
                        "embedding": [2.0, 1.0],
                    },
                    {
                        "source": "b.md",
                        "chunk_id": "b.md#chunk-0",
                        "text": "xyz",
                        "embedding": [3.0, 1.0],
                    },
                ],
            )
            self.assertEqual(embedding_function.call_count, 3)

    def test_writes_empty_index_for_whitespace_only_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_directory = temporary_path / "raw"
            output_path = temporary_path / "processed" / "index.json"
            input_directory.mkdir()
            (input_directory / "empty.txt").write_text("   ", encoding="utf-8")
            embedding_function = Mock(return_value=[1.0])

            summary = build_index(
                input_directory,
                output_path,
                embedding_function=embedding_function,
            )

            self.assertEqual(
                summary,
                {"file_count": 1, "chunk_count": 0, "embedding_count": 0},
            )
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), [])
            embedding_function.assert_not_called()


if __name__ == "__main__":
    unittest.main()
