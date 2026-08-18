"""Tests for the ingestion command entry point."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import ingest


class IngestCommandTest(unittest.TestCase):
    @patch("ingest.build_index")
    def test_prints_file_chunk_and_embedding_counts(self, build_index_mock) -> None:
        build_index_mock.return_value = {
            "file_count": 2,
            "chunk_count": 4,
            "embedding_count": 4,
        }
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = ingest.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "files: 2",
                "chunks: 4",
                "embeddings: 4",
                "Chroma index directory: "
                f"{ingest.OUTPUT_PATH.relative_to(ingest.PROJECT_ROOT)}",
            ],
        )
        build_index_mock.assert_called_once()
        positional_args = build_index_mock.call_args.args
        self.assertEqual(
            positional_args,
            (ingest.INPUT_DIRECTORY, ingest.OUTPUT_PATH),
        )
        embedding_function = build_index_mock.call_args.kwargs["embedding_function"]
        self.assertIs(embedding_function.func, ingest.embed_texts)
        self.assertEqual(embedding_function.keywords, {"show_progress": True})


if __name__ == "__main__":
    unittest.main()
