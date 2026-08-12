"""Tests for the retrieval command entry point."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import retrieve


class RetrieveCommandTest(unittest.TestCase):
    @patch("retrieve.retrieve")
    def test_prints_json_results_and_passes_cli_options(self, retrieve_mock) -> None:
        retrieve_mock.return_value = [
            {
                "text": "RAG 是检索增强生成。",
                "source": "rag.md",
                "chunk_id": "rag.md#chunk-0",
                "score": 0.9,
            }
        ]
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = retrieve.main(
                ["什么是 RAG？", "--top-k", "1", "--index", "custom.json"]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue()), retrieve_mock.return_value)
        retrieve_mock.assert_called_once_with(
            "什么是 RAG？", Path("custom.json"), top_k=1
        )


if __name__ == "__main__":
    unittest.main()
