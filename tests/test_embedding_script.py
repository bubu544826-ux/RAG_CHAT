"""Tests for the interactive embedding inspection script."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts.test_embedding import main


class EmbeddingScriptTest(unittest.TestCase):
    @patch("scripts.test_embedding.embed_text", return_value=[0.1, 0.2, 0.3])
    @patch("builtins.input", return_value="What is RAG?")
    def test_prints_only_vector_type_and_dimension(
        self, input_mock, embed_text_mock
    ) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            output.getvalue().splitlines(),
            ["vector type: list", "vector dimension: 3"],
        )
        embed_text_mock.assert_called_once_with("What is RAG?")


if __name__ == "__main__":
    unittest.main()
