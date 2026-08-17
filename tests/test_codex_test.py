"""Tests for the requested codex_test_.py entry point."""

import unittest
from unittest.mock import patch

import codex_test_


class MainTest(unittest.TestCase):
    @patch("codex_test_.visualize_embeddings.main", return_value=0)
    def test_delegates_to_3d_visualizer(self, visualizer_main) -> None:
        arguments = ["--output", "embeddings.png"]

        exit_code = codex_test_.main(arguments)

        self.assertEqual(exit_code, 0)
        visualizer_main.assert_called_once_with(arguments)


if __name__ == "__main__":
    unittest.main()
