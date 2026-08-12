"""Smoke tests for the initialized application."""

import io
import unittest
from contextlib import redirect_stdout

from src.rag_app.app import main


class ApplicationSmokeTest(unittest.TestCase):
    def test_main_starts_successfully(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().strip(), "RAG project initialized successfully.")


if __name__ == "__main__":
    unittest.main()
