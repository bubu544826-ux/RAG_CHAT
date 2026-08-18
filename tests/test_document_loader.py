"""Tests for the document loader module."""

import tempfile
import unittest
from pathlib import Path

from src.rag_app.document_loader import load_documents


class DocumentLoaderTest(unittest.TestCase):
    def test_loads_txt_and_md_documents_in_common_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_directory = Path(temporary_directory) / "raw"
            raw_directory.mkdir()
            (raw_directory / "rag_notes.md").write_text(
                "# RAG study notes\n\nThe document loader reads the raw documents.\n",
                encoding="utf-8",
            )
            (raw_directory / "rag_overview.txt").write_text(
                "Retrieval-Augmented Generation combines retrieval with generation.\n",
                encoding="utf-8",
            )

            documents = load_documents(raw_directory)

        self.assertEqual(
            documents,
            [
                {
                    "text": "# RAG study notes\n\nThe document loader reads the raw documents.\n",
                    "source": "rag_notes.md",
                },
                {
                    "text": (
                        "Retrieval-Augmented Generation combines retrieval "
                        "with generation.\n"
                    ),
                    "source": "rag_overview.txt",
                },
            ],
        )

    def test_ignores_unsupported_file_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "document.pdf").write_text("ignored", encoding="utf-8")

            self.assertEqual(load_documents(directory), [])

    def test_uses_relative_paths_to_distinguish_same_named_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "first").mkdir()
            (directory / "second").mkdir()
            (directory / "first" / "notes.txt").write_text(
                "first document", encoding="utf-8"
            )
            (directory / "second" / "notes.txt").write_text(
                "second document", encoding="utf-8"
            )

            documents = load_documents(directory)

        self.assertEqual(
            [document["source"] for document in documents],
            ["first/notes.txt", "second/notes.txt"],
        )


if __name__ == "__main__":
    unittest.main()
