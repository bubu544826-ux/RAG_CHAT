"""Tests for the character-based document chunker."""

import unittest

from src.rag_app.chunker import chunk_document


class ChunkDocumentTest(unittest.TestCase):
    def test_splits_text_using_chunk_size(self) -> None:
        document = {"text": "abcdefghij", "source": "example.txt"}

        chunks = chunk_document(document, chunk_size=4)

        self.assertEqual(
            chunks,
            [
                {
                    "text": "abcd",
                    "source": "example.txt",
                    "chunk_id": "example.txt#chunk-0",
                },
                {
                    "text": "efgh",
                    "source": "example.txt",
                    "chunk_id": "example.txt#chunk-1",
                },
                {
                    "text": "ij",
                    "source": "example.txt",
                    "chunk_id": "example.txt#chunk-2",
                },
            ],
        )

    def test_repeats_overlap_between_adjacent_chunks(self) -> None:
        document = {"text": "abcdefghij", "source": "example.txt"}

        chunks = chunk_document(document, chunk_size=4, overlap=2)

        self.assertEqual(
            [chunk["text"] for chunk in chunks],
            ["abcd", "cdef", "efgh", "ghij"],
        )

    def test_rejects_invalid_chunk_settings(self) -> None:
        document = {"text": "abc", "source": "example.txt"}

        with self.assertRaises(ValueError):
            chunk_document(document, chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_document(document, chunk_size=3, overlap=-1)
        with self.assertRaises(ValueError):
            chunk_document(document, chunk_size=3, overlap=3)


if __name__ == "__main__":
    unittest.main()
