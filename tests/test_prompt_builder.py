"""Tests for building a prompt from retrieved chunks."""

import unittest

from src.rag_app.prompt_builder import (
    ANSWER_LANGUAGE_REMINDER,
    SYSTEM_INSTRUCTION,
    build_prompt,
)


class PromptBuilderTest(unittest.TestCase):
    def test_builds_instruction_context_and_question_in_order(self) -> None:
        chunks = [
            {
                "text": "RAG retrieves relevant documents first, then lets the model answer.",
                "source": "rag.md",
                "chunk_id": "rag.md#chunk-0",
                "score": 0.95,
            },
            {
                "text": "The retrieved results become the context for the model.",
                "source": "notes.txt",
                "chunk_id": "notes.txt#chunk-1",
                "score": 0.80,
            },
        ]

        prompt = build_prompt("How does RAG work?", chunks)

        instruction_position = prompt.index(SYSTEM_INSTRUCTION)
        context_position = prompt.index("context:")
        question_position = prompt.index("question:")
        self.assertLess(instruction_position, context_position)
        self.assertLess(context_position, question_position)
        self.assertIn("Write the entire answer in English", prompt)
        self.assertIn("Answer from the provided context first", prompt)
        self.assertIn("does not hold enough information", prompt)
        self.assertIn("Do not invent information that is not in the context", prompt)
        self.assertIn("[Chunk 1]", prompt)
        self.assertIn("source: rag.md", prompt)
        self.assertIn("chunk_id: notes.txt#chunk-1", prompt)
        self.assertIn("The retrieved results become the context for the model.", prompt)
        self.assertNotIn("0.95", prompt)
        self.assertIn("question:\nHow does RAG work?", prompt)
        self.assertTrue(prompt.endswith(ANSWER_LANGUAGE_REMINDER))

    def test_marks_context_as_empty_when_no_chunks_are_retrieved(self) -> None:
        prompt = build_prompt("a question outside the knowledge base", [])

        self.assertIn("context:\n(no relevant content retrieved)", prompt)
        self.assertIn("does not hold enough information", prompt)

    def test_rejects_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt("   ", [])

    def test_rejects_chunk_without_text(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt("What is RAG?", [{"source": "rag.md"}])


if __name__ == "__main__":
    unittest.main()
