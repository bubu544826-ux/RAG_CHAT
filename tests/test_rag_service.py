"""Tests for the end-to-end RAG service orchestration."""

import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from src.rag_app.rag_service import RAGService


class RAGServiceTest(unittest.TestCase):
    @patch("src.rag_app.rag_service.build_prompt")
    @patch("src.rag_app.rag_service.retrieve")
    def test_ask_runs_retrieve_build_prompt_generate_in_order(
        self,
        retrieve_mock: Mock,
        build_prompt_mock: Mock,
    ) -> None:
        chunks = [
            {
                "text": "RAG 是检索增强生成。",
                "source": "rag.txt",
                "chunk_id": 3,
                "score": 0.92,
            }
        ]
        retrieve_mock.return_value = chunks
        build_prompt_mock.return_value = "built prompt"
        generator = Mock()
        generator.generate.return_value = "final answer"
        service = RAGService(
            index_path="custom-index.json",
            top_k=2,
            generator=generator,
        )
        flow = Mock()
        flow.attach_mock(retrieve_mock, "retrieve")
        flow.attach_mock(build_prompt_mock, "build_prompt")
        flow.attach_mock(generator.generate, "generate")

        result = service.ask("什么是 RAG？")

        self.assertEqual(
            result,
            {
                "answer": "final answer",
                "sources": [
                    {"source": "rag.txt", "chunk_id": 3, "score": 0.92}
                ],
            },
        )
        self.assertEqual(
            flow.mock_calls,
            [
                call.retrieve("什么是 RAG？", Path("custom-index.json"), top_k=2),
                call.build_prompt("什么是 RAG？", chunks),
                call.generate("built prompt"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
