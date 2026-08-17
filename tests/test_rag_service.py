"""Tests for the end-to-end RAG service orchestration."""

import json
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
                call.retrieve(
                    "什么是 RAG？",
                    Path("custom-index.json"),
                    top_k=2,
                    strategy="hybrid",
                    vector_top_k=30,
                    lexical_top_k=30,
                    candidate_k=30,
                    rrf_k=60,
                    reranker_enabled=True,
                    reranker_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
                    query_rewrite_enabled=True,
                    query_rewrite_mode="multi_query",
                    max_queries=3,
                ),
                call.build_prompt("什么是 RAG？", chunks),
                call.generate("built prompt"),
            ],
        )

    @patch("src.rag_app.rag_service.time.perf_counter", side_effect=[10.0, 10.125])
    @patch("src.rag_app.rag_service.build_prompt", return_value="built prompt")
    @patch("src.rag_app.rag_service.retrieve")
    def test_logs_request_metadata_without_embedding(
        self,
        retrieve_mock: Mock,
        _build_prompt_mock: Mock,
        _perf_counter_mock: Mock,
    ) -> None:
        retrieve_mock.return_value = [
            {
                "text": "RAG 是检索增强生成。",
                "source": "rag.txt",
                "chunk_id": "rag.txt#chunk-0",
                "score": 0.92,
                "embedding": [12345.6789, 98765.4321],
            }
        ]
        generator = Mock()
        generator.generate.return_value = "final answer"
        service = RAGService(generator=generator)

        with self.assertLogs("src.rag_app.rag_service", level="INFO") as logs:
            service.ask("什么是 RAG？")

        record = json.loads(logs.records[-1].getMessage())
        self.assertEqual(record["question"], "什么是 RAG？")
        self.assertEqual(record["retrieved_chunk_ids"], ["rag.txt#chunk-0"])
        self.assertEqual(record["similarity_scores"], [0.92])
        self.assertEqual(record["request_latency_ms"], 125.0)
        self.assertTrue(record["success"])
        self.assertIsNone(record["error"])
        self.assertNotIn("embedding", logs.output[-1])
        self.assertNotIn("12345.6789", logs.output[-1])

    @patch("src.rag_app.rag_service.time.perf_counter", side_effect=[20.0, 20.05])
    @patch("src.rag_app.rag_service.build_prompt", return_value="built prompt")
    @patch("src.rag_app.rag_service.retrieve")
    def test_logs_failure_and_redacts_api_keys(
        self,
        retrieve_mock: Mock,
        _build_prompt_mock: Mock,
        _perf_counter_mock: Mock,
    ) -> None:
        retrieve_mock.return_value = [
            {
                "text": "context",
                "source": "rag.txt",
                "chunk_id": 1,
                "score": 0.8,
            }
        ]
        api_key = "sk-ant-test-secret-12345678"
        generator = Mock()
        generator.generate.side_effect = RuntimeError(f"request failed: {api_key}")
        service = RAGService(generator=generator)

        with self.assertLogs("src.rag_app.rag_service", level="INFO") as logs:
            with self.assertRaises(RuntimeError):
                service.ask(f"使用 API_KEY={api_key} 测试")

        record = json.loads(logs.records[-1].getMessage())
        self.assertEqual(record["question"], "使用 [REDACTED] 测试")
        self.assertEqual(record["retrieved_chunk_ids"], [1])
        self.assertEqual(record["similarity_scores"], [0.8])
        self.assertEqual(record["request_latency_ms"], 50.0)
        self.assertFalse(record["success"])
        self.assertEqual(record["error"], "request failed: [REDACTED]")
        self.assertNotIn(api_key, logs.output[-1])


if __name__ == "__main__":
    unittest.main()
