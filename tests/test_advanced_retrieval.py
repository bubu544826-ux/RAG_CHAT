"""Tests for hybrid retrieval, reranking, and query rewriting."""

import unittest
from unittest.mock import Mock, patch

from src.rag_app.retriever import (
    generate_search_queries,
    merge_multi_query_results,
    reciprocal_rank_fusion,
    rerank_candidates,
    retrieve,
    retrieve_candidates,
)


def _item(chunk_id: str, score: float = 1.0) -> dict[str, object]:
    return {
        "text": f"text for {chunk_id}",
        "source": "doc.txt",
        "document_id": "doc.txt",
        "chunk_id": chunk_id,
        "score": score,
        "retrieval_score": score,
        "original_rank": 1,
        "final_rank": 1,
        "retrieval_source": ["test"],
    }


class AdvancedRetrievalTest(unittest.TestCase):
    def test_rrf_rewards_chunks_found_by_both_retrievers(self) -> None:
        fused = reciprocal_rank_fusion(
            [
                ("vector", [_item("vector-first"), _item("shared")]),
                ("lexical", [_item("shared"), _item("lexical-second")]),
            ],
            rrf_k=60,
        )

        self.assertEqual(fused[0]["chunk_id"], "shared")
        self.assertEqual(fused[0]["retrieval_source"], ["lexical", "vector"])
        self.assertAlmostEqual(fused[0]["retrieval_score"], 1 / 62 + 1 / 61)

    def test_rrf_deduplicates_within_and_across_rankings(self) -> None:
        fused = reciprocal_rank_fusion(
            [("one", [_item("a"), _item("a")]), ("two", [_item("a")])]
        )

        self.assertEqual([item["chunk_id"] for item in fused], ["a"])

    @patch("src.rag_app.retriever.lexical_search")
    @patch("src.rag_app.retriever.vector_search")
    def test_hybrid_merges_vector_and_lexical_results(
        self, vector_search: Mock, lexical_search: Mock
    ) -> None:
        vector_search.return_value = [_item("semantic"), _item("shared")]
        lexical_search.return_value = [_item("shared"), _item("keyword")]

        results = retrieve_candidates(
            "query",
            "index",
            strategy="hybrid",
            vector_top_k=2,
            lexical_top_k=2,
            candidate_k=3,
            rrf_k=60,
            embedding_function=Mock(),
        )

        self.assertEqual(results[0]["chunk_id"], "shared")
        self.assertEqual(len({item["chunk_id"] for item in results}), 3)

    @patch("src.rag_app.retriever.lexical_search")
    @patch("src.rag_app.retriever.vector_search")
    def test_component_failure_falls_back_to_working_backend(
        self, vector_search: Mock, lexical_search: Mock
    ) -> None:
        vector_search.side_effect = RuntimeError("vector unavailable")
        lexical_search.return_value = [_item("keyword")]

        results = retrieve_candidates(
            "query",
            "index",
            strategy="hybrid",
            vector_top_k=2,
            lexical_top_k=2,
            candidate_k=2,
            rrf_k=60,
            embedding_function=Mock(),
        )

        self.assertEqual([item["chunk_id"] for item in results], ["keyword"])
        self.assertEqual(results[0]["retrieval_source"], ["lexical"])

    @patch("src.rag_app.retriever.lexical_search", return_value=[])
    @patch("src.rag_app.retriever.vector_search", return_value=[])
    def test_empty_backends_return_an_empty_result(
        self, _vector_search: Mock, _lexical_search: Mock
    ) -> None:
        results = retrieve_candidates(
            "query",
            "index",
            strategy="hybrid",
            vector_top_k=2,
            lexical_top_k=2,
            candidate_k=2,
            rrf_k=60,
            embedding_function=Mock(),
        )

        self.assertEqual(results, [])

    def test_reranker_orders_by_new_score_and_retains_original_rank(self) -> None:
        candidates = [_item("a"), _item("b"), _item("c")]
        for rank, candidate in enumerate(candidates, start=1):
            candidate["original_rank"] = rank

        reranked = rerank_candidates(
            "query",
            candidates,
            rerank_function=Mock(return_value=[0.1, 0.9, 0.2]),
            final_k=2,
        )

        self.assertEqual([item["chunk_id"] for item in reranked], ["b", "c"])
        self.assertEqual(reranked[0]["original_rank"], 2)
        self.assertEqual(reranked[0]["final_rank"], 1)
        self.assertEqual(reranked[0]["rerank_score"], 0.9)

    def test_rewrite_rejects_variants_that_drop_constraints(self) -> None:
        rewriter = Mock(
            return_value=[
                "QUIC v1 limit 2^60 not permitted",
                "QUIC v1 limit 2^60 not permitted Toronto",
            ]
        )

        queries = generate_search_queries(
            "Can QUIC v1 use a limit above 2^60 in Toronto and not fail?",
            enabled=True,
            mode="multi_query",
            max_queries=3,
            rewrite_function=rewriter,
        )

        self.assertEqual(len(queries), 2)
        self.assertIn("QUIC v1 limit 2^60 not permitted Toronto", queries)
        self.assertNotIn("QUIC v1 limit 2^60 not permitted", queries)

    def test_precise_query_bypasses_rewriting(self) -> None:
        rewriter = Mock(return_value=["changed query"])

        queries = generate_search_queries(
            "Why does FINAL_SIZE_ERROR happen?",
            enabled=True,
            mode="multi_query",
            max_queries=3,
            rewrite_function=rewriter,
        )

        self.assertEqual(queries, ["Why does FINAL_SIZE_ERROR happen?"])
        rewriter.assert_not_called()

    def test_multi_query_merge_deduplicates_and_records_queries(self) -> None:
        merged = merge_multi_query_results(
            [
                ("original", [_item("shared"), _item("a")]),
                ("rewrite", [_item("shared"), _item("b")]),
            ],
            rrf_k=60,
            limit=3,
        )

        self.assertEqual(merged[0]["chunk_id"], "shared")
        self.assertEqual(len({item["chunk_id"] for item in merged}), 3)
        self.assertEqual(merged[0]["search_queries"], ["original", "rewrite"])

    def test_multi_query_merge_does_not_evict_original_candidates(self) -> None:
        merged = merge_multi_query_results(
            [
                ("original", [_item("original-a"), _item("original-b")]),
                ("rewrite-one", [_item("new-a"), _item("new-b")]),
                ("rewrite-two", [_item("new-a"), _item("new-b")]),
            ],
            rrf_k=60,
            limit=2,
        )

        self.assertEqual(
            {item["chunk_id"] for item in merged},
            {"original-a", "original-b"},
        )

    @patch("src.rag_app.retriever.retrieve_candidates")
    def test_rewrite_and_reranker_failures_use_original_fused_ranking(
        self, retrieve_candidates_mock: Mock
    ) -> None:
        retrieve_candidates_mock.return_value = [_item("a"), _item("b")]

        results = retrieve(
            "ordinary question",
            "index",
            top_k=1,
            strategy="hybrid",
            vector_top_k=2,
            lexical_top_k=2,
            candidate_k=2,
            query_rewrite_enabled=True,
            rewrite_function=Mock(side_effect=RuntimeError("rewrite failed")),
            reranker_enabled=True,
            rerank_function=Mock(side_effect=RuntimeError("reranker failed")),
        )

        self.assertEqual([item["chunk_id"] for item in results], ["a"])
        retrieve_candidates_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
