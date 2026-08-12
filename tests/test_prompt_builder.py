"""Tests for building a prompt from retrieved chunks."""

import unittest

from src.rag_app.prompt_builder import SYSTEM_INSTRUCTION, build_prompt


class PromptBuilderTest(unittest.TestCase):
    def test_builds_instruction_context_and_question_in_order(self) -> None:
        chunks = [
            {
                "text": "RAG 会先检索相关文档，再让模型生成答案。",
                "source": "rag.md",
                "chunk_id": "rag.md#chunk-0",
                "score": 0.95,
            },
            {
                "text": "检索结果会作为模型的上下文。",
                "source": "notes.txt",
                "chunk_id": "notes.txt#chunk-1",
                "score": 0.80,
            },
        ]

        prompt = build_prompt("RAG 如何工作？", chunks)

        instruction_position = prompt.index(SYSTEM_INSTRUCTION)
        context_position = prompt.index("context:")
        question_position = prompt.index("question:")
        self.assertLess(instruction_position, context_position)
        self.assertLess(context_position, question_position)
        self.assertIn("优先根据提供的 context 回答问题", prompt)
        self.assertIn("context 中没有足够信息", prompt)
        self.assertIn("不要编造 context 中不存在的信息", prompt)
        self.assertIn("[Chunk 1]", prompt)
        self.assertIn("来源：rag.md", prompt)
        self.assertIn("编号：notes.txt#chunk-1", prompt)
        self.assertIn("检索结果会作为模型的上下文。", prompt)
        self.assertNotIn("0.95", prompt)
        self.assertTrue(prompt.endswith("question:\nRAG 如何工作？"))

    def test_marks_context_as_empty_when_no_chunks_are_retrieved(self) -> None:
        prompt = build_prompt("知识库之外的问题", [])

        self.assertIn("context:\n（没有检索到相关内容）", prompt)
        self.assertIn("根据提供的 context 无法回答该问题", prompt)

    def test_rejects_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt("   ", [])

    def test_rejects_chunk_without_text(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt("RAG 是什么？", [{"source": "rag.md"}])


if __name__ == "__main__":
    unittest.main()
