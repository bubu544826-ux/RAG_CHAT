"""Tests for generating answers with Anthropic."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from anthropic import AnthropicError

from src.rag_app.config import ANTHROPIC_MODEL_NAME
from src.rag_app.generator import GenerationError, Generator


class GeneratorTest(unittest.TestCase):
    def test_uses_model_name_from_configuration_by_default(self) -> None:
        generator = Generator(client=Mock())

        self.assertEqual(generator.model_name, ANTHROPIC_MODEL_NAME)

    def test_sends_prompt_and_returns_text_answer(self) -> None:
        client = Mock()
        client.messages.create.return_value = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="RAG 先检索，"),
                SimpleNamespace(type="text", text="再生成回答。"),
            ]
        )
        generator = Generator(
            model_name="test-claude-model",
            max_tokens=256,
            client=client,
        )

        answer = generator.generate("请解释 RAG。")

        self.assertEqual(answer, "RAG 先检索，再生成回答。")
        client.messages.create.assert_called_once_with(
            model="test-claude-model",
            max_tokens=256,
            messages=[{"role": "user", "content": "请解释 RAG。"}],
        )

    def test_rejects_empty_prompt_without_calling_api(self) -> None:
        client = Mock()
        generator = Generator(client=client)

        with self.assertRaises(ValueError):
            generator.generate("   ")

        client.messages.create.assert_not_called()

    def test_wraps_anthropic_errors(self) -> None:
        client = Mock()
        api_error = AnthropicError("API unavailable")
        client.messages.create.side_effect = api_error
        generator = Generator(client=client)

        with self.assertRaises(GenerationError) as context:
            generator.generate("什么是 RAG？")

        self.assertIs(context.exception.__cause__, api_error)

    def test_rejects_response_without_text(self) -> None:
        client = Mock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="tool_use")]
        )
        generator = Generator(client=client)

        with self.assertRaises(GenerationError):
            generator.generate("什么是 RAG？")


if __name__ == "__main__":
    unittest.main()
