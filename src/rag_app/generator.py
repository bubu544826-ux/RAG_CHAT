"""Generate an answer from a prompt with the Anthropic SDK."""

from anthropic import Anthropic, AnthropicError

from .config import ANTHROPIC_MODEL_NAME


DEFAULT_MAX_TOKENS = 1024


class GenerationError(RuntimeError):
    """Raised when the LLM cannot generate a text answer."""


class Generator:
    """Small wrapper around Anthropic's Messages API."""

    def __init__(
        self,
        model_name: str = ANTHROPIC_MODEL_NAME,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Anthropic | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name 不能为空。")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise TypeError("max_tokens 必须是整数。")
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0。")

        self.model_name = model_name
        self.max_tokens = max_tokens

        try:
            self.client = client if client is not None else Anthropic()
        except AnthropicError as exc:
            raise GenerationError("无法初始化 Anthropic 客户端。") from exc

    def generate(self, prompt: str) -> str:
        """Send one prompt to Anthropic and return its text answer."""
        if not isinstance(prompt, str):
            raise TypeError("prompt 必须是字符串。")
        if not prompt.strip():
            raise ValueError("prompt 不能为空。")

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except AnthropicError as exc:
            raise GenerationError("调用 Anthropic API 生成回答失败。") from exc

        try:
            text_parts = [
                block.text for block in message.content if block.type == "text"
            ]
            answer = "".join(text_parts).strip()
        except (AttributeError, TypeError) as exc:
            raise GenerationError("Anthropic API 返回了无效的响应。") from exc

        if not answer:
            raise GenerationError("Anthropic API 没有返回文本回答。")

        return answer
