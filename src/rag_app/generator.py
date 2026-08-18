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
            raise ValueError("model_name must not be empty.")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise TypeError("max_tokens must be an integer.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")

        self.model_name = model_name
        self.max_tokens = max_tokens

        try:
            self.client = client if client is not None else Anthropic()
        except AnthropicError as exc:
            raise GenerationError("Failed to initialize the Anthropic client.") from exc

    def generate(self, prompt: str) -> str:
        """Send one prompt to Anthropic and return its text answer."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except AnthropicError as exc:
            raise GenerationError("Failed to generate an answer through the Anthropic API.") from exc

        try:
            text_parts = [
                block.text for block in message.content if block.type == "text"
            ]
            answer = "".join(text_parts).strip()
        except (AttributeError, TypeError) as exc:
            raise GenerationError("The Anthropic API returned an invalid response.") from exc

        if not answer:
            raise GenerationError("The Anthropic API did not return any text answer.")

        return answer
