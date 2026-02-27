"""Anthropic Claude provider backend."""

from typing import Iterator

from orkestra.providers.base import ProviderBackend

try:
    import anthropic
except ImportError:
    anthropic = None


class AnthropicBackend(ProviderBackend):
    """Backend for Anthropic Claude models."""

    name = "anthropic"

    def __init__(self, api_key: str):
        if anthropic is None:
            raise ImportError(
                "anthropic is required for the Anthropic provider. "
                "Install it with: pip install orkestra[anthropic]"
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def call(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        message = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in message.content if block.type == "text"
        )
        return {
            "text": text,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }

    def stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
