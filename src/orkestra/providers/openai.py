"""OpenAI provider backend."""

from typing import Iterator

from orkestra.providers.base import ProviderBackend

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAIBackend(ProviderBackend):
    """Backend for OpenAI models."""

    name = "openai"

    def __init__(self, api_key: str):
        if OpenAI is None:
            raise ImportError(
                "openai is required for the OpenAI provider. "
                "Install it with: pip install orkestra[openai]"
            )
        self._client = OpenAI(api_key=api_key)

    def call(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = response.choices[0]
        return {
            "text": choice.message.content or "",
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    def stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
