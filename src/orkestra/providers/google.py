"""Google Gemini provider backend."""

from typing import Iterator

from orkestra.providers.base import ProviderBackend

try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai.errors import ClientError
except ImportError:
    raise ImportError(
        "google-genai is required for the Google provider. "
        "Install it with: pip install orkestra[google]"
    )


class GoogleBackend(ProviderBackend):
    """Backend for Google Gemini models."""

    name = "google"

    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    def call(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        try:
            result = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                raise RuntimeError(
                    f"Quota exhausted for model {model}. "
                    "Check your API key and billing status."
                ) from e
            raise

        return {
            "text": result.text or "",
            "input_tokens": result.usage_metadata.prompt_token_count or 0,
            "output_tokens": result.usage_metadata.candidates_token_count or 0,
        }

    def stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        response = self._client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
