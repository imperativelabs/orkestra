"""Main Provider class for single-provider LLM routing."""

from __future__ import annotations

from typing import Iterator

from orkestra._types import Response
from orkestra.providers import create_backend
from orkestra.providers.base import ProviderBackend
from orkestra.registry.models import (
    PROVIDER_MODELS,
    DEFAULT_BASE_MODELS,
    calculate_cost,
)
from orkestra.router.knn import KNNRouter


class Provider:
    """Routes prompts to the optimal model for a single LLM provider.

    Usage:
        import orkestra as o

        provider = o.Provider("google", "YOUR_API_KEY")
        response = provider.chat("Explain quantum computing")
        print(response.text)
        print(f"Model: {response.model}, Cost: ${response.cost:.6f}")

    Supported providers: google, anthropic, openai.
    """

    def __init__(self, name: str, api_key: str):
        """Initialize a provider with routing.

        Args:
            name: Provider name ("google", "anthropic", "openai").
            api_key: API key for the provider.
        """
        self._backend: ProviderBackend = create_backend(name, api_key)
        self._router = KNNRouter(provider=name)
        self._base_model = DEFAULT_BASE_MODELS.get(name)

    @property
    def name(self) -> str:
        return self._backend.name

    def chat(
        self,
        prompt: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 1.0,
    ) -> Response:
        """Generate a response using the optimally-routed model.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            A Response with the generated text, cost, and savings info.
        """
        model = self._router.route(prompt)
        result = self._backend.call(model, prompt, max_tokens, temperature)

        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        cost = calculate_cost(self.name, model, input_tokens, output_tokens)

        model_info = PROVIDER_MODELS[self.name][model]
        input_cost = (input_tokens * model_info["input_price"]) / 1_000_000
        output_cost = (output_tokens * model_info["output_price"]) / 1_000_000

        base_model = self._base_model
        base_cost = 0.0
        savings = 0.0
        savings_percent = 0.0

        if base_model and base_model in PROVIDER_MODELS[self.name]:
            base_cost = calculate_cost(self.name, base_model, input_tokens, output_tokens)
            savings = base_cost - cost
            savings_percent = (savings / base_cost * 100) if base_cost > 0 else 0.0

        return Response(
            text=result["text"],
            model=model,
            provider=self.name,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            savings=savings,
            savings_percent=savings_percent,
            base_model=base_model or model,
            base_cost=base_cost,
        )

    def stream_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 1.0,
    ) -> Iterator[str]:
        """Stream text from the optimally-routed model.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Yields:
            Text chunks as they arrive.
        """
        model = self._router.route(prompt)
        yield from self._backend.stream(model, prompt, max_tokens, temperature)
