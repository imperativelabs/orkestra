"""Main Provider class for single-provider LLM routing."""

from __future__ import annotations

from typing import Iterator

from orkestra._types import Response
from orkestra.providers import create_backend
from orkestra.providers.base import ProviderBackend
from orkestra.registry.models import (
    PROVIDER_MODELS,
    DEFAULT_BASE_MODELS,
    DEFAULT_FALLBACK_MODELS,
    calculate_cost,
)


class Provider:
    """Routes prompts to the optimal model for a single LLM provider.

    Usage:
        import orkestra as o

        provider = o.Provider("google", "YOUR_API_KEY")
        response = provider.chat("Explain quantum computing")
        print(response.text)
        print(f"Model: {response.model}, Cost: ${response.cost:.6f}")

        # Disable smart routing and use a fixed model
        provider = o.Provider("google", "YOUR_API_KEY", smart_routing=False)
        response = provider.chat("Hello")  # uses gemini-3-flash-preview

        # Disable smart routing with a custom default model
        provider = o.Provider("google", "YOUR_API_KEY", smart_routing=False, default_model="gemini-2.5-flash-lite")
        response = provider.chat("Hello", model="gemini-3-pro-preview")  # per-call override

    Supported providers: google, anthropic, openai.
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        smart_routing: bool = True,
        default_model: str | None = None,
    ):
        """Initialize a provider with optional smart routing.

        Args:
            name: Provider name ("google", "anthropic", "openai").
            api_key: API key for the provider.
            smart_routing: If True (default), uses KNN routing to pick the best model
                per prompt. If False, uses a fixed default model.
            default_model: Fixed model to use when smart_routing=False. If not
                provided, falls back to the balanced-tier model for the provider.
                Cannot be an empty string when smart_routing=False.

        Raises:
            ValueError: If smart_routing=False and default_model is an empty string.
        """
        if not smart_routing and default_model is not None and default_model == "":
            raise ValueError("default_model cannot be an empty string when smart_routing=False")

        self._backend: ProviderBackend = create_backend(name, api_key)
        self._smart_routing = smart_routing
        self._base_model = DEFAULT_BASE_MODELS.get(name)

        if smart_routing:
            from orkestra.router.knn import KNNRouter
            self._router = KNNRouter(provider=name)
            self._default_model: str | None = None
        else:
            self._router = None
            self._default_model = default_model if default_model is not None else DEFAULT_FALLBACK_MODELS.get(name)

    @property
    def name(self) -> str:
        return self._backend.name

    def _resolve_model(self, model: str | None, prompt: str) -> str:
        """Resolve which model to use for a request."""
        if not self._smart_routing:
            if model is not None and model == "":
                raise ValueError("model cannot be an empty string when smart_routing=False")
            return model if model is not None else self._default_model
        return self._router.route(prompt)

    def chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 1.0,
    ) -> Response:
        """Generate a response using the optimally-routed model.

        Args:
            prompt: The input prompt.
            model: Override the model to use. Only applied when smart_routing=False;
                cannot be an empty string in that case.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            A Response with the generated text, cost, and savings info.
        """
        resolved_model = self._resolve_model(model, prompt)
        result = self._backend.call(resolved_model, prompt, max_tokens, temperature)

        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        cost = calculate_cost(self.name, resolved_model, input_tokens, output_tokens)

        model_info = PROVIDER_MODELS[self.name][resolved_model]
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
            model=resolved_model,
            provider=self.name,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            savings=savings,
            savings_percent=savings_percent,
            base_model=base_model or resolved_model,
            base_cost=base_cost,
        )

    def stream_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 1.0,
    ) -> Iterator[str]:
        """Stream text from the optimally-routed model.

        Args:
            prompt: The input prompt.
            model: Override the model to use. Only applied when smart_routing=False;
                cannot be an empty string in that case.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Yields:
            Text chunks as they arrive.
        """
        resolved_model = self._resolve_model(model, prompt)
        yield from self._backend.stream(resolved_model, prompt, max_tokens, temperature)
