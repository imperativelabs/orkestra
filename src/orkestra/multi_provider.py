"""MultiProvider for strategy-based routing across multiple providers."""

from __future__ import annotations

from typing import Iterator

from orkestra._types import Response
from orkestra.registry.strategies import STRATEGIES
from orkestra.registry.models import PROVIDER_MODELS, calculate_cost


class MultiProvider:
    """Combines multiple providers and selects the best one per query.

    Usage:
        import orkestra as o

        google = o.Provider("google", "GOOGLE_KEY")
        anthropic = o.Provider("anthropic", "ANTHROPIC_KEY")

        multi = o.MultiProvider([google, anthropic])
        response = multi.chat("Explain quantum computing", strategy="cheapest")
    """

    def __init__(self, providers: list):
        """Initialize with a list of Provider instances.

        Args:
            providers: List of Provider instances.

        Raises:
            ValueError: If providers list is empty.
        """
        if not providers:
            raise ValueError("At least one provider is required.")
        self._providers = providers

    def chat(
        self,
        prompt: str,
        *,
        strategy: str = "cheapest",
        max_tokens: int = 8192,
        temperature: float = 1.0,
    ) -> Response:
        """Generate a response using strategy-based provider selection.

        Each provider's router picks its optimal model, then the strategy
        selects the best provider+model combination.

        Args:
            prompt: The input prompt.
            strategy: Selection strategy ("cheapest", "smartest", "balanced").
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            A Response from the selected provider.

        Raises:
            ValueError: If strategy is not recognized.
        """
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {sorted(STRATEGIES.keys())}"
            )

        # Ask each provider's router for its model selection
        selections = {}
        for provider in self._providers:
            model = provider._router.route(prompt)
            selections[provider] = model

        # Apply strategy
        strategy_fn = STRATEGIES[strategy]
        winner, model = strategy_fn(self._providers, selections)

        # Call the winning provider's backend
        result = winner._backend.call(model, prompt, max_tokens, temperature)

        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        cost = calculate_cost(winner.name, model, input_tokens, output_tokens)

        model_info = PROVIDER_MODELS[winner.name][model]
        input_cost = (input_tokens * model_info["input_price"]) / 1_000_000
        output_cost = (output_tokens * model_info["output_price"]) / 1_000_000

        return Response(
            text=result["text"],
            model=model,
            provider=winner.name,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            savings=0.0,
            savings_percent=0.0,
            base_model=model,
            base_cost=cost,
        )

    def stream_text(
        self,
        prompt: str,
        *,
        strategy: str = "cheapest",
        max_tokens: int = 8192,
        temperature: float = 1.0,
    ) -> Iterator[str]:
        """Stream text using strategy-based provider selection.

        Args:
            prompt: The input prompt.
            strategy: Selection strategy ("cheapest", "smartest", "balanced").
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Yields:
            Text chunks as they arrive.
        """
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {sorted(STRATEGIES.keys())}"
            )

        selections = {}
        for provider in self._providers:
            model = provider._router.route(prompt)
            selections[provider] = model

        strategy_fn = STRATEGIES[strategy]
        winner, model = strategy_fn(self._providers, selections)

        yield from winner._backend.stream(model, prompt, max_tokens, temperature)
