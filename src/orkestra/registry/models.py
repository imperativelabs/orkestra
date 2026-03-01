"""Per-provider model definitions and pricing."""

PROVIDER_MODELS: dict[str, dict[str, dict]] = {
    "google": {
        "gemini-2.5-flash-lite": {
            "input_price": 0.10,
            "output_price": 0.40,
            "context_window": 1_048_576,
            "tier": "budget",
        },
        "gemini-3-flash-preview": {
            "input_price": 0.50,
            "output_price": 3.00,
            "context_window": 1_048_576,
            "tier": "balanced",
        },
        "gemini-3-pro-preview": {
            "input_price": 2.00,
            "output_price": 12.00,
            "context_window": 1_048_576,
            "tier": "premium",
        },
    },
    "anthropic": {
        "claude-haiku-4": {
            "input_price": 0.80,
            "output_price": 4.00,
            "context_window": 200_000,
            "tier": "budget",
        },
        "claude-sonnet-4-5": {
            "input_price": 3.00,
            "output_price": 15.00,
            "context_window": 200_000,
            "tier": "balanced",
        },
        "claude-opus-4": {
            "input_price": 15.00,
            "output_price": 75.00,
            "context_window": 200_000,
            "tier": "premium",
        },
    },
    "openai": {
        "gpt-4o-mini": {
            "input_price": 0.15,
            "output_price": 0.60,
            "context_window": 128_000,
            "tier": "budget",
        },
        "gpt-4o": {
            "input_price": 2.50,
            "output_price": 10.00,
            "context_window": 128_000,
            "tier": "balanced",
        },
        "o3": {
            "input_price": 10.00,
            "output_price": 40.00,
            "context_window": 200_000,
            "tier": "premium",
        },
    },
}

# Default base model per provider (used for savings comparison)
DEFAULT_BASE_MODELS: dict[str, str] = {
    "google": "gemini-3-pro-preview",
    "anthropic": "claude-opus-4",
    "openai": "o3",
}

# Fallback model per provider when smart_routing=False and no default_model is given
DEFAULT_FALLBACK_MODELS: dict[str, str] = {
    "google": "gemini-3-flash-preview",
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o",
}

# Tier ordering for strategy comparisons
TIER_RANK = {"budget": 0, "balanced": 1, "premium": 2}


def get_models(provider: str) -> dict[str, dict]:
    """Get model definitions for a provider.

    Raises:
        ValueError: If provider has no model definitions.
    """
    if provider not in PROVIDER_MODELS:
        raise ValueError(
            f"No model definitions for '{provider}'. "
            f"Available: {sorted(PROVIDER_MODELS.keys())}"
        )
    return PROVIDER_MODELS[provider]


def calculate_cost(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> float:
    """Calculate cost in dollars for a given provider, model, and token counts."""
    models = PROVIDER_MODELS[provider]
    info = models[model]
    return (
        input_tokens * info["input_price"] + output_tokens * info["output_price"]
    ) / 1_000_000
