"""Tests for registry/models.py."""

import pytest

from orkestra.registry.models import (
    PROVIDER_MODELS,
    DEFAULT_BASE_MODELS,
    TIER_RANK,
    get_models,
    calculate_cost,
)


class TestProviderModels:
    def test_all_providers_have_models(self):
        for provider in ("google", "anthropic", "openai"):
            assert provider in PROVIDER_MODELS
            assert len(PROVIDER_MODELS[provider]) >= 3

    def test_all_models_have_required_fields(self):
        required = {"input_price", "output_price", "context_window", "tier"}
        for provider, models in PROVIDER_MODELS.items():
            for model_name, info in models.items():
                missing = required - set(info.keys())
                assert not missing, f"{provider}/{model_name} missing: {missing}"

    def test_all_tiers_are_valid(self):
        for provider, models in PROVIDER_MODELS.items():
            for model_name, info in models.items():
                assert info["tier"] in TIER_RANK, (
                    f"{provider}/{model_name} has invalid tier: {info['tier']}"
                )

    def test_pricing_is_positive(self):
        for provider, models in PROVIDER_MODELS.items():
            for model_name, info in models.items():
                assert info["input_price"] > 0
                assert info["output_price"] > 0

    def test_each_provider_has_all_tiers(self):
        for provider, models in PROVIDER_MODELS.items():
            tiers = {info["tier"] for info in models.values()}
            assert tiers == {"budget", "balanced", "premium"}, (
                f"{provider} missing tiers: {{'budget', 'balanced', 'premium'}} - {tiers}"
            )


class TestDefaultBaseModels:
    def test_all_providers_have_default(self):
        for provider in PROVIDER_MODELS:
            assert provider in DEFAULT_BASE_MODELS

    def test_defaults_are_valid_models(self):
        for provider, model in DEFAULT_BASE_MODELS.items():
            assert model in PROVIDER_MODELS[provider]


class TestGetModels:
    def test_returns_models(self):
        models = get_models("google")
        assert "gemini-2.5-flash-lite" in models

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="No model definitions"):
            get_models("nonexistent")


class TestCalculateCost:
    def test_google_flash_lite(self):
        cost = calculate_cost("google", "gemini-2.5-flash-lite", 1000, 500)
        expected = (1000 * 0.10 + 500 * 0.40) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_google_pro(self):
        cost = calculate_cost("google", "gemini-3-pro-preview", 1000, 500)
        expected = (1000 * 2.00 + 500 * 12.00) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_zero_tokens(self):
        cost = calculate_cost("google", "gemini-2.5-flash-lite", 0, 0)
        assert cost == 0.0

    def test_anthropic_cost(self):
        cost = calculate_cost("anthropic", "claude-haiku-4", 1000, 500)
        expected = (1000 * 0.80 + 500 * 4.00) / 1_000_000
        assert cost == pytest.approx(expected)
