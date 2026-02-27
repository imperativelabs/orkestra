"""Tests for registry/strategies.py."""

from unittest.mock import MagicMock

from orkestra.registry.strategies import STRATEGIES, balanced, cheapest, smartest


def _make_provider(name: str):
    p = MagicMock()
    p.name = name
    p._backend = MagicMock()
    p._backend.name = name
    return p


class TestCheapest:
    def test_picks_cheapest_provider(self):
        google = _make_provider("google")
        openai = _make_provider("openai")
        selections = {
            google: "gemini-2.5-flash-lite",  # 0.10 + 0.40 = 0.50
            openai: "gpt-4o-mini",            # 0.15 + 0.60 = 0.75
        }
        winner, model = cheapest([google, openai], selections) #type:ignore
        assert winner is google
        assert model == "gemini-2.5-flash-lite"

    def test_picks_cheapest_model(self):
        google = _make_provider("google")
        selections = {google: "gemini-2.5-flash-lite"}
        winner, model = cheapest([google], selections) #type:ignore
        assert model == "gemini-2.5-flash-lite"


class TestSmartest:
    def test_picks_premium_tier(self):
        google = _make_provider("google")
        openai = _make_provider("openai")
        selections = {
            google: "gemini-3-pro-preview",  # premium
            openai: "gpt-4o-mini",           # budget
        }
        winner, model = smartest([google, openai], selections) #type:ignore
        assert winner is google
        assert model == "gemini-3-pro-preview"

    def test_same_tier_picks_cheaper(self):
        google = _make_provider("google")
        openai = _make_provider("openai")
        selections = {
            google: "gemini-3-flash-preview",  # balanced, 0.50 + 3.00
            openai: "gpt-4o",                  # balanced, 2.50 + 10.00
        }
        winner, model = smartest([google, openai], selections) #type:ignore
        assert winner is google


class TestBalanced:
    def test_prefers_balanced_tier(self):
        google = _make_provider("google")
        openai = _make_provider("openai")
        selections = {
            google: "gemini-3-flash-preview",  # balanced
            openai: "gpt-4o-mini",             # budget
        }
        winner, model = balanced([google, openai], selections) #type:ignore
        assert winner is google
        assert model == "gemini-3-flash-preview"

    def test_falls_back_to_cheapest_without_balanced(self):
        google = _make_provider("google")
        openai = _make_provider("openai")
        selections = {
            google: "gemini-2.5-flash-lite",  # budget, 0.50
            openai: "o3",                      # premium, 50.00
        }
        winner, model = balanced([google, openai], selections) #type:ignore
        assert winner is google
        assert model == "gemini-2.5-flash-lite"


class TestStrategyRegistry:
    def test_all_strategies_registered(self):
        assert set(STRATEGIES.keys()) == {"cheapest", "smartest", "balanced"}
