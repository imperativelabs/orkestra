"""Tests for multi_provider.py."""

import pytest
from unittest.mock import patch, MagicMock

from orkestra.multi_provider import MultiProvider
from orkestra._types import Response


def _make_mock_provider(name, routed_model):
    p = MagicMock()
    p.name = name
    p._backend = MagicMock()
    p._backend.name = name
    p._router = MagicMock()
    p._router.route.return_value = routed_model
    p._backend.call.return_value = {
        "text": f"Response from {name}",
        "input_tokens": 10,
        "output_tokens": 50,
    }
    p._backend.stream.return_value = iter([f"chunk from {name}"])
    return p


class TestMultiProviderInit:
    def test_requires_providers(self):
        with pytest.raises(ValueError, match="At least one"):
            MultiProvider([])

    def test_accepts_providers(self):
        p = _make_mock_provider("google", "gemini-3-flash-preview")
        multi = MultiProvider([p])
        assert len(multi._providers) == 1


class TestMultiProviderChat:
    def test_cheapest_strategy(self):
        google = _make_mock_provider("google", "gemini-2.5-flash-lite")
        openai = _make_mock_provider("openai", "gpt-4o-mini")
        multi = MultiProvider([google, openai])

        response = multi.chat("Hello", strategy="cheapest")
        assert isinstance(response, Response)
        # google flash-lite is cheapest (0.10 + 0.40 = 0.50 vs 0.15 + 0.60 = 0.75)
        assert response.provider == "google"

    def test_smartest_strategy(self):
        google = _make_mock_provider("google", "gemini-3-pro-preview")
        openai = _make_mock_provider("openai", "gpt-4o-mini")
        multi = MultiProvider([google, openai])

        response = multi.chat("Hello", strategy="smartest")
        assert response.provider == "google"  # premium beats budget

    def test_invalid_strategy_raises(self):
        p = _make_mock_provider("google", "gemini-3-flash-preview")
        multi = MultiProvider([p])
        with pytest.raises(ValueError, match="Unknown strategy"):
            multi.chat("Hello", strategy="nonexistent")

    def test_routes_all_providers(self):
        google = _make_mock_provider("google", "gemini-3-flash-preview")
        openai = _make_mock_provider("openai", "gpt-4o")
        multi = MultiProvider([google, openai])

        multi.chat("Hello", strategy="cheapest")
        google._router.route.assert_called_once_with("Hello")
        openai._router.route.assert_called_once_with("Hello")


class TestMultiProviderStream:
    def test_streams_from_winner(self):
        google = _make_mock_provider("google", "gemini-2.5-flash-lite")
        multi = MultiProvider([google])

        chunks = list(multi.stream_text("Hello", strategy="cheapest"))
        assert chunks == ["chunk from google"]

    def test_invalid_strategy_raises(self):
        p = _make_mock_provider("google", "gemini-3-flash-preview")
        multi = MultiProvider([p])
        with pytest.raises(ValueError, match="Unknown strategy"):
            list(multi.stream_text("Hello", strategy="bad"))
