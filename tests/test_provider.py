"""Tests for provider.py."""

import pytest
from unittest.mock import patch, MagicMock

from orkestra.provider import Provider
from orkestra._types import Response


@pytest.fixture
def mock_provider():
    """Create a Provider with mocked backend and router."""
    with (
        patch("orkestra.provider.create_backend") as mock_backend_fn,
        patch("orkestra.provider.KNNRouter") as mock_router_cls,
    ):
        backend = MagicMock()
        backend.name = "google"
        mock_backend_fn.return_value = backend

        router = MagicMock()
        router.route.return_value = "gemini-3-flash-preview"
        mock_router_cls.return_value = router

        backend.call.return_value = {
            "text": "Hello, world!",
            "input_tokens": 10,
            "output_tokens": 50,
        }

        provider = Provider("google", "fake-key")
        yield provider, backend, router


class TestProviderInit:
    def test_creates_backend_and_router(self, mock_provider):
        provider, backend, router = mock_provider
        assert provider.name == "google"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            Provider("nonexistent", "key")


class TestChat:
    def test_returns_response(self, mock_provider):
        provider, backend, router = mock_provider
        response = provider.chat("Hello")
        assert isinstance(response, Response)
        assert response.text == "Hello, world!"
        assert response.model == "gemini-3-flash-preview"
        assert response.provider == "google"

    def test_calls_router_then_backend(self, mock_provider):
        provider, backend, router = mock_provider
        provider.chat("Hello")
        router.route.assert_called_once_with("Hello")
        backend.call.assert_called_once()

    def test_cost_calculation(self, mock_provider):
        provider, backend, router = mock_provider
        response = provider.chat("Hello")
        # gemini-3-flash-preview: input=0.50, output=3.00
        expected = (10 * 0.50 + 50 * 3.00) / 1_000_000
        assert response.cost == pytest.approx(expected)

    def test_savings_vs_base_model(self, mock_provider):
        provider, backend, router = mock_provider
        response = provider.chat("Hello")
        # base model is gemini-3-pro-preview: input=2.00, output=12.00
        base_cost = (10 * 2.00 + 50 * 12.00) / 1_000_000
        assert response.base_cost == pytest.approx(base_cost)
        assert response.savings > 0


class TestStreamText:
    def test_yields_chunks(self, mock_provider):
        provider, backend, router = mock_provider
        backend.stream.return_value = iter(["Hello", " world"])
        chunks = list(provider.stream_text("Hi"))
        assert chunks == ["Hello", " world"]

    def test_routes_before_streaming(self, mock_provider):
        provider, backend, router = mock_provider
        backend.stream.return_value = iter([])
        list(provider.stream_text("Hi"))
        router.route.assert_called_once_with("Hi")
