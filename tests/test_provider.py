"""Tests for provider.py."""

import pytest
from unittest.mock import patch, MagicMock

from orkestra.provider import Provider
from orkestra._types import Response


@pytest.fixture
def mock_provider():
    """Create a Provider with mocked backend and router (smart_routing=True)."""
    import sys

    router = MagicMock()
    router.route.return_value = "gemini-3-flash-preview"
    mock_knn_module = MagicMock()
    mock_knn_module.KNNRouter = MagicMock(return_value=router)

    with (
        patch("orkestra.provider.create_backend") as mock_backend_fn,
        patch.dict(sys.modules, {"orkestra.router.knn": mock_knn_module}),
    ):
        backend = MagicMock()
        backend.name = "google"
        mock_backend_fn.return_value = backend

        backend.call.return_value = {
            "text": "Hello, world!",
            "input_tokens": 10,
            "output_tokens": 50,
        }

        provider = Provider("google", "fake-key")
        yield provider, backend, router


@pytest.fixture
def mock_provider_no_routing():
    """Create a Provider with smart_routing=False (no KNNRouter instantiated)."""
    with patch("orkestra.provider.create_backend") as mock_backend_fn:
        backend = MagicMock()
        backend.name = "anthropic"
        mock_backend_fn.return_value = backend

        backend.call.return_value = {
            "text": "Hello, world!",
            "input_tokens": 10,
            "output_tokens": 50,
        }

        yield backend


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


class TestSmartRoutingDisabled:
    def test_uses_fallback_model_when_no_default_given(self, mock_provider_no_routing):
        backend = mock_provider_no_routing
        provider = Provider("anthropic", "fake-key", smart_routing=False)
        provider.chat("Hello")
        backend.call.assert_called_once()
        called_model = backend.call.call_args[0][0]
        assert called_model == "claude-sonnet-4-5"

    def test_uses_explicit_default_model(self, mock_provider_no_routing):
        backend = mock_provider_no_routing
        provider = Provider("anthropic", "fake-key", smart_routing=False, default_model="claude-haiku-4")
        provider.chat("Hello")
        called_model = backend.call.call_args[0][0]
        assert called_model == "claude-haiku-4"

    def test_per_call_model_override(self, mock_provider_no_routing):
        backend = mock_provider_no_routing
        provider = Provider("anthropic", "fake-key", smart_routing=False)
        provider.chat("Hello", model="claude-opus-4")
        called_model = backend.call.call_args[0][0]
        assert called_model == "claude-opus-4"

    def test_router_is_none(self, mock_provider_no_routing):
        provider = Provider("anthropic", "fake-key", smart_routing=False)
        assert provider._router is None

    def test_router_modules_not_imported_when_disabled(self, mock_provider_no_routing):
        import sys
        # Save and remove cached router modules to detect fresh imports
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if "orkestra.router" in k}
        try:
            Provider("anthropic", "fake-key", smart_routing=False)
            assert "orkestra.router.knn" not in sys.modules
            assert "orkestra.router.cache" not in sys.modules
            assert "orkestra.router.embedder" not in sys.modules
        finally:
            sys.modules.update(saved)

    def test_empty_default_model_raises(self):
        with patch("orkestra.provider.create_backend") as mock_backend_fn:
            backend = MagicMock()
            backend.name = "anthropic"
            mock_backend_fn.return_value = backend
            with pytest.raises(ValueError, match="default_model cannot be an empty string"):
                Provider("anthropic", "fake-key", smart_routing=False, default_model="")

    def test_empty_per_call_model_raises(self, mock_provider_no_routing):
        provider = Provider("anthropic", "fake-key", smart_routing=False)
        with pytest.raises(ValueError, match="model cannot be an empty string"):
            provider.chat("Hello", model="")

    def test_stream_uses_fallback_model(self, mock_provider_no_routing):
        backend = mock_provider_no_routing
        backend.stream.return_value = iter(["chunk"])
        provider = Provider("anthropic", "fake-key", smart_routing=False)
        list(provider.stream_text("Hi"))
        called_model = backend.stream.call_args[0][0]
        assert called_model == "claude-sonnet-4-5"

    def test_stream_per_call_model_override(self, mock_provider_no_routing):
        backend = mock_provider_no_routing
        backend.stream.return_value = iter(["chunk"])
        provider = Provider("anthropic", "fake-key", smart_routing=False)
        list(provider.stream_text("Hi", model="claude-haiku-4"))
        called_model = backend.stream.call_args[0][0]
        assert called_model == "claude-haiku-4"

    def test_stream_empty_model_raises(self, mock_provider_no_routing):
        backend = mock_provider_no_routing
        backend.stream.return_value = iter([])
        provider = Provider("anthropic", "fake-key", smart_routing=False)
        with pytest.raises(ValueError, match="model cannot be an empty string"):
            list(provider.stream_text("Hi", model=""))

    def test_google_fallback_model(self, mock_provider_no_routing):
        with patch("orkestra.provider.create_backend") as mock_backend_fn:
            backend = MagicMock()
            backend.name = "google"
            mock_backend_fn.return_value = backend
            backend.call.return_value = {"text": "hi", "input_tokens": 5, "output_tokens": 5}
            provider = Provider("google", "fake-key", smart_routing=False)
            provider.chat("Hello")
            called_model = backend.call.call_args[0][0]
            assert called_model == "gemini-3-flash-preview"

    def test_openai_fallback_model(self):
        with patch("orkestra.provider.create_backend") as mock_backend_fn:
            backend = MagicMock()
            backend.name = "openai"
            mock_backend_fn.return_value = backend
            backend.call.return_value = {"text": "hi", "input_tokens": 5, "output_tokens": 5}
            provider = Provider("openai", "fake-key", smart_routing=False)
            provider.chat("Hello")
            called_model = backend.call.call_args[0][0]
            assert called_model == "gpt-4o"
