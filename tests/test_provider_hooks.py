"""Integration tests for Provider event and middleware hooks."""

import sys
import pytest
from unittest.mock import MagicMock, patch

from orkestra.provider import Provider
from orkestra._events import (
    EventData,
    _global_bus,
    emit_event,
    register_event,
    ON_REQUEST,
    ON_CHAT,
    ON_STREAM,
    ON_ROUTE,
    ON_RESPONSE,
    ON_CHUNK,
    ON_STREAM_COMPLETE,
)
from orkestra._middleware import (
    MiddlewareData,
    _global_middlewares,
    register_middleware,
)


# ---------------------------------------------------------------------------
# Isolation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_globals():
    """Clear global event bus and middleware list before/after each test."""
    orig_handlers = dict(_global_bus._handlers)
    orig_middlewares = list(_global_middlewares)

    _global_bus._handlers.clear()
    _global_middlewares.clear()

    yield

    _global_bus._handlers.clear()
    _global_bus._handlers.update(orig_handlers)
    _global_middlewares.clear()
    _global_middlewares.extend(orig_middlewares)


@pytest.fixture
def provider_ctx():
    """Yields (provider, backend) with mocked backend and router."""
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
            "text": "Hello!",
            "input_tokens": 10,
            "output_tokens": 20,
        }
        backend.stream.return_value = iter(["Hello", " world"])

        p = Provider("google", "fake-key")
        yield p, backend


@pytest.fixture
def provider_ctx_b():
    """A second independent provider (anthropic) for cross-provider tests."""
    with patch("orkestra.provider.create_backend") as mock_backend_fn:
        backend = MagicMock()
        backend.name = "anthropic"
        mock_backend_fn.return_value = backend
        backend.call.return_value = {
            "text": "Hi from anthropic!",
            "input_tokens": 5,
            "output_tokens": 10,
        }
        p = Provider("anthropic", "fake-key", smart_routing=False)
        yield p, backend


# ---------------------------------------------------------------------------
# Provider decorator API
# ---------------------------------------------------------------------------

class TestProviderDecoratorAPI:
    def test_middleware_decorator_returns_fn(self, provider_ctx):
        p, _ = provider_ctx

        def my_mw(data, next): next()

        result = p.middleware(my_mw)
        assert result is my_mw

    def test_middleware_decorator_registers_on_provider(self, provider_ctx):
        p, _ = provider_ctx

        @p.middleware
        def my_mw(data, next): next()

        assert my_mw in p._middlewares

    def test_event_decorator_returns_fn(self, provider_ctx):
        p, _ = provider_ctx

        def my_handler(data: EventData): pass

        result = p.event("on_response")(my_handler)
        assert result is my_handler

    def test_event_decorator_registers_on_provider_bus(self, provider_ctx):
        p, _ = provider_ctx
        log = []

        @p.event("on_test")
        def handler(data: EventData):
            log.append("fired")

        p._event_bus.emit("on_test", EventData(event="on_test", provider="google", prompt="q"))
        assert log == ["fired"]

    def test_multiple_provider_middlewares_registered(self, provider_ctx):
        p, _ = provider_ctx

        @p.middleware
        def mw1(data, next): next()

        @p.middleware
        def mw2(data, next): next()

        assert mw1 in p._middlewares
        assert mw2 in p._middlewares
        assert p._middlewares.index(mw1) < p._middlewares.index(mw2)


# ---------------------------------------------------------------------------
# Global events fired during chat()
# ---------------------------------------------------------------------------

class TestGlobalEventsOnChat:
    def test_on_request_fires(self, provider_ctx):
        p, _ = provider_ctx
        fired = []

        @register_event(ON_REQUEST)
        def handler(data: EventData):
            fired.append(data.event)

        p.chat("hello")
        assert ON_REQUEST in fired

    def test_on_chat_fires(self, provider_ctx):
        p, _ = provider_ctx
        fired = []

        @register_event(ON_CHAT)
        def handler(data: EventData):
            fired.append(data.event)

        p.chat("hello")
        assert ON_CHAT in fired

    def test_on_route_fires_with_resolved_model(self, provider_ctx):
        p, _ = provider_ctx
        routed = []

        @register_event(ON_ROUTE)
        def handler(data: EventData):
            routed.append(data.model)

        p.chat("hello")
        assert routed == ["gemini-3-flash-preview"]

    def test_on_response_fires_with_response(self, provider_ctx):
        p, _ = provider_ctx
        responses = []

        @register_event(ON_RESPONSE)
        def handler(data: EventData):
            responses.append(data.response)

        p.chat("hello")
        assert len(responses) == 1
        assert responses[0].text == "Hello!"

    def test_on_response_has_provider_and_prompt(self, provider_ctx):
        p, _ = provider_ctx
        captured = []

        @register_event(ON_RESPONSE)
        def handler(data: EventData):
            captured.append((data.provider, data.prompt))

        p.chat("my prompt")
        assert captured[0] == ("google", "my prompt")

    def test_on_request_fires_before_on_response(self, provider_ctx):
        p, _ = provider_ctx
        order = []

        @register_event(ON_REQUEST)
        def h1(data: EventData): order.append("request")

        @register_event(ON_RESPONSE)
        def h2(data: EventData): order.append("response")

        p.chat("hello")
        assert order.index("request") < order.index("response")

    def test_on_route_fires_before_on_response(self, provider_ctx):
        p, _ = provider_ctx
        order = []

        @register_event(ON_ROUTE)
        def h1(data: EventData): order.append("route")

        @register_event(ON_RESPONSE)
        def h2(data: EventData): order.append("response")

        p.chat("hello")
        assert order.index("route") < order.index("response")


# ---------------------------------------------------------------------------
# Global events fired during stream_text()
# ---------------------------------------------------------------------------

class TestGlobalEventsOnStream:
    def test_on_request_fires(self, provider_ctx):
        p, backend = provider_ctx
        backend.stream.return_value = iter(["chunk"])
        fired = []

        @register_event(ON_REQUEST)
        def handler(data: EventData): fired.append(True)

        list(p.stream_text("hello"))
        assert fired

    def test_on_stream_fires(self, provider_ctx):
        p, backend = provider_ctx
        backend.stream.return_value = iter(["chunk"])
        fired = []

        @register_event(ON_STREAM)
        def handler(data: EventData): fired.append(True)

        list(p.stream_text("hello"))
        assert fired

    def test_on_chunk_fires_per_chunk(self, provider_ctx):
        p, backend = provider_ctx
        backend.stream.return_value = iter(["a", "b", "c"])
        chunks = []

        @register_event(ON_CHUNK)
        def handler(data: EventData):
            chunks.append(data.metadata["chunk"])

        list(p.stream_text("hello"))
        assert chunks == ["a", "b", "c"]

    def test_on_stream_complete_fires_after_exhaustion(self, provider_ctx):
        p, backend = provider_ctx
        backend.stream.return_value = iter(["x", "y"])
        complete = []

        @register_event(ON_STREAM_COMPLETE)
        def handler(data: EventData): complete.append(True)

        list(p.stream_text("hello"))
        assert complete == [True]

    def test_on_stream_complete_fires_only_once(self, provider_ctx):
        p, backend = provider_ctx
        backend.stream.return_value = iter(["a", "b"])
        count = [0]

        @register_event(ON_STREAM_COMPLETE)
        def handler(data: EventData): count[0] += 1

        list(p.stream_text("hello"))
        assert count[0] == 1

    def test_on_chat_does_not_fire_on_stream(self, provider_ctx):
        p, backend = provider_ctx
        backend.stream.return_value = iter(["x"])
        chat_fired = []

        @register_event(ON_CHAT)
        def handler(data: EventData): chat_fired.append(True)

        list(p.stream_text("hello"))
        assert chat_fired == []

    def test_on_stream_does_not_fire_on_chat(self, provider_ctx):
        p, _ = provider_ctx
        stream_fired = []

        @register_event(ON_STREAM)
        def handler(data: EventData): stream_fired.append(True)

        p.chat("hello")
        assert stream_fired == []

    def test_on_route_fires_with_model_for_stream(self, provider_ctx):
        p, backend = provider_ctx
        backend.stream.return_value = iter(["x"])
        routed = []

        @register_event(ON_ROUTE)
        def handler(data: EventData): routed.append(data.model)

        list(p.stream_text("hello"))
        assert routed == ["gemini-3-flash-preview"]


# ---------------------------------------------------------------------------
# Provider-level events
# ---------------------------------------------------------------------------

class TestProviderLevelEvents:
    def test_provider_event_fires_for_its_own_calls(self, provider_ctx):
        p, _ = provider_ctx
        fired = []

        @p.event(ON_RESPONSE)
        def handler(data: EventData): fired.append(True)

        p.chat("hello")
        assert fired == [True]

    def test_provider_event_does_not_fire_for_other_provider(self, provider_ctx, provider_ctx_b):
        p_google, _ = provider_ctx
        p_anthropic, _ = provider_ctx_b
        fired_by_google_handler = []

        @p_google.event(ON_RESPONSE)
        def handler(data: EventData): fired_by_google_handler.append(True)

        p_anthropic.chat("hello")
        assert fired_by_google_handler == []

    def test_global_and_provider_events_both_fire(self, provider_ctx):
        p, _ = provider_ctx
        log = []

        @register_event(ON_RESPONSE)
        def global_handler(data: EventData): log.append("global")

        @p.event(ON_RESPONSE)
        def provider_handler(data: EventData): log.append("provider")

        p.chat("hello")
        assert "global" in log
        assert "provider" in log

    def test_provider_event_receives_correct_data(self, provider_ctx):
        p, _ = provider_ctx
        captured = []

        @p.event(ON_ROUTE)
        def handler(data: EventData): captured.append(data.model)

        p.chat("hello")
        assert captured == ["gemini-3-flash-preview"]


# ---------------------------------------------------------------------------
# Global middleware on chat()
# ---------------------------------------------------------------------------

class TestGlobalMiddlewareOnChat:
    def test_global_middleware_runs(self, provider_ctx):
        p, _ = provider_ctx
        ran = []

        @register_middleware
        def mw(data: MiddlewareData, next):
            ran.append(True)
            next()

        p.chat("hello")
        assert ran == [True]

    def test_global_middleware_can_mutate_prompt(self, provider_ctx):
        p, backend = provider_ctx

        @register_middleware
        def mw(data: MiddlewareData, next):
            data.prompt = "MODIFIED"
            next()

        p.chat("original")
        called_prompt = backend.call.call_args[0][1]
        assert called_prompt == "MODIFIED"

    def test_global_middleware_can_read_response_after_next(self, provider_ctx):
        p, _ = provider_ctx
        post_responses = []

        @register_middleware
        def mw(data: MiddlewareData, next):
            next()
            post_responses.append(data.response)

        p.chat("hello")
        assert post_responses[0].text == "Hello!"

    def test_multiple_global_middlewares_run_in_order(self, provider_ctx):
        p, _ = provider_ctx
        order = []

        @register_middleware
        def mw1(data: MiddlewareData, next): order.append(1); next()

        @register_middleware
        def mw2(data: MiddlewareData, next): order.append(2); next()

        p.chat("hello")
        assert order == [1, 2]

    def test_global_middleware_that_skips_next_prevents_backend_call(self, provider_ctx):
        p, backend = provider_ctx

        @register_middleware
        def mw(data: MiddlewareData, next):
            pass  # no next()

        p.chat("hello")
        backend.call.assert_not_called()

    def test_global_middleware_runs_on_stream(self, provider_ctx):
        p, backend = provider_ctx
        backend.stream.return_value = iter(["chunk"])
        ran = []

        @register_middleware
        def mw(data: MiddlewareData, next): ran.append(True); next()

        list(p.stream_text("hello"))
        assert ran == [True]

    def test_global_middleware_can_mutate_prompt_for_stream(self, provider_ctx):
        p, backend = provider_ctx
        backend.stream.return_value = iter(["chunk"])

        @register_middleware
        def mw(data: MiddlewareData, next):
            data.prompt = "STREAM_MODIFIED"
            next()

        list(p.stream_text("original"))
        called_prompt = backend.stream.call_args[0][1]
        assert called_prompt == "STREAM_MODIFIED"

    def test_middleware_event_field_is_chat_for_chat(self, provider_ctx):
        p, _ = provider_ctx
        events_seen = []

        @register_middleware
        def mw(data: MiddlewareData, next): events_seen.append(data.event); next()

        p.chat("hello")
        assert events_seen == ["chat"]

    def test_middleware_event_field_is_stream_for_stream(self, provider_ctx):
        p, backend = provider_ctx
        backend.stream.return_value = iter(["x"])
        events_seen = []

        @register_middleware
        def mw(data: MiddlewareData, next): events_seen.append(data.event); next()

        list(p.stream_text("hello"))
        assert events_seen == ["stream"]

    def test_middleware_metadata_bag_is_passable(self, provider_ctx):
        p, _ = provider_ctx

        @register_middleware
        def mw1(data: MiddlewareData, next):
            data.metadata["trace_id"] = "abc123"
            next()

        @register_middleware
        def mw2(data: MiddlewareData, next):
            data.metadata["from_mw2"] = True
            next()

        p.chat("hello")  # no assertion needed — just must not raise


# ---------------------------------------------------------------------------
# Provider-level middleware
# ---------------------------------------------------------------------------

class TestProviderLevelMiddleware:
    def test_provider_middleware_runs_on_chat(self, provider_ctx):
        p, _ = provider_ctx
        ran = []

        @p.middleware
        def mw(data: MiddlewareData, next): ran.append(True); next()

        p.chat("hello")
        assert ran == [True]

    def test_provider_middleware_does_not_run_for_other_provider(self, provider_ctx, provider_ctx_b):
        p_google, _ = provider_ctx
        p_anthropic, _ = provider_ctx_b
        ran_for_google = []

        @p_google.middleware
        def mw(data: MiddlewareData, next): ran_for_google.append(True); next()

        p_anthropic.chat("hello")
        assert ran_for_google == []

    def test_global_middleware_runs_before_provider_middleware(self, provider_ctx):
        p, _ = provider_ctx
        order = []

        @register_middleware
        def global_mw(data: MiddlewareData, next): order.append("global"); next()

        @p.middleware
        def provider_mw(data: MiddlewareData, next): order.append("provider"); next()

        p.chat("hello")
        assert order == ["global", "provider"]

    def test_multiple_provider_middlewares_run_in_order(self, provider_ctx):
        p, _ = provider_ctx
        order = []

        @p.middleware
        def mw1(data: MiddlewareData, next): order.append(1); next()

        @p.middleware
        def mw2(data: MiddlewareData, next): order.append(2); next()

        p.chat("hello")
        assert order == [1, 2]

    def test_provider_middleware_can_mutate_prompt(self, provider_ctx):
        p, backend = provider_ctx

        @p.middleware
        def mw(data: MiddlewareData, next):
            data.prompt = "PROVIDER_MODIFIED"
            next()

        p.chat("original")
        called_prompt = backend.call.call_args[0][1]
        assert called_prompt == "PROVIDER_MODIFIED"

    def test_provider_middleware_can_read_response_after_next(self, provider_ctx):
        p, _ = provider_ctx
        captured = []

        @p.middleware
        def mw(data: MiddlewareData, next):
            next()
            captured.append(data.response.text)

        p.chat("hello")
        assert captured == ["Hello!"]

    def test_provider_middleware_runs_on_stream(self, provider_ctx):
        p, backend = provider_ctx
        backend.stream.return_value = iter(["x"])
        ran = []

        @p.middleware
        def mw(data: MiddlewareData, next): ran.append(True); next()

        list(p.stream_text("hello"))
        assert ran == [True]

    def test_functional_registration_works_on_provider(self, provider_ctx):
        p, _ = provider_ctx
        ran = []

        def my_mw(data: MiddlewareData, next): ran.append(True); next()

        p.middleware(my_mw)
        p.chat("hello")
        assert ran == [True]


# ---------------------------------------------------------------------------
# Middleware short-circuit edge cases
# ---------------------------------------------------------------------------

class TestMiddlewareShortCircuit:
    def test_skipping_next_means_response_is_none(self, provider_ctx):
        p, _ = provider_ctx

        @register_middleware
        def mw(data: MiddlewareData, next):
            pass  # no next()

        result = p.chat("hello")
        assert result is None

    def test_skipping_next_prevents_backend_call(self, provider_ctx):
        p, backend = provider_ctx

        @p.middleware
        def mw(data: MiddlewareData, next):
            pass  # no next()

        p.chat("hello")
        backend.call.assert_not_called()

    def test_middleware_can_set_response_manually_on_short_circuit(self, provider_ctx):
        p, backend = provider_ctx
        from orkestra._types import Response

        fake = Response(
            text="intercepted", model="fake", provider="google",
            cost=0.0, input_tokens=0, output_tokens=0,
            input_cost=0.0, output_cost=0.0,
            savings=0.0, savings_percent=0.0,
            base_model="fake", base_cost=0.0,
        )

        @p.middleware
        def mw(data: MiddlewareData, next):
            data.response = fake
            # no next() — backend never called

        result = p.chat("hello")
        assert result is fake
        backend.call.assert_not_called()

    def test_on_response_fires_even_after_short_circuit(self, provider_ctx):
        p, _ = provider_ctx
        fired = []

        @register_middleware
        def mw(data: MiddlewareData, next):
            pass  # no next()

        @register_event(ON_RESPONSE)
        def handler(data: EventData): fired.append(data.response)

        p.chat("hello")
        # on_response fires with response=None since short-circuit left it unset
        assert fired == [None]
