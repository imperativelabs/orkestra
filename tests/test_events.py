"""Tests for _events.py — EventData, EventBus, register_event, emit_event."""

import pytest

from orkestra._events import (
    EventBus,
    EventData,
    emit_event,
    register_event,
    _global_bus,
)


# ---------------------------------------------------------------------------
# Isolation — reset global bus state between every test in this file
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_global_bus():
    original = dict(_global_bus._handlers)
    _global_bus._handlers.clear()
    yield
    _global_bus._handlers.clear()
    _global_bus._handlers.update(original)


# ---------------------------------------------------------------------------
# EventData
# ---------------------------------------------------------------------------

class TestEventData:
    def test_required_fields(self):
        data = EventData(event="on_chat", provider="google", prompt="hello")
        assert data.event == "on_chat"
        assert data.provider == "google"
        assert data.prompt == "hello"

    def test_optional_model_defaults_none(self):
        data = EventData(event="on_request", provider="anthropic", prompt="hi")
        assert data.model is None

    def test_optional_response_defaults_none(self):
        data = EventData(event="on_response", provider="openai", prompt="hi")
        assert data.response is None

    def test_metadata_defaults_empty_dict(self):
        data = EventData(event="on_chunk", provider="google", prompt="hi")
        assert data.metadata == {}

    def test_metadata_is_independent_per_instance(self):
        a = EventData(event="e", provider="p", prompt="q")
        b = EventData(event="e", provider="p", prompt="q")
        a.metadata["key"] = "value"
        assert "key" not in b.metadata

    def test_all_fields_settable(self):
        data = EventData(
            event="on_response",
            provider="anthropic",
            prompt="hello",
            model="claude-haiku-4",
            response=object(),
            metadata={"x": 1},
        )
        assert data.model == "claude-haiku-4"
        assert data.metadata == {"x": 1}


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class TestEventBus:
    def test_on_registers_handler(self):
        bus = EventBus()
        fn = lambda d: None
        bus.on("my_event", fn)
        assert fn in bus._handlers["my_event"]

    def test_emit_calls_handler(self):
        bus = EventBus()
        received = []
        bus.on("ev", lambda d: received.append(d))
        data = EventData(event="ev", provider="p", prompt="q")
        bus.emit("ev", data)
        assert received == [data]

    def test_emit_unknown_event_no_error(self):
        bus = EventBus()
        data = EventData(event="nope", provider="p", prompt="q")
        bus.emit("nope", data)  # should not raise

    def test_emit_empty_handler_list_no_error(self):
        bus = EventBus()
        bus._handlers["ev"] = []
        bus.emit("ev", EventData(event="ev", provider="p", prompt="q"))

    def test_multiple_handlers_same_event_all_called(self):
        bus = EventBus()
        log = []
        bus.on("ev", lambda d: log.append("first"))
        bus.on("ev", lambda d: log.append("second"))
        bus.emit("ev", EventData(event="ev", provider="p", prompt="q"))
        assert log == ["first", "second"]

    def test_handlers_called_in_registration_order(self):
        bus = EventBus()
        order = []
        for i in range(5):
            bus.on("ev", lambda d, n=i: order.append(n))
        bus.emit("ev", EventData(event="ev", provider="p", prompt="q"))
        assert order == [0, 1, 2, 3, 4]

    def test_handlers_for_different_events_dont_interfere(self):
        bus = EventBus()
        log = []
        bus.on("ev_a", lambda d: log.append("a"))
        bus.on("ev_b", lambda d: log.append("b"))
        bus.emit("ev_a", EventData(event="ev_a", provider="p", prompt="q"))
        assert log == ["a"]

    def test_emit_passes_exact_data_object(self):
        bus = EventBus()
        received = []
        bus.on("ev", lambda d: received.append(d))
        data = EventData(event="ev", provider="p", prompt="q", model="m", metadata={"k": 1})
        bus.emit("ev", data)
        assert received[0] is data

    def test_multiple_emits_call_handler_each_time(self):
        bus = EventBus()
        count = [0]
        bus.on("ev", lambda d: count.__setitem__(0, count[0] + 1))
        data = EventData(event="ev", provider="p", prompt="q")
        bus.emit("ev", data)
        bus.emit("ev", data)
        assert count[0] == 2

    def test_independent_buses_dont_share_handlers(self):
        bus_a = EventBus()
        bus_b = EventBus()
        log = []
        bus_a.on("ev", lambda d: log.append("a"))
        bus_b.emit("ev", EventData(event="ev", provider="p", prompt="q"))
        assert log == []


# ---------------------------------------------------------------------------
# register_event (global decorator)
# ---------------------------------------------------------------------------

class TestRegisterEvent:
    def test_returns_original_function_unchanged(self):
        def my_handler(data: EventData) -> None:
            pass

        result = register_event("on_test")(my_handler)
        assert result is my_handler

    def test_registered_handler_fires_on_emit(self):
        received = []

        @register_event("on_test_fire")
        def handler(data: EventData):
            received.append(data.prompt)

        emit_event("on_test_fire", EventData(event="on_test_fire", provider="p", prompt="hello"))
        assert received == ["hello"]

    def test_multiple_decorators_same_event_all_fire(self):
        log = []

        @register_event("on_multi")
        def h1(data: EventData):
            log.append(1)

        @register_event("on_multi")
        def h2(data: EventData):
            log.append(2)

        emit_event("on_multi", EventData(event="on_multi", provider="p", prompt="q"))
        assert log == [1, 2]

    def test_different_events_registered_independently(self):
        log = []

        @register_event("ev_x")
        def hx(data: EventData):
            log.append("x")

        @register_event("ev_y")
        def hy(data: EventData):
            log.append("y")

        emit_event("ev_x", EventData(event="ev_x", provider="p", prompt="q"))
        assert log == ["x"]

    def test_handler_receives_full_event_data(self):
        received = []

        @register_event("on_data_check")
        def handler(data: EventData):
            received.append(data)

        data = EventData(
            event="on_data_check",
            provider="anthropic",
            prompt="hello",
            model="claude-haiku-4",
            metadata={"key": "val"},
        )
        emit_event("on_data_check", data)
        assert received[0].provider == "anthropic"
        assert received[0].model == "claude-haiku-4"
        assert received[0].metadata == {"key": "val"}


# ---------------------------------------------------------------------------
# emit_event (global helper)
# ---------------------------------------------------------------------------

class TestEmitEvent:
    def test_fires_all_global_handlers(self):
        log = []
        _global_bus.on("ev_emit", lambda d: log.append("called"))
        emit_event("ev_emit", EventData(event="ev_emit", provider="p", prompt="q"))
        assert log == ["called"]

    def test_unknown_event_no_error(self):
        emit_event("unknown_event", EventData(event="unknown_event", provider="p", prompt="q"))

    def test_does_not_affect_separate_bus(self):
        separate = EventBus()
        log = []
        separate.on("ev", lambda d: log.append("hit"))
        emit_event("ev", EventData(event="ev", provider="p", prompt="q"))
        assert log == []  # global emit doesn't touch separate bus
