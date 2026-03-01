"""Event system for orkestra lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from orkestra._types import Response

# ---------------------------------------------------------------------------
# Event name type
# ---------------------------------------------------------------------------

EventName = Literal[
    "on_request",        # any request (chat or stream), before execution
    "on_chat",           # chat() call, before execution
    "on_stream",         # stream_text() call, before execution
    "on_route",          # after model is resolved
    "on_response",       # after chat() returns a Response
    "on_chunk",          # each streaming chunk (metadata["chunk"])
    "on_stream_complete", # stream generator exhausted
]

# ---------------------------------------------------------------------------
# Event name constants
# ---------------------------------------------------------------------------

ON_REQUEST: EventName = "on_request"
ON_CHAT: EventName = "on_chat"
ON_STREAM: EventName = "on_stream"
ON_ROUTE: EventName = "on_route"
ON_RESPONSE: EventName = "on_response"
ON_CHUNK: EventName = "on_chunk"
ON_STREAM_COMPLETE: EventName = "on_stream_complete"


# ---------------------------------------------------------------------------
# EventData
# ---------------------------------------------------------------------------

@dataclass
class EventData:
    """Data passed to every event handler.

    Fields are read-only by convention — use middleware to mutate requests.
    """

    event: EventName
    provider: str
    prompt: str
    model: str | None = None
    response: Any = None  # Response | None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """Holds event handlers and dispatches events to them."""

    def __init__(self) -> None:
        self._handlers: dict[EventName, list[Callable[[EventData], None]]] = {}

    def on(self, event_name: EventName, handler: Callable[[EventData], None]) -> None:
        """Register a handler for an event name."""
        self._handlers.setdefault(event_name, []).append(handler)

    def emit(self, event_name: EventName, data: EventData) -> None:
        """Fire all handlers registered for event_name."""
        for handler in self._handlers.get(event_name, []):
            handler(data)


# ---------------------------------------------------------------------------
# Global bus + public API
# ---------------------------------------------------------------------------

_global_bus = EventBus()


def register_event(event_name: EventName) -> Callable:
    """Decorator — registers a handler on the global event bus.

    Usage::

        @register_event("on_response")
        def log_cost(data: EventData):
            print(f"cost=${data.response.cost:.6f}")
    """
    def decorator(fn: Callable[[EventData], None]) -> Callable[[EventData], None]:
        _global_bus.on(event_name, fn)
        return fn
    return decorator


def emit_event(event_name: EventName, data: EventData) -> None:
    """Internal helper — fires the global bus for event_name."""
    _global_bus.emit(event_name, data)
