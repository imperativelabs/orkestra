"""Middleware system for orkestra request/response pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# MiddlewareData
# ---------------------------------------------------------------------------

@dataclass
class MiddlewareData:
    """Mutable context object threaded through the middleware chain.

    Middleware handlers receive this object and may mutate it before calling
    ``next()`` (to influence the request) or after (to inspect/alter the
    response).

    Attributes:
        prompt: The prompt text. Can be mutated pre-``next()`` to transform it.
        provider: Provider name (read-only, informational).
        model: Resolved model name. Set after routing; middleware may override.
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.
        event: ``"chat"`` or ``"stream"``.
        response: Populated by the final handler after the LLM call. Readable
            (and mutable) in post-``next()`` code.
        metadata: User-extensible bag for passing arbitrary data through the chain.
    """

    prompt: str
    provider: str
    model: str | None
    max_tokens: int
    temperature: float
    event: str
    response: Any = None  # Response | None, set by final handler
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Global middleware registry
# ---------------------------------------------------------------------------

_global_middlewares: list[Callable] = []


def register_middleware(fn: Callable) -> Callable:
    """Register a middleware on the global stack.

    Works both as a decorator and as a plain function call::

        # Decorator style
        @register_middleware
        def my_middleware(data: MiddlewareData, next):
            next()

        # Functional style (e.g. from a third-party package)
        register_middleware(some_package.my_middleware)
    """
    _global_middlewares.append(fn)
    return fn


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------

def _run_chain(
    middlewares: list[Callable],
    data: MiddlewareData,
    final: Callable[[MiddlewareData], None],
) -> None:
    """Run the middleware chain followed by the final handler.

    Each middleware receives ``(data, next_fn)``.  Calling ``next_fn()``
    advances to the next middleware (or the final handler).  All state is
    communicated through mutation of ``data``.

    Args:
        middlewares: Ordered list of middleware callables.
        data: Shared mutable context.
        final: The actual LLM call handler; writes ``data.response``.
    """
    def make_next(index: int) -> Callable[[], None]:
        def next_fn() -> None:
            if index >= len(middlewares):
                final(data)
            else:
                middlewares[index](data, make_next(index + 1))
        return next_fn

    if middlewares:
        middlewares[0](data, make_next(1))
    else:
        final(data)
