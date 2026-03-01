"""Main Provider class for single-provider LLM routing."""

from __future__ import annotations

from typing import Callable, Iterator

from orkestra._events import (
    ON_CHAT,
    ON_CHUNK,
    ON_REQUEST,
    ON_RESPONSE,
    ON_ROUTE,
    ON_STREAM,
    ON_STREAM_COMPLETE,
    EventBus,
    EventData,
    EventName,
    emit_event,
)
from orkestra._middleware import MiddlewareData, _global_middlewares, _run_chain
from orkestra._types import Response
from orkestra.providers import create_backend
from orkestra.providers.base import ProviderBackend
from orkestra.registry.models import (
    DEFAULT_BASE_MODELS,
    DEFAULT_FALLBACK_MODELS,
    PROVIDER_MODELS,
    calculate_cost,
)


class Provider:
    """Routes prompts to the optimal model for a single LLM provider.

    Usage::

        import orkestra as o

        provider = o.Provider("google", "YOUR_API_KEY")
        response = provider.chat("Explain quantum computing")
        print(response.text)
        print(f"Model: {response.model}, Cost: ${response.cost:.6f}")

        # Disable smart routing and use a fixed model
        provider = o.Provider("google", "YOUR_API_KEY", smart_routing=False)
        response = provider.chat("Hello")  # uses gemini-3-flash-preview

        # Disable smart routing with a custom default model
        provider = o.Provider("google", "YOUR_API_KEY", smart_routing=False,
                              default_model="gemini-2.5-flash-lite")
        response = provider.chat("Hello", model="gemini-3-pro-preview")

    Supported providers: google, anthropic, openai.
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        smart_routing: bool = True,
        default_model: str | None = None,
    ):
        """Initialize a provider with optional smart routing.

        Args:
            name: Provider name ("google", "anthropic", "openai").
            api_key: API key for the provider.
            smart_routing: If True (default), uses KNN routing to pick the best
                model per prompt. If False, uses a fixed default model.
            default_model: Fixed model to use when smart_routing=False. If not
                provided, falls back to the balanced-tier model for the provider.
                Cannot be an empty string when smart_routing=False.

        Raises:
            ValueError: If smart_routing=False and default_model is an empty string.
        """
        if not smart_routing and default_model is not None and default_model == "":
            raise ValueError("default_model cannot be an empty string when smart_routing=False")

        self._backend: ProviderBackend = create_backend(name, api_key)
        self._smart_routing = smart_routing
        self._base_model = DEFAULT_BASE_MODELS.get(name)

        if smart_routing:
            from orkestra.router.knn import KNNRouter
            self._router = KNNRouter(provider=name)
            self._default_model: str | None = None
        else:
            self._router = None
            self._default_model = (
                default_model if default_model is not None
                else DEFAULT_FALLBACK_MODELS.get(name)
            )

        # Per-provider middleware stack and event bus
        self._middlewares: list[Callable] = []
        self._event_bus = EventBus()

    # ------------------------------------------------------------------
    # Public decorator API
    # ------------------------------------------------------------------

    def middleware(self, fn: Callable) -> Callable:
        """Register a provider-level middleware.

        Usage::

            @provider.middleware
            def my_middleware(data: MiddlewareData, next):
                print(f"prompt: {data.prompt}")
                next()
        """
        self._middlewares.append(fn)
        return fn

    def event(self, event_name: EventName) -> Callable:
        """Register a provider-level event handler.

        Usage::

            @provider.event("on_response")
            def log_response(data: EventData):
                print(f"model={data.model}")
        """
        def decorator(fn: Callable[[EventData], None]) -> Callable[[EventData], None]:
            self._event_bus.on(event_name, fn)
            return fn
        return decorator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._backend.name

    def _resolve_model(self, model: str | None, prompt: str) -> str:
        """Resolve which model to use for a request."""
        if not self._smart_routing:
            if model is not None and model == "":
                raise ValueError("model cannot be an empty string when smart_routing=False")
            return model if model is not None else self._default_model #type:ignore
        return self._router.route(prompt) #type:ignore

    def _emit(self, event_name: EventName, data: EventData) -> None:
        """Fire event on both the global bus and this provider's bus."""
        emit_event(event_name, data)
        self._event_bus.emit(event_name, data)

    def _build_response(
        self,
        model: str,
        result: dict,
    ) -> Response:
        """Construct a Response from a raw backend result dict."""
        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        cost = calculate_cost(self.name, model, input_tokens, output_tokens)

        model_info = PROVIDER_MODELS[self.name][model]
        input_cost = (input_tokens * model_info["input_price"]) / 1_000_000
        output_cost = (output_tokens * model_info["output_price"]) / 1_000_000

        base_model = self._base_model
        base_cost = 0.0
        savings = 0.0
        savings_percent = 0.0

        if base_model and base_model in PROVIDER_MODELS[self.name]:
            base_cost = calculate_cost(self.name, base_model, input_tokens, output_tokens)
            savings = base_cost - cost
            savings_percent = (savings / base_cost * 100) if base_cost > 0 else 0.0

        return Response(
            text=result["text"],
            model=model,
            provider=self.name,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            savings=savings,
            savings_percent=savings_percent,
            base_model=base_model or model,
            base_cost=base_cost,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 1.0,
    ) -> Response:
        """Generate a response using the optimally-routed model.

        Args:
            prompt: The input prompt.
            model: Override the model to use. Only applied when
                smart_routing=False; cannot be an empty string in that case.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            A Response with the generated text, cost, and savings info.
        """
        mw_data = MiddlewareData(
            prompt=prompt,
            provider=self.name,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            event="chat",
        )

        # Fire pre-execution events
        pre_event = EventData(event=ON_REQUEST, provider=self.name, prompt=prompt)
        self._emit(ON_REQUEST, pre_event)
        self._emit(ON_CHAT, EventData(event=ON_CHAT, provider=self.name, prompt=prompt))

        def final_handler(data: MiddlewareData) -> None:
            resolved = self._resolve_model(data.model, data.prompt)
            data.model = resolved

            route_event = EventData(
                event=ON_ROUTE, provider=self.name, prompt=data.prompt, model=resolved
            )
            self._emit(ON_ROUTE, route_event)

            result = self._backend.call(resolved, data.prompt, data.max_tokens, data.temperature)
            data.response = self._build_response(resolved, result)

        _run_chain(_global_middlewares + self._middlewares, mw_data, final_handler)

        response_event = EventData(
            event=ON_RESPONSE,
            provider=self.name,
            prompt=mw_data.prompt,
            model=mw_data.model,
            response=mw_data.response,
        )
        self._emit(ON_RESPONSE, response_event)

        return mw_data.response

    def stream_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 1.0,
    ) -> Iterator[str]:
        """Stream text from the optimally-routed model.

        Middleware runs before the stream begins (can mutate prompt/model).
        Events fire per-chunk and on completion.

        Args:
            prompt: The input prompt.
            model: Override the model to use. Only applied when
                smart_routing=False; cannot be an empty string in that case.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Yields:
            Text chunks as they arrive.
        """
        mw_data = MiddlewareData(
            prompt=prompt,
            provider=self.name,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            event="stream",
        )

        # Fire pre-execution events
        self._emit(ON_REQUEST, EventData(event=ON_REQUEST, provider=self.name, prompt=prompt))
        self._emit(ON_STREAM, EventData(event=ON_STREAM, provider=self.name, prompt=prompt))

        def pre_handler(data: MiddlewareData) -> None:
            resolved = self._resolve_model(data.model, data.prompt)
            data.model = resolved
            self._emit(
                ON_ROUTE,
                EventData(event=ON_ROUTE, provider=self.name, prompt=data.prompt, model=resolved),
            )

        _run_chain(_global_middlewares + self._middlewares, mw_data, pre_handler)

        resolved_model = mw_data.model
        resolved_prompt = mw_data.prompt

        def _wrapped_stream() -> Iterator[str]:
            for chunk in self._backend.stream(resolved_model, resolved_prompt, max_tokens, temperature): #type:ignore
                self._emit(
                    ON_CHUNK,
                    EventData(
                        event=ON_CHUNK,
                        provider=self.name,
                        prompt=resolved_prompt,
                        model=resolved_model,
                        metadata={"chunk": chunk},
                    ),
                )
                yield chunk
            self._emit(
                ON_STREAM_COMPLETE,
                EventData(
                    event=ON_STREAM_COMPLETE,
                    provider=self.name,
                    prompt=resolved_prompt,
                    model=resolved_model,
                ),
            )

        yield from _wrapped_stream()
