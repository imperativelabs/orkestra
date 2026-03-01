# Orkestra

**Stop overpaying for LLM calls. Orkestra automatically routes every prompt to the cheapest model that can handle it.**

Simple questions go to budget models. Hard ones go to premium models. You pay for what you actually need — automatically.

[![PyPI](https://img.shields.io/pypi/v/orkestra-router)](https://pypi.org/project/orkestra-router/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The Problem

Most teams call GPT-4 or Claude Opus for everything — even when asking "What's the capital of France?" That's like hiring a surgeon to put on a bandage. You're burning money on every simple call.

## The Solution

Orkestra uses a KNN-based router trained on benchmark queries to classify prompt complexity in real time. Simple prompts get routed to cheap, fast models. Complex prompts get premium models that can actually handle them. You just call `.chat()` — Orkestra handles the rest.

> 💡 **Typical savings: 70–80%** on mixed workloads, with no measurable quality loss on simple tasks.

---

## Install

```bash
pip install orkestra-router
```

---

## Quick Start

```python
import orkestra as o

# Connect to a provider
provider = o.Provider("google", "YOUR_GEMINI_API_KEY")

# Send a prompt — Orkestra picks the right model automatically
response = provider.chat("Explain quantum computing")

print(response.text)
print(f"Provider:  {response.provider}")   # → google
print(f"Model:     {response.model}")      # → gemini-2.5-flash-lite
print(f"Cost:      ${response.cost:.6f}") # → $0.000250
print(f"Saved:     {response.savings_percent:.1f}%")  # → 75.0%
```

---

## Route Across Multiple Providers

Connect all your providers and let Orkestra pick the best one based on your strategy:

```python
import orkestra as o

google    = o.Provider("google",    "GOOGLE_KEY")
anthropic = o.Provider("anthropic", "ANTHROPIC_KEY")
openai    = o.Provider("openai",    "OPENAI_KEY")

multi = o.MultiProvider([google, anthropic, openai])

# Always pick the cheapest option that fits the task
response = multi.chat("What is 2+2?", strategy="cheapest")

# Pull out the most capable model for hard problems
response = multi.chat("Prove the Riemann hypothesis", strategy="smartest")

# Balance cost and capability for everyday tasks
response = multi.chat("Write a Python function", strategy="balanced")
```

---

## Streaming

```python
provider = o.Provider("google", "YOUR_KEY")

for chunk in provider.stream_text("Write a poem about the sea"):
    print(chunk, end="", flush=True)
```

---

## Disable Smart Routing

When you need a fixed model instead of KNN routing:

```python
import orkestra as o

# Uses claude-sonnet-4-5 by default when smart_routing=False
provider = o.Provider("anthropic", "YOUR_KEY", smart_routing=False)

# Or specify your own default model
provider = o.Provider("anthropic", "YOUR_KEY", smart_routing=False, default_model="claude-haiku-4")

# Override per call
response = provider.chat("Hello", model="claude-opus-4")
```

---

## Events

Orkestra fires lifecycle events at every stage of a request. Register handlers globally or per-provider to log, monitor, or instrument your calls.

**Global events** fire for every provider:

```python
from orkestra import register_event, EventData

@register_event("on_response")
def log_cost(data: EventData):
    print(f"[{data.provider}] {data.model} — ${data.response.cost:.6f}")

@register_event("on_route")
def track_routing(data: EventData):
    print(f"Routed to: {data.model}")

@register_event("on_chunk")
def on_chunk(data: EventData):
    print(data.metadata["chunk"], end="", flush=True)

@register_event("on_stream_complete")
def on_done(data: EventData):
    print()  # newline after stream
```

**Provider-level events** fire only for that provider instance:

```python
provider = o.Provider("anthropic", "YOUR_KEY")

@provider.event("on_response")
def log_anthropic(data: EventData):
    print(f"Anthropic cost: ${data.response.cost:.6f}")
```

**All event names:**

| Event | When it fires | Notable `data` fields |
|-------|--------------|----------------------|
| `"on_request"` | Before any call (chat or stream) | `provider`, `prompt` |
| `"on_chat"` | Before `chat()` executes | `provider`, `prompt` |
| `"on_stream"` | Before `stream_text()` executes | `provider`, `prompt` |
| `"on_route"` | After the model is selected | `model` |
| `"on_response"` | After `chat()` returns | `model`, `response` |
| `"on_chunk"` | Per chunk in `stream_text()` | `metadata["chunk"]` |
| `"on_stream_complete"` | Stream generator exhausted | `model` |

---

## Middleware

Middleware intercepts every request/response in a pipeline — like Express.js. Call `next()` to continue, skip it to short-circuit. Mutate `data` before `next()` to transform the request; read `data.response` after to inspect or alter the result.

**Global middleware** runs for every provider:

```python
from orkestra import register_middleware, MiddlewareData

@register_middleware
def add_system_context(data: MiddlewareData, next):
    data.prompt = f"You are a helpful assistant.\n\n{data.prompt}"
    next()

@register_middleware
def log_latency(data: MiddlewareData, next):
    import time
    start = time.time()
    next()
    elapsed = time.time() - start
    print(f"[{data.provider}] {elapsed:.2f}s — {data.response.output_tokens} tokens")
```

**Provider-level middleware** runs only for that instance, after global middleware:

```python
provider = o.Provider("anthropic", "YOUR_KEY")

@provider.middleware
def anthropic_audit(data: MiddlewareData, next):
    print(f"Sending to Anthropic: {data.prompt[:80]}")
    next()
    print(f"Response: {data.response.text[:80]}")
```

**Register without decorators** (useful for third-party middleware packages):

```python
from orkestra import register_middleware
import my_logging_middleware

register_middleware(my_logging_middleware.track)  # global
provider.middleware(my_logging_middleware.track)  # provider-level
```

**Short-circuit a request** by not calling `next()`:

```python
from orkestra import register_middleware

blocked_terms = ["confidential", "internal only"]

@register_middleware
def content_filter(data: MiddlewareData, next):
    if any(term in data.prompt.lower() for term in blocked_terms):
        data.response = None  # block the call
        return
    next()
```

**`MiddlewareData` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `str` | The prompt — mutate before `next()` to transform it |
| `provider` | `str` | Provider name |
| `model` | `str \| None` | Resolved model (set after routing) |
| `max_tokens` | `int` | Max output tokens |
| `temperature` | `float` | Sampling temperature |
| `event` | `str` | `"chat"` or `"stream"` |
| `response` | `Response \| None` | Populated after `next()` returns |
| `metadata` | `dict` | User-extensible bag for passing data through the chain |

---

## How It Works

Orkestra classifies every prompt at call time using a lightweight ML router — no config required.

```
Your Prompt
    ↓
Embed with Longformer (768-dim)
    ↓
KNN finds 5 nearest benchmark queries
    ↓
Predict: budget / balanced / premium
    ↓
Call selected model via provider API
    ↓
Return response + cost + savings info
```

Router models download automatically on first use and are cached at `~/.orkestra/routers/`.

---

## Real-World Cost Example

Here's what Orkestra saves on a mix of simple, moderate, and complex prompts (500 input / 1,000 output tokens each):

| Prompt | Model Selected | Cost | Savings vs Premium |
|--------|---------------|------|-------------------|
| "What's the capital of Japan?" | gemini-3-flash-preview | $0.0033 | **75%** |
| "Explain hash tables with collision handling" | gemini-3-flash-preview | $0.0033 | **75%** |
| "Implement a B-tree with insert + search" | gemini-3-pro-preview | $0.0130 | 0% (needs premium) |

Orkestra knows when to save and when to spend.

---

## Supported Models

### Google Gemini

| Tier | Model | Input / 1M tokens | Output / 1M tokens |
|------|-------|-------------------|-------------------|
| Budget | `gemini-2.5-flash-lite` | $0.10 | $0.40 |
| Balanced | `gemini-3-flash-preview` | $0.50 | $3.00 |
| Premium | `gemini-3-pro-preview` | $2.00 | $12.00 |

### Anthropic Claude

| Tier | Model | Input / 1M tokens | Output / 1M tokens |
|------|-------|-------------------|-------------------|
| Budget | `claude-haiku-4` | $0.80 | $4.00 |
| Balanced | `claude-sonnet-4-5` | $3.00 | $15.00 |
| Premium | `claude-opus-4` | $15.00 | $75.00 |

### OpenAI

| Tier | Model | Input / 1M tokens | Output / 1M tokens |
|------|-------|-------------------|-------------------|
| Budget | `gpt-4o-mini` | $0.15 | $0.60 |
| Balanced | `gpt-4o` | $2.50 | $10.00 |
| Premium | `o3` | $10.00 | $40.00 |

---

## API Reference

### `o.Provider(name, api_key, *, smart_routing=True, default_model=None)`

Create a single-provider router.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | `"google"`, `"anthropic"`, or `"openai"` |
| `api_key` | `str` | — | Your API key for the chosen provider |
| `smart_routing` | `bool` | `True` | When `False`, skips KNN routing and uses a fixed model |
| `default_model` | `str \| None` | `None` | Fixed model to use when `smart_routing=False`; defaults to the balanced-tier model |

---

### `provider.chat(prompt, *, model=None, max_tokens=8192, temperature=1.0)`

Route a prompt and return a full response. Returns an `orkestra.Response`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str \| None` | Per-call model override (only used when `smart_routing=False`) |

---

### `provider.stream_text(prompt, *, model=None, max_tokens=8192, temperature=1.0)`

Stream response tokens as they arrive. Yields `str` chunks.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str \| None` | Per-call model override (only used when `smart_routing=False`) |

---

### `provider.middleware(fn)`

Register a middleware on this provider. Works as a decorator or plain call.

---

### `provider.event(event_name)`

Register an event handler on this provider. Use as a decorator: `@provider.event("on_response")`.

---

### `register_middleware(fn)`

Register a global middleware that runs for every provider. Works as a decorator or plain call.

---

### `register_event(event_name)`

Register a global event handler. Use as a decorator: `@register_event("on_response")`.

---

### `o.MultiProvider(providers)`

Combine multiple `Provider` instances for cross-provider routing.

---

### `multi.chat(prompt, *, strategy="cheapest", max_tokens=8192, temperature=1.0)`

Route across providers using a selection strategy.

| Strategy | Behavior |
|----------|----------|
| `"cheapest"` | Always picks the lowest-cost model that fits the task |
| `"smartest"` | Always picks the highest-capability model available |
| `"balanced"` | Prefers mid-tier models; breaks ties by cost |

---

### `orkestra.Response`

Every call returns a `Response` object with full transparency into what was used and what it cost.

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | The generated response |
| `model` | `str` | Model selected (e.g. `"gemini-2.5-flash-lite"`) |
| `provider` | `str` | Provider used (e.g. `"google"`) |
| `cost` | `float` | Actual cost in USD |
| `input_tokens` | `int` | Tokens in your prompt |
| `output_tokens` | `int` | Tokens in the response |
| `savings` | `float` | USD saved vs the premium baseline |
| `savings_percent` | `float` | Percentage saved vs the premium baseline |
| `base_model` | `str` | The premium model used as the cost baseline |
| `base_cost` | `float` | What the call would have cost with the premium model |

---

## License

MIT

---

## Integrations

- [OpenClaw](integrations/openclaw/README.md) — use Orkestra as a cost-routing skill inside the [OpenClaw](https://github.com/openclaw/openclaw) personal AI assistant
