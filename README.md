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

### `o.Provider(name, api_key)`

Create a single-provider router.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | `"google"`, `"anthropic"`, or `"openai"` |
| `api_key` | `str` | Your API key for the chosen provider |

---

### `provider.chat(prompt, *, max_tokens=8192, temperature=1.0)`

Route a prompt and return a full response. Returns an `orkestra.Response`.

---

### `provider.stream_text(prompt, *, max_tokens=8192, temperature=1.0)`

Stream response tokens as they arrive. Yields `str` chunks.

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
