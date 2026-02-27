# orkestra

Smart LLM routing across providers. Automatically selects the most cost-efficient model for your prompt using KNN-based routing.

## Install

```bash
pip install orkestra-router
```

## Quick Start

```python
import orkestra as o

provider = o.Provider("google", "YOUR_GEMINI_API_KEY")
response = provider.chat("Explain quantum computing")

print(response.text)
print(f"Provider: {response.provider}")
print(f"Model: {response.model}")
print(f"Cost: ${response.cost:.6f}")
print(f"Saved: {response.savings_percent:.1f}%")
```

## Multi-Provider Routing

Combine multiple providers and let orkestra pick the best one:

```python
import orkestra as o

google = o.Provider("google", "GOOGLE_KEY")
anthropic = o.Provider("anthropic", "ANTHROPIC_KEY")
openai = o.Provider("openai", "OPENAI_KEY")

multi = o.MultiProvider([google, anthropic, openai])

# Route to cheapest option across all providers
response = multi.chat("What is 2+2?", strategy="cheapest")

# Route to smartest option
response = multi.chat("Prove the Riemann hypothesis", strategy="smartest")

# Balanced: prefer mid-tier models, break ties by cost
response = multi.chat("Write a Python function", strategy="balanced")
```

## Streaming

```python
provider = o.Provider("google", "YOUR_KEY")
for chunk in provider.stream_text("Write a poem"):
    print(chunk, end="")
```

## Explore
```
======================================================================
PROVIDER MODEL CATALOG
======================================================================

  GOOGLE
    gemini-2.5-flash-lite        budget     $0.10/$0.40 per 1M tokens
    gemini-3-flash-preview       balanced   $0.50/$3.00 per 1M tokens
    gemini-3-pro-preview         premium    $2.00/$12.00 per 1M tokens

  ANTHROPIC
    claude-haiku-4               budget     $0.80/$4.00 per 1M tokens
    claude-sonnet-4-5            balanced   $3.00/$15.00 per 1M tokens
    claude-opus-4                premium    $15.00/$75.00 per 1M tokens

  OPENAI
    gpt-4o-mini                  budget     $0.15/$0.60 per 1M tokens
    gpt-4o                       balanced   $2.50/$10.00 per 1M tokens
    o3                           premium    $10.00/$40.00 per 1M tokens

======================================================================
ROUTER PREDICTIONS (which model gets picked per prompt)
======================================================================

  [Simple] What is the capital of Japan?
    google       → gemini-3-flash-preview (balanced)
    anthropic    → claude-sonnet-4-5 (balanced)
    openai       → gpt-4o (balanced)

  [Moderate] Explain how a hash table works with collision handling
    google       → gemini-3-flash-preview (balanced)
    anthropic    → claude-sonnet-4-5 (balanced)
    openai       → gpt-4o (balanced)

  [Complex] Implement a B-tree with insert and search in Python. Handle node ...
    google       → gemini-3-pro-preview (premium)
    anthropic    → claude-opus-4 (premium)
    openai       → o3 (premium)

======================================================================
COST SIMULATION (500 input tokens, 1000 output tokens)
======================================================================

  [Simple] What is the capital of Japan?
    google       gemini-3-flash-preview       $0.003250  (saves $0.009750, 75% vs gemini-3-pro-preview)
    anthropic    claude-sonnet-4-5            $0.016500  (saves $0.066000, 80% vs claude-opus-4)
    openai       gpt-4o                       $0.011250  (saves $0.033750, 75% vs o3)

  [Moderate] Explain how a hash table works with collision handling
    google       gemini-3-flash-preview       $0.003250  (saves $0.009750, 75% vs gemini-3-pro-preview)
    anthropic    claude-sonnet-4-5            $0.016500  (saves $0.066000, 80% vs claude-opus-4)
    openai       gpt-4o                       $0.011250  (saves $0.033750, 75% vs o3)

  [Complex] Implement a B-tree with insert and search in Python. Handle node ...
    google       gemini-3-pro-preview         $0.013000  (saves $0.000000, 0% vs gemini-3-pro-preview)
    anthropic    claude-opus-4                $0.082500  (saves $0.000000, 0% vs claude-opus-4)
    openai       o3                           $0.045000  (saves $0.000000, 0% vs o3)

======================================================================
STRATEGY COMPARISON (multi-provider selection)
======================================================================

  [Simple] What is the capital of Japan?
    Per-provider routes: google→gemini-3-flash-preview, anthropic→claude-sonnet-4-5, openai→gpt-4o
    cheapest   → google/gemini-3-flash-preview (balanced) $0.003250
    smartest   → google/gemini-3-flash-preview (balanced) $0.003250
    balanced   → google/gemini-3-flash-preview (balanced) $0.003250

  [Moderate] Explain how a hash table works with collision handling
    Per-provider routes: google→gemini-3-flash-preview, anthropic→claude-sonnet-4-5, openai→gpt-4o
    cheapest   → google/gemini-3-flash-preview (balanced) $0.003250
    smartest   → google/gemini-3-flash-preview (balanced) $0.003250
    balanced   → google/gemini-3-flash-preview (balanced) $0.003250

  [Complex] Implement a B-tree with insert and search in Python. Handle node ...
    Per-provider routes: google→gemini-3-pro-preview, anthropic→claude-opus-4, openai→o3
    cheapest   → google/gemini-3-pro-preview (premium) $0.013000
    smartest   → google/gemini-3-pro-preview (premium) $0.013000
    balanced   → google/gemini-3-pro-preview (premium) $0.013000

======================================================================
PUBLIC API
======================================================================

  import orkestra as o

  # Single provider (routes within one provider's model family)
  provider = o.Provider("google", "API_KEY")
  response = provider.chat("your prompt")
  stream   = provider.stream_text("your prompt")

  # Multi-provider (picks best provider+model using a strategy)
  multi = o.MultiProvider([provider1, provider2])
  response = multi.chat("your prompt", strategy="cheapest")
  response = multi.chat("your prompt", strategy="smartest")
  response = multi.chat("your prompt", strategy="balanced")

  # Response fields
  response.text             # generated text
  response.model            # model used (e.g. "gemini-2.5-flash-lite")
  response.provider         # provider name (e.g. "google")
  response.cost             # total cost in dollars
  response.input_tokens     # input token count
  response.output_tokens    # output token count
  response.savings          # dollars saved vs premium model
  response.savings_percent  # savings as percentage
```

## How It Works

Orkestra uses a KNN (K-Nearest Neighbors) router trained on benchmark query embeddings to predict which model tier will perform best for your specific prompt. Simple queries get routed to cheaper models, complex ones to premium models.

Each call:
1. Embeds your prompt using Longformer (768-dim)
2. KNN finds the 5 nearest training queries
3. Routes to the model that performed best on similar queries
4. Calls the selected model via the provider's API
5. Returns the response with cost and savings info

Router models are downloaded automatically on first use and cached in `~/.orkestra/routers/`.

## Supported Providers

### Google Gemini

| Tier | Model | Input $/1M | Output $/1M |
|------|-------|-----------|------------|
| Budget | `gemini-2.5-flash-lite` | $0.10 | $0.40 |
| Balanced | `gemini-3-flash-preview` | $0.50 | $3.00 |
| Premium | `gemini-3-pro-preview` | $2.00 | $12.00 |

### Anthropic Claude

| Tier | Model | Input $/1M | Output $/1M |
|------|-------|-----------|------------|
| Budget | `claude-haiku-4` | $0.80 | $4.00 |
| Balanced | `claude-sonnet-4-5` | $3.00 | $15.00 |
| Premium | `claude-opus-4` | $15.00 | $75.00 |

### OpenAI

| Tier | Model | Input $/1M | Output $/1M |
|------|-------|-----------|------------|
| Budget | `gpt-4o-mini` | $0.15 | $0.60 |
| Balanced | `gpt-4o` | $2.50 | $10.00 |
| Premium | `o3` | $10.00 | $40.00 |

## API Reference

### `orkestra.Provider(name, api_key)`

Create a provider with automatic routing.

- `name`: `"google"`, `"anthropic"`, or `"openai"`
- `api_key`: Your API key for the provider

### `provider.chat(prompt, *, max_tokens=8192, temperature=1.0)`

Generate a response with automatic model routing. Returns a `Response`.

### `provider.stream_text(prompt, *, max_tokens=8192, temperature=1.0)`

Stream text chunks with automatic model routing. Yields `str`.

### `orkestra.MultiProvider(providers)`

Combine multiple `Provider` instances for cross-provider routing.

### `multi.chat(prompt, *, strategy="cheapest", max_tokens=8192, temperature=1.0)`

Generate with strategy-based provider selection. Strategies: `"cheapest"`, `"smartest"`, `"balanced"`.

### `orkestra.Response`

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Generated response text |
| `model` | `str` | Model that was selected |
| `provider` | `str` | Provider that was used |
| `cost` | `float` | Actual cost in dollars |
| `input_tokens` | `int` | Input token count |
| `output_tokens` | `int` | Output token count |
| `savings` | `float` | Dollars saved vs base model |
| `savings_percent` | `float` | Percentage saved vs base model |
| `base_model` | `str` | Comparison baseline model |
| `base_cost` | `float` | What it would have cost with base model |

## License

MIT
