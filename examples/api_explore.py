"""API explore: walk through orkestra's core API and see what it does.

No API keys needed — this script only exercises the router and registry,
showing which models get picked and what costs would look like.

Run: python examples/api_explore.py
"""

import orkestra as o
from orkestra.router.knn import KNNRouter
from orkestra.registry.models import PROVIDER_MODELS, calculate_cost
from orkestra.registry.strategies import STRATEGIES

print(f"orkestra v{o.__version__}\n")

# ── Available providers and their model tiers ────────────────────────────────

print("=" * 70)
print("PROVIDER MODEL CATALOG")
print("=" * 70)
for provider, models in PROVIDER_MODELS.items():
    print(f"\n  {provider.upper()}")
    for model, info in models.items():
        print(
            f"    {model:<28} {info['tier']:<10} "
            f"${info['input_price']:.2f}/${info['output_price']:.2f} per 1M tokens"
        )

# ── Router predictions ───────────────────────────────────────────────────────

prompts = [
    ("Simple",   "What is the capital of Japan?"),
    ("Moderate", "Explain how a hash table works with collision handling"),
    ("Complex",  "Implement a B-tree with insert and search in Python. "
                 "Handle node splits correctly and support configurable order."),
]

print(f"\n{'=' * 70}")
print("ROUTER PREDICTIONS (which model gets picked per prompt)")
print("=" * 70)

routers = {p: KNNRouter(p) for p in PROVIDER_MODELS}

for label, prompt in prompts:
    print(f"\n  [{label}] {prompt[:65]}{'...' if len(prompt) > 65 else ''}")
    for provider in PROVIDER_MODELS:
        model = routers[provider].route(prompt)
        tier = PROVIDER_MODELS[provider][model]["tier"]
        print(f"    {provider:<12} → {model} ({tier})")

# ── Cost simulation ──────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("COST SIMULATION (500 input tokens, 1000 output tokens)")
print("=" * 70)

input_tokens, output_tokens = 500, 1000

for label, prompt in prompts:
    print(f"\n  [{label}] {prompt[:65]}{'...' if len(prompt) > 65 else ''}")
    for provider in PROVIDER_MODELS:
        model = routers[provider].route(prompt)
        cost = calculate_cost(provider, model, input_tokens, output_tokens)

        # Compare to premium
        premium = [m for m, i in PROVIDER_MODELS[provider].items() if i["tier"] == "premium"][0]
        premium_cost = calculate_cost(provider, premium, input_tokens, output_tokens)
        savings = premium_cost - cost
        pct = (savings / premium_cost * 100) if premium_cost > 0 else 0

        print(
            f"    {provider:<12} {model:<28} "
            f"${cost:.6f}  (saves ${savings:.6f}, {pct:.0f}% vs {premium})"
        )

# ── Strategy comparison ──────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("STRATEGY COMPARISON (multi-provider selection)")
print("=" * 70)

for label, prompt in prompts:
    per_provider = {p: routers[p].route(prompt) for p in PROVIDER_MODELS}
    print(f"\n  [{label}] {prompt[:65]}{'...' if len(prompt) > 65 else ''}")
    print(f"    Per-provider routes: ", end="")
    print(", ".join(f"{p}→{m}" for p, m in per_provider.items()))

    for strategy_name in STRATEGIES:
        # Inline strategy logic (same as MultiProvider internals)
        from orkestra.registry.models import TIER_RANK
        if strategy_name == "cheapest":
            winner = min(per_provider.items(),
                         key=lambda x: PROVIDER_MODELS[x[0]][x[1]]["input_price"]
                                      + PROVIDER_MODELS[x[0]][x[1]]["output_price"])
        elif strategy_name == "smartest":
            winner = max(per_provider.items(),
                         key=lambda x: (TIER_RANK[PROVIDER_MODELS[x[0]][x[1]]["tier"]],
                                        -(PROVIDER_MODELS[x[0]][x[1]]["input_price"]
                                          + PROVIDER_MODELS[x[0]][x[1]]["output_price"])))
        else:  # balanced
            balanced_opts = [(p, m) for p, m in per_provider.items()
                             if PROVIDER_MODELS[p][m]["tier"] == "balanced"]
            if balanced_opts:
                winner = min(balanced_opts,
                             key=lambda x: PROVIDER_MODELS[x[0]][x[1]]["input_price"]
                                          + PROVIDER_MODELS[x[0]][x[1]]["output_price"])
            else:
                winner = min(per_provider.items(),
                             key=lambda x: PROVIDER_MODELS[x[0]][x[1]]["input_price"]
                                          + PROVIDER_MODELS[x[0]][x[1]]["output_price"])

        prov, model = winner
        tier = PROVIDER_MODELS[prov][model]["tier"]
        cost = calculate_cost(prov, model, input_tokens, output_tokens)
        print(f"    {strategy_name:<10} → {prov}/{model} ({tier}) ${cost:.6f}")

# ── Exports overview ─────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("PUBLIC API")
print("=" * 70)
print("""
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
""")
