"""Stress test: route 50 prompts and log which models are selected.

This does NOT call any LLM APIs — it only runs the KNN router to see
which model would be chosen for each prompt across providers and strategies.

Run: python examples/stress_test.py
"""

from __future__ import annotations

import time
from collections import Counter

from orkestra.registry.models import PROVIDER_MODELS, TIER_RANK
from orkestra.registry.strategies import STRATEGIES
from orkestra.router.knn import KNNRouter

# ── Prompts ──────────────────────────────────────────────────────────────────

PROMPTS: list[dict[str, str]] = [
    # --- Simple / casual (expect budget tier) ---
    {"tag": "simple", "text": "What time is it in Tokyo?"},
    {"tag": "simple", "text": "Convert 100 USD to EUR"},
    {"tag": "simple", "text": "What's the capital of France?"},
    {"tag": "simple", "text": "How many ounces in a pound?"},
    {"tag": "simple", "text": "Translate 'hello' to Spanish"},
    {"tag": "simple", "text": "What year did World War 2 end?"},
    {"tag": "simple", "text": "Give me a random fun fact"},
    {"tag": "simple", "text": "What does HTML stand for?"},
    {"tag": "simple", "text": "Is a tomato a fruit or vegetable?"},
    {"tag": "simple", "text": "Who painted the Mona Lisa?"},

    # --- Moderate knowledge (expect budget or balanced) ---
    {"tag": "moderate", "text": "Explain how photosynthesis works in simple terms"},
    {"tag": "moderate", "text": "What are the differences between HTTP and HTTPS?"},
    {"tag": "moderate", "text": "Summarize the plot of Romeo and Juliet in 3 sentences"},
    {"tag": "moderate", "text": "List 5 tips for improving sleep quality"},
    {"tag": "moderate", "text": "What is the difference between a stack and a queue?"},
    {"tag": "moderate", "text": "Explain supply and demand to a 10-year-old"},
    {"tag": "moderate", "text": "What are the pros and cons of electric vehicles?"},
    {"tag": "moderate", "text": "How does a binary search algorithm work?"},
    {"tag": "moderate", "text": "What is the difference between machine learning and deep learning?"},
    {"tag": "moderate", "text": "Explain the concept of compound interest with an example"},

    # --- Complex coding (expect balanced or premium) ---
    {"tag": "coding", "text": "Write a Python function that implements a trie data structure with insert, search, and prefix matching. Include type hints and handle edge cases."},
    {"tag": "coding", "text": "Implement a thread-safe LRU cache in Python using collections.OrderedDict and threading.Lock. Support get, put, and a max_size parameter."},
    {"tag": "coding", "text": "Write a recursive descent parser in Python for arithmetic expressions that handles operator precedence, parentheses, and floating point numbers."},
    {"tag": "coding", "text": "Implement the A* pathfinding algorithm in Python for a 2D grid with obstacles. Include a heuristic function and return the shortest path as a list of coordinates."},
    {"tag": "coding", "text": "Write a Python decorator that retries a function up to N times with exponential backoff, jitter, and configurable exception types."},
    {"tag": "coding", "text": "Implement a basic B-tree with insert and search operations in Python. Support configurable order and handle node splits correctly."},
    {"tag": "coding", "text": "Write a Python async web scraper using aiohttp that crawls a domain up to depth 3, respects robots.txt, and extracts all links. Include rate limiting and error handling."},
    {"tag": "coding", "text": "Implement a simple regex engine in Python that supports '.', '*', '+', '?', and character classes like [a-z]. Use Thompson's construction algorithm."},
    {"tag": "coding", "text": "Write a Python class that implements a skip list with probabilistic balancing, supporting insert, delete, and search operations in O(log n) expected time."},
    {"tag": "coding", "text": "Build a basic SQL query parser in Python that can parse SELECT, FROM, WHERE, JOIN, ORDER BY, and GROUP BY clauses into an AST."},

    # --- Creative / writing (tests language generation routing) ---
    {"tag": "creative", "text": "Write a haiku about programming"},
    {"tag": "creative", "text": "Create a short product description for a wireless Bluetooth speaker"},
    {"tag": "creative", "text": "Write a professional email declining a meeting invitation"},
    {"tag": "creative", "text": "Compose a limerick about machine learning"},
    {"tag": "creative", "text": "Write a 100-word horror story set in a server room"},

    # --- Reasoning / analysis (expect balanced or premium) ---
    {"tag": "reasoning", "text": "A farmer has 100 meters of fencing. What dimensions should he use for a rectangular pen to maximize area? Walk me through the calculus."},
    {"tag": "reasoning", "text": "Compare and contrast microservices architecture vs monolithic architecture. Consider scalability, deployment complexity, debugging, and team structure. Provide a detailed analysis with real-world trade-offs."},
    {"tag": "reasoning", "text": "Explain the CAP theorem with concrete examples of databases that prioritize each pair. Then discuss how modern distributed databases like CockroachDB and Cassandra handle these trade-offs in practice."},
    {"tag": "reasoning", "text": "Analyze the time and space complexity of merge sort vs quicksort. When would you prefer one over the other? Consider cache locality, stability, and worst-case guarantees."},
    {"tag": "reasoning", "text": "Design a rate limiter for a REST API that supports per-user and global limits, uses a sliding window algorithm, and can be distributed across multiple servers. Explain each design decision."},

    # --- Multi-step / agentic (expect premium) ---
    {"tag": "agentic", "text": "I have a PostgreSQL database with tables: users(id, name, email, created_at), orders(id, user_id, total, status, created_at), and items(id, order_id, product_name, price, quantity). Write optimized SQL queries for: 1) Top 10 customers by lifetime value, 2) Monthly revenue trend for the last 12 months, 3) Products frequently bought together, 4) Customers who haven't ordered in 90 days. Then create indexes to support these queries."},
    {"tag": "agentic", "text": "Review this architecture: a React frontend calls a Node.js API gateway, which routes to Python microservices (auth, billing, inventory). Each service has its own PostgreSQL database. Redis is used for caching and Kafka for async events. Identify potential failure modes, suggest improvements for resilience, and design a comprehensive monitoring strategy."},
    {"tag": "agentic", "text": "Design a complete CI/CD pipeline for a monorepo containing 3 Python services and 2 React apps. Include linting, testing, building Docker images, deploying to Kubernetes staging, running integration tests, and promoting to production. Handle rollbacks and canary deployments. Write the GitHub Actions workflow YAML."},
    {"tag": "agentic", "text": "I need to migrate a legacy Django app from Python 2.7 to Python 3.12. The app has 200+ models, uses raw SQL in places, has Celery tasks, and custom middleware. Create a detailed migration plan with phases, risk assessment for each phase, a testing strategy, and rollback procedures."},
    {"tag": "agentic", "text": "Build a complete REST API specification for an e-commerce platform. Include endpoints for: user auth (JWT with refresh tokens), product catalog with search and filters, shopping cart, checkout with Stripe integration, order management, and admin analytics. Define request/response schemas, error codes, pagination, and rate limits. Write it as an OpenAPI 3.0 spec."},

    # --- Math / logic puzzles (mixed difficulty) ---
    {"tag": "reasoning", "text": "Prove that the square root of 2 is irrational. Show every step of the proof by contradiction."},
    {"tag": "reasoning", "text": "You have 12 balls, one is heavier or lighter. Using a balance scale exactly 3 times, find the odd ball and determine if it's heavier or lighter. Explain your full decision tree."},
    {"tag": "simple", "text": "What is 17 times 23?"},
    {"tag": "creative", "text": "Write a satirical job posting for a 'Senior Coffee Retrieval Engineer' at a tech startup. Include absurd requirements like 10 years of experience in a 2-year-old framework."},
    {"tag": "coding", "text": "Implement a complete Red-Black tree in Python with insert, delete, and rebalancing. Include left/right rotations, color flipping, and all fixup cases. Add a method to verify the Red-Black tree properties."},
]

# ── Providers & Strategies ───────────────────────────────────────────────────

PROVIDERS = ["google", "anthropic", "openai"]
STRATEGY_NAMES = list(STRATEGIES.keys())


def tier_for(provider: str, model: str) -> str:
    return PROVIDER_MODELS[provider][model]["tier"]


def price_for(provider: str, model: str) -> str:
    info = PROVIDER_MODELS[provider][model]
    return f"${info['input_price']:.2f}/${info['output_price']:.2f}"


def pick_strategy_winner(
    strategy_name: str,
    per_provider: dict[str, str],
) -> tuple[str, str]:
    """Simulate MultiProvider strategy selection without actual Provider objects.

    Returns (provider_name, model_name).
    """
    if strategy_name == "cheapest":
        best, best_price = None, float("inf")
        for prov, model in per_provider.items():
            info = PROVIDER_MODELS[prov][model]
            price = info["input_price"] + info["output_price"]
            if price < best_price:
                best_price = price
                best = (prov, model)
        return best #type:ignore

    if strategy_name == "smartest":
        best, best_rank, best_price = None, -1, float("inf")
        for prov, model in per_provider.items():
            info = PROVIDER_MODELS[prov][model]
            rank = TIER_RANK.get(info["tier"], 0)
            price = info["input_price"] + info["output_price"]
            if rank > best_rank or (rank == best_rank and price < best_price):
                best_rank = rank
                best_price = price
                best = (prov, model)
        return best #type:ignore

    # balanced
    balanced_opts, all_opts = [], []
    for prov, model in per_provider.items():
        info = PROVIDER_MODELS[prov][model]
        price = info["input_price"] + info["output_price"]
        entry = (price, prov, model)
        all_opts.append(entry)
        if info["tier"] == "balanced":
            balanced_opts.append(entry)
    if balanced_opts:
        balanced_opts.sort()
        _, prov, model = balanced_opts[0]
        return (prov, model)
    all_opts.sort()
    _, prov, model = all_opts[0]
    return (prov, model)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 90)
    print("ORKESTRA STRESS TEST — 50 Prompts × 3 Providers × 3 Strategies")
    print("=" * 90)

    # Load routers
    routers: dict[str, KNNRouter] = {}
    for prov in PROVIDERS:
        t0 = time.perf_counter()
        routers[prov] = KNNRouter(prov)
        elapsed = time.perf_counter() - t0
        print(f"  Loaded {prov} router in {elapsed:.2f}s")
    print()

    # Track stats
    provider_model_counts: dict[str, Counter] = {p: Counter() for p in PROVIDERS}
    provider_tier_counts: dict[str, Counter] = {p: Counter() for p in PROVIDERS}
    tag_tier_counts: dict[str, Counter] = {tag: Counter() for tag in {p["tag"] for p in PROMPTS}}
    strategy_winner_counts: dict[str, Counter] = {s: Counter() for s in STRATEGY_NAMES}

    # ── Per-prompt routing ───────────────────────────────────────────────
    print("-" * 90)
    print(f"{'#':<4} {'Tag':<11} {'Prompt':<40} {'Google':<24} {'Anthropic':<24} {'OpenAI':<20}")
    print("-" * 90)

    for i, entry in enumerate(PROMPTS, 1):
        tag = entry["tag"]
        text = entry["text"]
        short = text[:37] + "..." if len(text) > 40 else text

        per_provider: dict[str, str] = {}
        cols = []
        for prov in PROVIDERS:
            model = routers[prov].route(text)
            per_provider[prov] = model
            t = tier_for(prov, model)
            provider_model_counts[prov][model] += 1
            provider_tier_counts[prov][t] += 1
            tag_tier_counts[tag][t] += 1
            tier_label = {"budget": "bud", "balanced": "bal", "premium": "pre"}[t]
            cols.append(f"{model} ({tier_label})")

        # Track strategy winners
        for strat in STRATEGY_NAMES:
            winner_prov, winner_model = pick_strategy_winner(strat, per_provider)
            strategy_winner_counts[strat][f"{winner_prov}/{winner_model}"] += 1

        print(f"{i:<4} {tag:<11} {short:<40} {cols[0]:<24} {cols[1]:<24} {cols[2]:<20}")

    # ── Summary: Per-provider model distribution ─────────────────────────
    print("\n" + "=" * 90)
    print("PER-PROVIDER MODEL DISTRIBUTION")
    print("=" * 90)
    for prov in PROVIDERS:
        print(f"\n  {prov.upper()}")
        for model, count in provider_model_counts[prov].most_common():
            t = tier_for(prov, model)
            pct = count / len(PROMPTS) * 100
            bar = "#" * int(pct / 2)
            print(f"    {model:<28} {t:<10} {count:>3} ({pct:5.1f}%)  {bar}")

    # ── Summary: Per-provider tier distribution ──────────────────────────
    print("\n" + "=" * 90)
    print("PER-PROVIDER TIER DISTRIBUTION")
    print("=" * 90)
    for prov in PROVIDERS:
        print(f"\n  {prov.upper()}")
        for t in ["budget", "balanced", "premium"]:
            count = provider_tier_counts[prov][t]
            pct = count / len(PROMPTS) * 100
            bar = "#" * int(pct / 2)
            print(f"    {t:<10} {count:>3} ({pct:5.1f}%)  {bar}")

    # ── Summary: Tier distribution by prompt tag ─────────────────────────
    print("\n" + "=" * 90)
    print("TIER DISTRIBUTION BY PROMPT CATEGORY (across all providers)")
    print("=" * 90)
    for tag in ["simple", "moderate", "coding", "creative", "reasoning", "agentic"]:
        total = sum(tag_tier_counts[tag].values())
        print(f"\n  {tag.upper()} ({total // 3} prompts × 3 providers = {total} routings)")
        for t in ["budget", "balanced", "premium"]:
            count = tag_tier_counts[tag][t]
            pct = count / total * 100 if total else 0
            bar = "#" * int(pct / 2)
            print(f"    {t:<10} {count:>3} ({pct:5.1f}%)  {bar}")

    # ── Summary: Strategy winners ────────────────────────────────────────
    print("\n" + "=" * 90)
    print("STRATEGY WINNER DISTRIBUTION (which provider/model wins across 50 prompts)")
    print("=" * 90)
    for strat in STRATEGY_NAMES:
        print(f"\n  Strategy: {strat.upper()}")
        for combo, count in strategy_winner_counts[strat].most_common():
            prov, model = combo.split("/")
            t = tier_for(prov, model)
            pct = count / len(PROMPTS) * 100
            bar = "#" * int(pct / 2)
            print(f"    {combo:<40} {t:<10} {count:>3} ({pct:5.1f}%)  {bar}")

    print("\n" + "=" * 90)
    print(f"Done. Routed {len(PROMPTS)} prompts × {len(PROVIDERS)} providers = {len(PROMPTS) * len(PROVIDERS)} total routings.")
    print("=" * 90)


if __name__ == "__main__":
    main()
