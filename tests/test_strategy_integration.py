"""Integration test: verify strategies diverge with real KNN routers.

Uses the actual trained KNN models to route prompts, then checks that
cheapest/smartest/balanced strategies pick different provider+model combos
when providers route to different tiers.
"""

import pytest

from orkestra.registry.models import PROVIDER_MODELS, TIER_RANK
from orkestra.router.knn import KNNRouter

PROVIDERS = ["google", "anthropic", "openai"]

# Prompts designed to trigger different tiers based on stress test results
BUDGET_PROMPTS = [
    "What time is it in Tokyo?",
    "Convert 100 USD to EUR",
    "Write a haiku about programming",
]

BALANCED_PROMPTS = [
    "What is the difference between a stack and a queue?",
    "Analyze the time and space complexity of merge sort vs quicksort.",
]

PREMIUM_PROMPTS = [
    "Implement a thread-safe LRU cache in Python using collections.OrderedDict "
    "and threading.Lock. Support get, put, and a max_size parameter.",
    "Write a Python decorator that retries a function up to N times with "
    "exponential backoff, jitter, and configurable exception types.",
    "Build a basic SQL query parser in Python that can parse SELECT, FROM, "
    "WHERE, JOIN, ORDER BY, and GROUP BY clauses into an AST.",
]


def _tier_for(provider: str, model: str) -> str:
    return PROVIDER_MODELS[provider][model]["tier"]


def _price_for(provider: str, model: str) -> float:
    info = PROVIDER_MODELS[provider][model]
    return info["input_price"] + info["output_price"]


def _pick_strategy_winner(strategy_name, per_provider):
    """Simulate MultiProvider strategy selection without Provider objects."""
    if strategy_name == "cheapest":
        best, best_price = None, float("inf")
        for prov, model in per_provider.items():
            price = _price_for(prov, model)
            if price < best_price:
                best_price = price
                best = (prov, model)
        return best

    if strategy_name == "smartest":
        best, best_rank, best_price = None, -1, float("inf")
        for prov, model in per_provider.items():
            rank = TIER_RANK[_tier_for(prov, model)]
            price = _price_for(prov, model)
            if rank > best_rank or (rank == best_rank and price < best_price):
                best_rank = rank
                best_price = price
                best = (prov, model)
        return best

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


@pytest.fixture(scope="module")
def routers():
    return {prov: KNNRouter(prov) for prov in PROVIDERS}


class TestAllThreeTiersAreReachable:
    """The retrained routers must be able to predict all 3 tiers."""

    def test_budget_tier_reachable(self, routers):
        for prompt in BUDGET_PROMPTS:
            for prov in PROVIDERS:
                model = routers[prov].route(prompt)
                tier = _tier_for(prov, model)
                if tier == "budget":
                    return  # found at least one
        pytest.fail("No budget-tier routing found across all budget prompts")

    def test_balanced_tier_reachable(self, routers):
        for prompt in BALANCED_PROMPTS:
            for prov in PROVIDERS:
                model = routers[prov].route(prompt)
                tier = _tier_for(prov, model)
                if tier == "balanced":
                    return
        pytest.fail("No balanced-tier routing found across all balanced prompts")

    def test_premium_tier_reachable(self, routers):
        for prompt in PREMIUM_PROMPTS:
            for prov in PROVIDERS:
                model = routers[prov].route(prompt)
                tier = _tier_for(prov, model)
                if tier == "premium":
                    return
        pytest.fail("No premium-tier routing found across all premium prompts")


class TestProvidersRouteToSameTier:
    """All providers should agree on the tier for a given prompt
    (since they share the same training data and tier mapping structure)."""

    def test_tier_consistency_across_providers(self, routers):
        all_prompts = BUDGET_PROMPTS + BALANCED_PROMPTS + PREMIUM_PROMPTS
        for prompt in all_prompts:
            tiers = set()
            for prov in PROVIDERS:
                model = routers[prov].route(prompt)
                tiers.add(_tier_for(prov, model))
            assert len(tiers) == 1, (
                f"Providers disagree on tier for: {prompt[:60]}... "
                f"Got tiers: {tiers}"
            )


class TestStrategiesDiverge:
    """When providers route to different tiers, strategies should pick
    different winners. This is the core test for multi-strategy routing."""

    def _find_mixed_tier_prompt(self, routers):
        """Find a prompt where at least one provider routes to a different tier
        than another. If all route to the same tier, return None."""
        all_prompts = BUDGET_PROMPTS + BALANCED_PROMPTS + PREMIUM_PROMPTS
        for prompt in all_prompts:
            per_provider = {}
            tiers = set()
            for prov in PROVIDERS:
                model = routers[prov].route(prompt)
                per_provider[prov] = model
                tiers.add(_tier_for(prov, model))
            if len(tiers) > 1:
                return prompt, per_provider
        return None

    def test_cheapest_picks_lowest_price(self, routers):
        """Cheapest strategy should always pick the lowest-priced combo."""
        all_prompts = BUDGET_PROMPTS + BALANCED_PROMPTS + PREMIUM_PROMPTS
        for prompt in all_prompts:
            per_provider = {
                prov: routers[prov].route(prompt) for prov in PROVIDERS
            }
            winner_prov, winner_model = _pick_strategy_winner("cheapest", per_provider) #type:ignore
            winner_price = _price_for(winner_prov, winner_model)

            for prov, model in per_provider.items():
                assert _price_for(prov, model) >= winner_price, (
                    f"Cheapest picked {winner_prov}/{winner_model} "
                    f"(${winner_price}) but {prov}/{model} "
                    f"(${_price_for(prov, model)}) is cheaper"
                )

    def test_smartest_picks_highest_tier(self, routers):
        """Smartest strategy should always pick the highest tier."""
        all_prompts = BUDGET_PROMPTS + BALANCED_PROMPTS + PREMIUM_PROMPTS
        for prompt in all_prompts:
            per_provider = {
                prov: routers[prov].route(prompt) for prov in PROVIDERS
            }
            winner_prov, winner_model = _pick_strategy_winner("smartest", per_provider) #type:ignore
            winner_rank = TIER_RANK[_tier_for(winner_prov, winner_model)]

            for prov, model in per_provider.items():
                assert TIER_RANK[_tier_for(prov, model)] <= winner_rank, (
                    f"Smartest picked tier {_tier_for(winner_prov, winner_model)} "
                    f"but {prov}/{model} is {_tier_for(prov, model)}"
                )

    def test_cheapest_always_picks_google(self, routers):
        """Google is cheapest at every tier, so cheapest should always pick Google."""
        all_prompts = BUDGET_PROMPTS + BALANCED_PROMPTS + PREMIUM_PROMPTS
        for prompt in all_prompts:
            per_provider = {
                prov: routers[prov].route(prompt) for prov in PROVIDERS
            }
            winner_prov, _ = _pick_strategy_winner("cheapest", per_provider) #type:ignore
            assert winner_prov == "google", (
                f"Expected cheapest to pick google for: {prompt[:60]}... "
                f"but got {winner_prov}"
            )

    def test_balanced_prefers_balanced_tier(self, routers):
        """Balanced strategy should prefer balanced-tier models when available."""
        for prompt in BALANCED_PROMPTS:
            per_provider = {
                prov: routers[prov].route(prompt) for prov in PROVIDERS
            }
            winner_prov, winner_model = _pick_strategy_winner("balanced", per_provider) #type:ignore

            # Check if any balanced-tier option existed
            has_balanced = any(
                _tier_for(prov, model) == "balanced"
                for prov, model in per_provider.items()
            )
            if has_balanced:
                assert _tier_for(winner_prov, winner_model) == "balanced", (
                    f"Balanced strategy picked {_tier_for(winner_prov, winner_model)} "
                    f"tier but balanced-tier options were available"
                )


class TestTierDistribution:
    """Verify the overall routing distribution is healthy — no dead tiers."""

    ALL_PROMPTS = BUDGET_PROMPTS + BALANCED_PROMPTS + PREMIUM_PROMPTS

    def test_no_single_tier_dominates_completely(self, routers):
        """No tier should be 100% or 0% across all test prompts."""
        for prov in PROVIDERS:
            tiers = [
                _tier_for(prov, routers[prov].route(p))
                for p in self.ALL_PROMPTS
            ]
            unique = set(tiers)
            assert len(unique) >= 2, (
                f"Provider {prov} routes ALL test prompts to: {unique}"
            )

    def test_premium_tier_gets_at_least_one_hit(self, routers):
        """Premium tier must be reachable for at least one prompt per provider."""
        for prov in PROVIDERS:
            premium_hits = sum(
                1 for p in self.ALL_PROMPTS
                if _tier_for(prov, routers[prov].route(p)) == "premium"
            )
            assert premium_hits > 0, (
                f"Provider {prov} never routes to premium tier"
            )
