"""Multi-provider selection strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from orkestra.registry.models import PROVIDER_MODELS, TIER_RANK

if TYPE_CHECKING:
    from orkestra.provider import Provider


def cheapest(
    providers: list[Provider],
    selections: dict[Provider, str],
) -> tuple[Provider, str]:
    """Pick the provider+model combo with lowest estimated cost.

    Compares input_price as a proxy (output cost varies with response length).
    """
    best = None
    best_price = float("inf")

    for provider, model in selections.items():
        models = PROVIDER_MODELS[provider._backend.name]
        info = models[model]
        price = info["input_price"] + info["output_price"]
        if price < best_price:
            best_price = price
            best = (provider, model)

    return best #type:ignore


def smartest(
    providers: list[Provider],
    selections: dict[Provider, str],
) -> tuple[Provider, str]:
    """Pick the provider+model combo in the highest tier.

    Among same-tier models, prefer the cheaper one.
    """
    best = None
    best_rank = -1
    best_price = float("inf")

    for provider, model in selections.items():
        models = PROVIDER_MODELS[provider._backend.name]
        info = models[model]
        rank = TIER_RANK.get(info["tier"], 0)
        price = info["input_price"] + info["output_price"]
        if rank > best_rank or (rank == best_rank and price < best_price):
            best_rank = rank
            best_price = price
            best = (provider, model)

    return best #type:ignore


def balanced(
    providers: list[Provider],
    selections: dict[Provider, str],
) -> tuple[Provider, str]:
    """Pick balanced tier, preferring lower cost at same tier.

    Prefers "balanced" tier models. Falls back to cheapest if none available.
    """
    balanced_options = []
    all_options = []

    for provider, model in selections.items():
        models = PROVIDER_MODELS[provider._backend.name]
        info = models[model]
        price = info["input_price"] + info["output_price"]
        entry = (price, provider, model)
        all_options.append(entry)
        if info["tier"] == "balanced":
            balanced_options.append(entry)

    if balanced_options:
        balanced_options.sort()
        _, provider, model = balanced_options[0]
        return (provider, model)

    all_options.sort()
    _, provider, model = all_options[0]
    return (provider, model)


STRATEGIES = {
    "cheapest": cheapest,
    "smartest": smartest,
    "balanced": balanced,
}
