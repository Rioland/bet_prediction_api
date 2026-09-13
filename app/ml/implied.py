"""Implied probability from decimal odds.

Raw 1/odds does not sum to 1: bookmakers build in a margin (the overround), so
the three 1X2 prices typically imply 105-108%. Feeding un-normalised values to
a model teaches it the margin along with the signal, and makes "implied
probability" mean different things in different leagues.

Normalising by the overround removes most of it. This is the simple
proportional method; it slightly overstates favourites, because margin is not
spread evenly across outcomes, but it is unbiased enough for a feature and
needs no fitting.
"""

from __future__ import annotations

from typing import Sequence

# Odds at or below this are not a real market (1.0 implies certainty).
MIN_VALID_ODDS = 1.01


def _valid(odds: float | None) -> bool:
    return odds is not None and odds >= MIN_VALID_ODDS


def overround(odds: Sequence[float | None]) -> float | None:
    """Total implied probability across a complete market. ~1.05 is typical."""
    prices = [o for o in odds if _valid(o)]
    if len(prices) != len(odds) or not prices:
        return None
    return sum(1 / o for o in prices)


def implied_probabilities(odds: Sequence[float | None]) -> list[float] | None:
    """Margin-adjusted probabilities that sum to 1, or None if the market is incomplete.

    Returning None rather than zeros matters: a fixture with no odds has
    *unknown* market probabilities, and zeros would read as "the market thinks
    this cannot happen".
    """
    total = overround(odds)
    if total is None or total <= 0:
        return None
    return [(1 / o) / total for o in odds]  # type: ignore[operator]


def implied_or_default(
    odds: Sequence[float | None], defaults: Sequence[float]
) -> list[float]:
    """Implied probabilities, falling back to sport-level base rates.

    The fallback is what stops missing odds from silently becoming a strong
    signal; ``market_available`` records which case applied so the model can
    tell them apart.
    """
    probabilities = implied_probabilities(odds)
    return list(probabilities) if probabilities is not None else list(defaults)
