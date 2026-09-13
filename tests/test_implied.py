"""Implied probability from decimal odds."""

import pytest

from app.ml.implied import (
    implied_or_default,
    implied_probabilities,
    overround,
)


def test_overround_reflects_the_bookmaker_margin() -> None:
    # A real 1X2 market prices at 105-108%, not 100%.
    assert overround([1.80, 3.75, 4.33]) == pytest.approx(1.0532, abs=1e-4)


def test_normalised_probabilities_sum_to_one() -> None:
    probabilities = implied_probabilities([1.80, 3.75, 4.33])
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-9)
    assert all(0 < p < 1 for p in probabilities)


def test_normalisation_preserves_the_ordering_of_prices() -> None:
    home, draw, away = implied_probabilities([1.80, 3.75, 4.33])
    assert home > draw > away


def test_two_way_markets_work_without_a_draw() -> None:
    home, away = implied_probabilities([1.55, 2.45])
    assert home + away == pytest.approx(1.0, abs=1e-9)
    assert home > away


def test_incomplete_market_returns_none_not_zeros() -> None:
    """Zeros would tell the model the market rules an outcome out."""
    assert implied_probabilities([1.80, None, 4.33]) is None
    assert implied_probabilities([None, None, None]) is None


def test_degenerate_odds_are_rejected() -> None:
    assert implied_probabilities([1.0, 3.75, 4.33]) is None
    assert implied_probabilities([0.0, 3.75, 4.33]) is None


def test_fallback_supplies_base_rates_when_odds_are_missing() -> None:
    defaults = (0.46, 0.26, 0.28)
    assert implied_or_default([None, None, None], defaults) == list(defaults)
    assert implied_or_default([1.80, 3.75, 4.33], defaults) != list(defaults)
