"""Sport adapters and lookup."""

import pytest
from fastapi import HTTPException

from app.sports import SportAdapter
from app.sports.registry import available_sports, get_adapter


def test_both_sports_are_registered() -> None:
    assert available_sports() == ["basketball", "football"]


@pytest.mark.parametrize("sport", ["football", "basketball"])
def test_adapters_satisfy_the_protocol(sport: str) -> None:
    adapter = get_adapter(sport)
    assert isinstance(adapter, SportAdapter)
    assert adapter.feature_columns and adapter.targets
    assert adapter.min_team_history > 0


def test_lookup_is_case_insensitive_and_trims() -> None:
    assert get_adapter("  FootBall ").name == "football"


def test_unknown_sport_is_a_404_listing_what_exists() -> None:
    with pytest.raises(HTTPException) as excinfo:
        get_adapter("tennis")
    assert excinfo.value.status_code == 404
    assert "football" in excinfo.value.detail


def test_only_football_has_a_draw() -> None:
    assert get_adapter("football").has_draw is True
    assert get_adapter("basketball").has_draw is False


def test_sports_do_not_share_feature_columns() -> None:
    """A model trained for one sport must never be fed the other's rows."""
    football = set(get_adapter("football").feature_columns)
    basketball = set(get_adapter("basketball").feature_columns)
    assert football != basketball
    assert "home_goals_scored_avg" in football and "home_goals_scored_avg" not in basketball
    assert "home_back_to_back" in basketball and "home_back_to_back" not in football


@pytest.mark.parametrize(
    "sport,market,selection,home,away,expected",
    [
        ("football", "home_win", "1", 2, 0, "won"),
        ("football", "draws", "X", 1, 1, "won"),
        ("basketball", "moneyline", "1", 110, 104, "won"),
        ("basketball", "moneyline", "1", 99, 104, "lost"),
        ("basketball", "moneyline", "2", 99, 104, "won"),
        ("basketball", "total_points", "over", 130, 120, "won"),
        ("basketball", "total_points", "under", 100, 99, "won"),
        ("basketball", "unknown", "?", 110, 100, "void"),
    ],
)
def test_settlement_is_per_sport(sport, market, selection, home, away, expected) -> None:
    assert get_adapter(sport).settle(market, selection, home, away) == expected


def test_football_markets_are_void_for_basketball() -> None:
    """Football selections must not silently settle under basketball rules."""
    assert get_adapter("basketball").settle("btts", "GG", 110, 100) == "void"
