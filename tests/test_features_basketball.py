"""Basketball features, centred on the same no-leakage guarantee as football."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from app.ml.features_basketball import FEATURE_COLUMNS, build_dataset, features_for_upcoming

BASE = datetime(2026, 1, 1, 19, 0)


def _game(idx: int, home: int, away: int, hs: int, aws: int, *, hours: int = 48, **extra) -> dict:
    return {
        "id": idx,
        "league_id": 1,
        "home_team_id": home,
        "away_team_id": away,
        "kickoff_time": BASE + timedelta(hours=hours * idx),
        "home_goals": hs,
        "away_goals": aws,
        **extra,
    }


def _season() -> list[dict]:
    return [
        _game(0, 1, 2, 112, 104),
        _game(1, 3, 1, 98, 121),
        _game(2, 2, 3, 130, 118),
        _game(3, 1, 3, 105, 107),
        _game(4, 2, 1, 99, 115),
        _game(5, 3, 2, 122, 122 - 5),
    ]


def test_features_do_not_depend_on_future_games() -> None:
    original = build_dataset(_season())

    tampered = _season()
    tampered[-1]["home_goals"] = 180
    rebuilt = build_dataset(tampered)

    pd.testing.assert_frame_equal(
        original[FEATURE_COLUMNS].iloc[:-1].reset_index(drop=True),
        rebuilt[FEATURE_COLUMNS].iloc[:-1].reset_index(drop=True),
    )


def test_appending_a_future_game_leaves_history_untouched() -> None:
    base = build_dataset(_season())
    extended = build_dataset(_season() + [_game(6, 1, 2, 100, 99)])
    pd.testing.assert_frame_equal(
        base[FEATURE_COLUMNS],
        extended[FEATURE_COLUMNS].iloc[: len(base)].reset_index(drop=True),
    )


def test_there_is_no_draw_class() -> None:
    """Basketball games do not end level, so the target is binary."""
    df = build_dataset(_season())
    assert set(df["match_winner"].unique()) <= {"H", "A"}


def test_winner_label_follows_the_score() -> None:
    df = build_dataset([_game(0, 1, 2, 120, 100), _game(1, 1, 2, 95, 110)])
    assert list(df["match_winner"]) == ["H", "A"]


def test_totals_label_uses_the_combined_score() -> None:
    df = build_dataset([_game(0, 1, 2, 130, 120), _game(1, 1, 2, 95, 96)])
    assert list(df["total_points_over"]) == ["over", "under"]


def test_back_to_back_flag_fires_on_short_rest() -> None:
    games = [
        _game(0, 1, 2, 110, 100, hours=48),
        {**_game(1, 1, 3, 105, 101), "kickoff_time": BASE + timedelta(hours=24)},
    ]
    df = build_dataset(games)
    second = df.iloc[1]
    assert second["home_back_to_back"] == 1.0
    assert second["home_rest_days"] == pytest.approx(1.0)


def test_rested_team_is_not_flagged() -> None:
    df = build_dataset(_season())
    assert (df["home_back_to_back"] == 0.0).all(), "48h apart is not a back-to-back"


def test_first_appearance_uses_neutral_priors() -> None:
    first = build_dataset(_season()).iloc[0]
    assert first["home_matches_played"] == 0
    assert first["away_matches_played"] == 0
    assert first["home_win_rate"] == first["away_win_rate"]


def test_market_columns_reflect_available_odds() -> None:
    with_odds = build_dataset([_game(0, 1, 2, 110, 100, odds_home=1.55, odds_away=2.45)])
    row = with_odds.iloc[0]
    assert row["market_available"] == 1.0
    assert row["market_home_prob"] + row["market_away_prob"] == pytest.approx(1.0, abs=1e-9)
    assert row["market_home_prob"] > row["market_away_prob"]


def test_missing_odds_fall_back_and_are_flagged() -> None:
    row = build_dataset(_season()).iloc[0]
    assert row["market_available"] == 0.0
    assert row["market_home_prob"] == pytest.approx(0.58)


def test_upcoming_games_get_features_but_no_labels() -> None:
    frame = features_for_upcoming(
        _season(), [{"id": 99, "home_team_id": 1, "away_team_id": 2,
                     "kickoff_time": BASE + timedelta(days=30), "league_id": 1}]
    )
    assert len(frame) == 1
    assert "match_winner" not in frame.columns
    assert frame.iloc[0]["home_matches_played"] > 0


def test_unplayed_games_are_excluded_from_training_rows() -> None:
    games = _season() + [_game(6, 1, 2, None, None)]
    assert len(build_dataset(games)) == len(_season())
