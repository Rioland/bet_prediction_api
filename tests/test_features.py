"""Feature-engineering tests, centred on the no-leakage guarantee."""

from datetime import datetime, timedelta

import pandas as pd

from app.ml.features import FEATURE_COLUMNS, build_dataset

BASE = datetime(2024, 1, 1)


def _match(idx: int, home: int, away: int, hg: int, ag: int, **extra) -> dict:
    return {
        "id": idx,
        "league_id": 1,
        "season": 2024,
        "home_team_id": home,
        "away_team_id": away,
        "kickoff_time": BASE + timedelta(days=idx),
        "home_goals": hg,
        "away_goals": ag,
        **extra,
    }


def _fixtures() -> list[dict]:
    return [
        _match(0, 1, 2, 2, 0),
        _match(1, 3, 1, 1, 1),
        _match(2, 2, 3, 0, 3),
        _match(3, 1, 3, 2, 1),
        _match(4, 2, 1, 1, 2),
        _match(5, 3, 2, 2, 2),
    ]


def test_first_appearance_uses_neutral_priors() -> None:
    df = build_dataset(_fixtures())
    first = df.iloc[0]
    # Neither team has history, so both sides get identical neutral values.
    assert first["home_matches_played"] == 0
    assert first["away_matches_played"] == 0
    assert first["home_ppg"] == first["away_ppg"]


def test_features_do_not_depend_on_future_matches() -> None:
    """The core guarantee: rewriting a later result must not move earlier rows."""
    original = build_dataset(_fixtures())

    tampered_fixtures = _fixtures()
    # Flip the last match to a wildly different scoreline.
    tampered_fixtures[-1]["home_goals"] = 9
    tampered_fixtures[-1]["away_goals"] = 0
    tampered = build_dataset(tampered_fixtures)

    earlier = original[FEATURE_COLUMNS].iloc[:-1].reset_index(drop=True)
    earlier_tampered = tampered[FEATURE_COLUMNS].iloc[:-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(earlier, earlier_tampered)


def test_appending_a_future_match_leaves_history_untouched() -> None:
    base = build_dataset(_fixtures())
    extended = build_dataset(_fixtures() + [_match(6, 1, 2, 4, 4)])

    pd.testing.assert_frame_equal(
        base[FEATURE_COLUMNS], extended[FEATURE_COLUMNS].iloc[: len(base)].reset_index(drop=True)
    )


def test_history_accumulates_in_chronological_order() -> None:
    df = build_dataset(_fixtures())
    # Team 1 plays in matches 0, 1, 3, 4 - by its fourth appearance it has 3 played.
    row = df[df["match_id"] == 4].iloc[0]
    assert row["away_matches_played"] == 3  # team 1 is away in match 4


def test_labels_are_derived_correctly() -> None:
    df = build_dataset([_match(0, 1, 2, 2, 0), _match(1, 1, 2, 1, 1), _match(2, 1, 2, 0, 3)])
    assert list(df["match_winner"]) == ["H", "D", "A"]
    assert list(df["btts"]) == ["no", "yes", "no"]
    assert list(df["over_under_2_5"]) == ["under", "under", "over"]


def test_unplayed_matches_are_excluded() -> None:
    fixtures = _fixtures() + [_match(6, 1, 2, None, None)]
    assert len(build_dataset(fixtures)) == len(_fixtures())


def test_rest_days_reflect_gap_since_last_fixture() -> None:
    df = build_dataset(_fixtures())
    row = df[df["match_id"] == 3].iloc[0]
    # Team 1 last played on day 1; match 3 is on day 3.
    assert row["home_rest_days"] == 2.0
