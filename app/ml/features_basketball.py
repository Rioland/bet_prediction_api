"""Leak-free basketball features.

Same guarantee as the football builder - a row may only reflect matches that
started before it - but the sport is different enough to need its own:

* No draw. The outcome is binary, so there are two classes, not three.
* Scores are ~110 per side rather than ~1.4, so goal-style averages are
  replaced by points for/against and a points differential that actually
  separates teams.
* Schedule density matters far more. Back-to-back games measurably depress
  performance, so rest days carry a dedicated flag rather than just a number.
* No Poisson scoreline model: basketball totals are near-normal, not Poisson,
  so nothing here derives a correct-score distribution.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

import pandas as pd

from app.ml.implied import implied_or_default, implied_probabilities

FORM_WINDOW = 10
H2H_WINDOW = 10

# League-average points per team per game, used before a team has history.
DEFAULT_POINTS = 112.0
DEFAULT_WIN_RATE = 0.5
# Two-way markets have no draw.
BASE_RATE_MONEYLINE = (0.58, 0.42)  # home sides win about 58% of the time
# Typical combined total, for the over/under label.
DEFAULT_TOTAL_LINE = 224.5
# A game within this many days of the last one is a back-to-back.
BACK_TO_BACK_DAYS = 1.5

FEATURE_COLUMNS = [
    "home_win_rate",
    "away_win_rate",
    "home_venue_win_rate",
    "away_venue_win_rate",
    "home_points_for_avg",
    "away_points_for_avg",
    "home_points_against_avg",
    "away_points_against_avg",
    "home_point_diff_avg",
    "away_point_diff_avg",
    "h2h_home_wins",
    "h2h_away_wins",
    "home_rest_days",
    "away_rest_days",
    "home_back_to_back",
    "away_back_to_back",
    "home_matches_played",
    "away_matches_played",
    "market_home_prob",
    "market_away_prob",
    "market_available",
]

TARGETS = ["match_winner", "total_points_over"]


@dataclass
class _TeamHistory:
    results: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    home_results: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    away_results: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    last_played: datetime | None = None
    played: int = 0


def _mean(values: Iterable[float], default: float) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else default


def _win_rate(results: deque, default: float = DEFAULT_WIN_RATE) -> float:
    return _mean([r["won"] for r in results], default)


def _rest_days(history: _TeamHistory, tipoff: datetime) -> float:
    if history.last_played is None:
        return 3.0
    return min((tipoff - history.last_played).total_seconds() / 86400.0, 14.0)


def _labels(home_score: int, away_score: int, line: float = DEFAULT_TOTAL_LINE) -> dict[str, str]:
    return {
        "match_winner": "H" if home_score > away_score else "A",
        "total_points_over": "over" if home_score + away_score > line else "under",
    }


def build_feature_row(
    home: _TeamHistory, away: _TeamHistory, h2h: deque, match: dict[str, Any]
) -> dict[str, float]:
    tipoff = match["kickoff_time"]
    h2h_home = sum(1 for m in h2h if m["winner"] == match["home_team_id"])
    h2h_away = sum(1 for m in h2h if m["winner"] == match["away_team_id"])

    prices = (match.get("odds_home"), match.get("odds_away"))
    market = implied_or_default(prices, BASE_RATE_MONEYLINE)
    home_rest = _rest_days(home, tipoff)
    away_rest = _rest_days(away, tipoff)

    return {
        "home_win_rate": _win_rate(home.results),
        "away_win_rate": _win_rate(away.results),
        "home_venue_win_rate": _win_rate(home.home_results),
        "away_venue_win_rate": _win_rate(away.away_results),
        "home_points_for_avg": _mean([r["scored"] for r in home.results], DEFAULT_POINTS),
        "away_points_for_avg": _mean([r["scored"] for r in away.results], DEFAULT_POINTS),
        "home_points_against_avg": _mean([r["conceded"] for r in home.results], DEFAULT_POINTS),
        "away_points_against_avg": _mean([r["conceded"] for r in away.results], DEFAULT_POINTS),
        "home_point_diff_avg": _mean([r["scored"] - r["conceded"] for r in home.results], 0.0),
        "away_point_diff_avg": _mean([r["scored"] - r["conceded"] for r in away.results], 0.0),
        "h2h_home_wins": float(h2h_home),
        "h2h_away_wins": float(h2h_away),
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "home_back_to_back": 1.0 if home_rest <= BACK_TO_BACK_DAYS else 0.0,
        "away_back_to_back": 1.0 if away_rest <= BACK_TO_BACK_DAYS else 0.0,
        "home_matches_played": float(home.played),
        "away_matches_played": float(away.played),
        "market_home_prob": market[0],
        "market_away_prob": market[1],
        "market_available": 1.0 if implied_probabilities(prices) is not None else 0.0,
    }


def _record(
    history: _TeamHistory, *, scored: int, conceded: int, at_home: bool, tipoff: datetime
) -> None:
    entry = {"won": 1.0 if scored > conceded else 0.0, "scored": scored, "conceded": conceded}
    history.results.append(entry)
    (history.home_results if at_home else history.away_results).append(entry)
    history.last_played = tipoff
    history.played += 1


def build_dataset(matches: list[dict[str, Any]]) -> pd.DataFrame:
    """Training rows from completed games, in chronological order."""
    ordered = sorted(matches, key=lambda m: m["kickoff_time"])
    histories: dict[int, _TeamHistory] = defaultdict(_TeamHistory)
    h2h_log: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))
    rows: list[dict[str, Any]] = []

    for match in ordered:
        home_id, away_id = match["home_team_id"], match["away_team_id"]
        home_score, away_score = match.get("home_goals"), match.get("away_goals")
        pair = (min(home_id, away_id), max(home_id, away_id))

        features = build_feature_row(histories[home_id], histories[away_id], h2h_log[pair], match)
        if home_score is None or away_score is None:
            continue

        rows.append({
            **features,
            **_labels(home_score, away_score),
            "match_id": match.get("id"),
            "kickoff_time": match["kickoff_time"],
            "league_id": match.get("league_id"),
            "odds_home": match.get("odds_home"),
            "odds_away": match.get("odds_away"),
        })

        tipoff = match["kickoff_time"]
        _record(histories[home_id], scored=home_score, conceded=away_score, at_home=True, tipoff=tipoff)
        _record(histories[away_id], scored=away_score, conceded=home_score, at_home=False, tipoff=tipoff)
        h2h_log[pair].append({"winner": home_id if home_score > away_score else away_id})

    return pd.DataFrame(rows)


def features_for_upcoming(
    finished: list[dict[str, Any]], upcoming: list[dict[str, Any]]
) -> pd.DataFrame:
    """Feature rows for games that have not tipped off."""
    histories: dict[int, _TeamHistory] = defaultdict(_TeamHistory)
    h2h_log: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))

    for match in sorted(finished, key=lambda m: m["kickoff_time"]):
        home_score, away_score = match.get("home_goals"), match.get("away_goals")
        if home_score is None or away_score is None:
            continue
        home_id, away_id = match["home_team_id"], match["away_team_id"]
        tipoff = match["kickoff_time"]
        _record(histories[home_id], scored=home_score, conceded=away_score, at_home=True, tipoff=tipoff)
        _record(histories[away_id], scored=away_score, conceded=home_score, at_home=False, tipoff=tipoff)
        h2h_log[(min(home_id, away_id), max(home_id, away_id))].append(
            {"winner": home_id if home_score > away_score else away_id}
        )

    rows = []
    for match in upcoming:
        home_id, away_id = match["home_team_id"], match["away_team_id"]
        pair = (min(home_id, away_id), max(home_id, away_id))
        rows.append({
            **build_feature_row(histories[home_id], histories[away_id], h2h_log[pair], match),
            "match_id": match.get("id"),
            "kickoff_time": match["kickoff_time"],
        })
    return pd.DataFrame(rows)
