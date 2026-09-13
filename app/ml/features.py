"""Leak-free feature engineering.

The single rule this module enforces: every feature for match M is derived only
from matches that kicked off strictly before M. Features are built by walking
fixtures in chronological order and updating per-team history *after* emitting
each row, so a match can never see its own result or any future match.

This is why the post-match columns on ``Match`` (possession, shots on target)
are safe to use here - they enter the feature set as a team's rolling average
over *previous* fixtures, never as values from the match being predicted.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

import pandas as pd

# How many recent fixtures feed a team's rolling form.
FORM_WINDOW = 10
# Prior meetings considered for head-to-head.
H2H_WINDOW = 10
# Neutral prior used before a team has any history, so early-season rows are not
# silently filled with zeros that the model would read as "very bad team".
DEFAULT_GOALS = 1.3
DEFAULT_PPG = 1.35

FEATURE_COLUMNS = [
    "home_ppg",
    "away_ppg",
    "home_venue_ppg",
    "away_venue_ppg",
    "home_goals_scored_avg",
    "away_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
    "home_goal_diff_avg",
    "away_goal_diff_avg",
    "home_shots_on_target_avg",
    "away_shots_on_target_avg",
    "home_possession_avg",
    "away_possession_avg",
    "home_season_ppg",
    "away_season_ppg",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "home_rest_days",
    "away_rest_days",
    "home_matches_played",
    "away_matches_played",
]

TARGETS = ["match_winner", "over_under_2_5", "btts"]


@dataclass
class _TeamHistory:
    """Rolling record of a single team's completed fixtures."""

    results: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    home_results: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    away_results: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    season_points: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    last_played: datetime | None = None
    played: int = 0


def _points(scored: int, conceded: int) -> int:
    if scored > conceded:
        return 3
    return 1 if scored == conceded else 0


def _mean(values: Iterable[float], default: float) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else default


def _ppg(results: deque, default: float = DEFAULT_PPG) -> float:
    return _mean([r["points"] for r in results], default)


def _rest_days(history: _TeamHistory, kickoff: datetime) -> float:
    if history.last_played is None:
        return 7.0  # neutral assumption for a team's first observed fixture
    return min((kickoff - history.last_played).total_seconds() / 86400.0, 30.0)


def _h2h_counts(meetings: deque, home_id: int) -> tuple[int, int, int]:
    home_wins = draws = away_wins = 0
    for meeting in meetings:
        if meeting["winner"] is None:
            draws += 1
        elif meeting["winner"] == home_id:
            home_wins += 1
        else:
            away_wins += 1
    return home_wins, draws, away_wins


def _labels(home_goals: int, away_goals: int) -> dict[str, str]:
    if home_goals > away_goals:
        winner = "H"
    elif home_goals == away_goals:
        winner = "D"
    else:
        winner = "A"
    return {
        "match_winner": winner,
        "over_under_2_5": "over" if home_goals + away_goals > 2.5 else "under",
        "btts": "yes" if home_goals > 0 and away_goals > 0 else "no",
    }


def build_feature_row(
    home: _TeamHistory, away: _TeamHistory, h2h: deque, match: dict[str, Any]
) -> dict[str, float]:
    """Features for one fixture from history known before its kickoff."""
    kickoff = match["kickoff_time"]
    season = match.get("season") or 0
    h2h_home, h2h_draw, h2h_away = _h2h_counts(h2h, match["home_team_id"])

    return {
        "home_ppg": _ppg(home.results),
        "away_ppg": _ppg(away.results),
        "home_venue_ppg": _ppg(home.home_results),
        "away_venue_ppg": _ppg(away.away_results),
        "home_goals_scored_avg": _mean([r["scored"] for r in home.results], DEFAULT_GOALS),
        "away_goals_scored_avg": _mean([r["scored"] for r in away.results], DEFAULT_GOALS),
        "home_goals_conceded_avg": _mean([r["conceded"] for r in home.results], DEFAULT_GOALS),
        "away_goals_conceded_avg": _mean([r["conceded"] for r in away.results], DEFAULT_GOALS),
        "home_goal_diff_avg": _mean([r["scored"] - r["conceded"] for r in home.results], 0.0),
        "away_goal_diff_avg": _mean([r["scored"] - r["conceded"] for r in away.results], 0.0),
        "home_shots_on_target_avg": _mean([r["shots"] for r in home.results], 4.5),
        "away_shots_on_target_avg": _mean([r["shots"] for r in away.results], 4.5),
        "home_possession_avg": _mean([r["possession"] for r in home.results], 50.0),
        "away_possession_avg": _mean([r["possession"] for r in away.results], 50.0),
        "home_season_ppg": _mean(home.season_points.get(season, []), DEFAULT_PPG),
        "away_season_ppg": _mean(away.season_points.get(season, []), DEFAULT_PPG),
        "h2h_home_wins": float(h2h_home),
        "h2h_draws": float(h2h_draw),
        "h2h_away_wins": float(h2h_away),
        "home_rest_days": _rest_days(home, kickoff),
        "away_rest_days": _rest_days(away, kickoff),
        "home_matches_played": float(home.played),
        "away_matches_played": float(away.played),
    }


def _record(
    history: _TeamHistory,
    *,
    scored: int,
    conceded: int,
    shots: float | None,
    possession: float | None,
    at_home: bool,
    season: int,
    kickoff: datetime,
) -> None:
    entry = {
        "points": _points(scored, conceded),
        "scored": scored,
        "conceded": conceded,
        "shots": shots,
        "possession": possession,
    }
    history.results.append(entry)
    (history.home_results if at_home else history.away_results).append(entry)
    history.season_points[season].append(entry["points"])
    history.last_played = kickoff
    history.played += 1


def build_dataset(matches: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a training frame from chronologically ordered finished matches.

    Each row's features reflect only what was knowable before that kickoff.
    """
    ordered = sorted(matches, key=lambda m: m["kickoff_time"])
    histories: dict[int, _TeamHistory] = defaultdict(_TeamHistory)
    h2h_log: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))
    rows: list[dict[str, Any]] = []

    for match in ordered:
        home_id, away_id = match["home_team_id"], match["away_team_id"]
        home_goals, away_goals = match.get("home_goals"), match.get("away_goals")
        pair = (min(home_id, away_id), max(home_id, away_id))

        home_hist, away_hist = histories[home_id], histories[away_id]
        features = build_feature_row(home_hist, away_hist, h2h_log[pair], match)

        # Unplayed fixtures still yield features (for live prediction) but no labels.
        if home_goals is not None and away_goals is not None:
            row = {
                **features,
                **_labels(home_goals, away_goals),
                "match_id": match.get("id"),
                "kickoff_time": match["kickoff_time"],
                "league_id": match.get("league_id"),
                "odds_home": match.get("odds_home"),
                "odds_draw": match.get("odds_draw"),
                "odds_away": match.get("odds_away"),
            }
            rows.append(row)

            season = match.get("season") or 0
            kickoff = match["kickoff_time"]
            _record(
                home_hist,
                scored=home_goals,
                conceded=away_goals,
                shots=match.get("home_shots_on_target"),
                possession=match.get("home_possession"),
                at_home=True,
                season=season,
                kickoff=kickoff,
            )
            _record(
                away_hist,
                scored=away_goals,
                conceded=home_goals,
                shots=match.get("away_shots_on_target"),
                possession=match.get("away_possession"),
                at_home=False,
                season=season,
                kickoff=kickoff,
            )
            winner = home_id if home_goals > away_goals else (away_id if away_goals > home_goals else None)
            h2h_log[pair].append({"winner": winner})

    return pd.DataFrame(rows)


def features_for_upcoming(
    finished: list[dict[str, Any]], upcoming: list[dict[str, Any]]
) -> pd.DataFrame:
    """Feature rows for fixtures that have not been played yet."""
    histories: dict[int, _TeamHistory] = defaultdict(_TeamHistory)
    h2h_log: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))

    for match in sorted(finished, key=lambda m: m["kickoff_time"]):
        hg, ag = match.get("home_goals"), match.get("away_goals")
        if hg is None or ag is None:
            continue
        home_id, away_id = match["home_team_id"], match["away_team_id"]
        season, kickoff = match.get("season") or 0, match["kickoff_time"]
        _record(histories[home_id], scored=hg, conceded=ag, shots=match.get("home_shots_on_target"),
                possession=match.get("home_possession"), at_home=True, season=season, kickoff=kickoff)
        _record(histories[away_id], scored=ag, conceded=hg, shots=match.get("away_shots_on_target"),
                possession=match.get("away_possession"), at_home=False, season=season, kickoff=kickoff)
        winner = home_id if hg > ag else (away_id if ag > hg else None)
        h2h_log[(min(home_id, away_id), max(home_id, away_id))].append({"winner": winner})

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
