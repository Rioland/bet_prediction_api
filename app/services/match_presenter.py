"""Turn model output into the card shape the frontend renders.

Probabilities come straight from the calibrated models. Expected goals are a
Poisson-style estimate from each side's rolling attack and the opponent's
rolling defence - a descriptive summary, not a separate trained model.
"""

from typing import Any

from app.models.entities import Match
from app.schemas.football import LeagueOut, MatchWithPrediction, PredictionOut
from app.services.prediction_service import predict

_WINNER_LABEL = {"H": "home", "D": "draw", "A": "away"}


def estimate_expected_goals(features: dict[str, float]) -> tuple[float, float]:
    """Blend each team's scoring rate with the opponent's concession rate."""
    home = (features["home_goals_scored_avg"] + features["away_goals_conceded_avg"]) / 2
    away = (features["away_goals_scored_avg"] + features["home_goals_conceded_avg"]) / 2
    return round(home * 1.08, 2), round(away * 0.94, 2)  # modest home advantage


def build_prediction(features: dict[str, float]) -> PredictionOut:
    winner = predict("match_winner", features)
    btts = predict("btts", features)
    totals = predict("over_under_2_5", features)

    probs = winner["probabilities"]
    home_xg, away_xg = estimate_expected_goals(features)

    return PredictionOut(
        home_win_prob=round(probs.get("H", 0.0), 4),
        draw_prob=round(probs.get("D", 0.0), 4),
        away_win_prob=round(probs.get("A", 0.0), 4),
        btts_prob=round(btts["probabilities"].get("yes", 0.0), 4),
        over_25_prob=round(totals["probabilities"].get("over", 0.0), 4),
        predicted_winner=_WINNER_LABEL.get(winner["prediction"], "draw"),
        # predict() reports a percentage; the frontend expects a 0-1 probability.
        confidence=round(winner["confidence"] / 100, 4),
        home_xg=home_xg,
        away_xg=away_xg,
        predicted_score=f"{round(home_xg)}-{round(away_xg)}",
    )


def to_card(match: Match, features: dict[str, float]) -> MatchWithPrediction:
    return MatchWithPrediction(
        fixture_id=match.external_id or match.id,
        league_id=match.league_id,
        league_name=match.league.name if match.league else "Unknown",
        league_country=match.league.country if match.league else None,
        home_team=match.home_team.name if match.home_team else "Unknown",
        home_logo=match.home_team.logo_url if match.home_team else None,
        away_team=match.away_team.name if match.away_team else "Unknown",
        away_logo=match.away_team.logo_url if match.away_team else None,
        kickoff=match.kickoff_time.isoformat(),
        status=match.status,
        home_score=match.home_goals,
        away_score=match.away_goals,
        prediction=build_prediction(features),
    )


def league_to_out(league: Any) -> LeagueOut:
    return LeagueOut(id=league.id, name=league.name, country=league.country)
