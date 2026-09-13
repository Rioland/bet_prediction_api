"""Sport-agnostic prediction endpoints.

GET /predictions/{sport} returns upcoming matches with a predicted outcome and
a confidence score. Confidence is the model's calibrated probability for the
predicted outcome - a real probability, not a rating out of ten - so a 0.62
means the outcome is expected about 62% of the time.
"""

from __future__ import annotations

from datetime import date as date_type, datetime, time, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Match
from app.services.prediction_service import predict
from app.sports import SportAdapter
from app.sports.registry import available_sports, get_adapter

router = APIRouter(prefix="/predictions", tags=["predictions"])

DbSession = Annotated[Session, Depends(get_db)]
FINISHED_STATUSES = {"FT", "AET", "PEN", "FINISHED"}
DEFAULT_HORIZON_DAYS = 3


@router.get("")
def list_sports() -> dict[str, Any]:
    """Which sports this deployment can predict."""
    return {
        "sports": [
            {
                "name": adapter.name,
                "display_name": adapter.display_name,
                "has_draw": adapter.has_draw,
                "targets": adapter.targets,
                "min_team_history": adapter.min_team_history,
            }
            for adapter in (get_adapter(name) for name in available_sports())
        ]
    }


def _upcoming(db: Session, sport: str, on_date: date_type | None, days: int) -> list[Match]:
    start = datetime.combine(on_date or datetime.utcnow().date(), time.min)
    stmt = (
        select(Match)
        .where(
            Match.sport == sport,
            Match.kickoff_time >= start,
            Match.kickoff_time < start + timedelta(days=days),
            Match.status.not_in(FINISHED_STATUSES),
        )
        .order_by(Match.kickoff_time)
    )
    return list(db.scalars(stmt))


def _outcome_label(adapter: SportAdapter, code: str, home: str, away: str) -> str:
    return {"H": home, "A": away, "D": "Draw"}.get(code, code)


def _predict_match(
    adapter: SportAdapter, match: Match, features: dict[str, float]
) -> dict[str, Any]:
    outcome = predict("match_winner", features, sport=adapter.name)
    home = match.home_team.name if match.home_team else "Home"
    away = match.away_team.name if match.away_team else "Away"

    return {
        "fixture_id": match.external_id or match.id,
        "sport": adapter.name,
        "league_id": match.league_id,
        "league_name": match.league.name if match.league else None,
        "home_team": home,
        "away_team": away,
        "kickoff": match.kickoff_time.isoformat(),
        "status": match.status,
        "predicted_outcome": outcome["prediction"],
        "predicted_label": _outcome_label(adapter, outcome["prediction"], home, away),
        # 0-1. The calibrated probability of the predicted outcome, not a score.
        "confidence": round(outcome["confidence"] / 100, 4),
        "probabilities": outcome["probabilities"],
    }


@router.get("/{sport}")
def predictions_for_sport(
    sport: str,
    db: DbSession,
    on_date: date_type | None = Query(default=None, alias="date"),
    days: int = Query(default=DEFAULT_HORIZON_DAYS, ge=1, le=14),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """Upcoming matches for one sport with predicted outcome and confidence.

    Fixtures whose teams lack enough completed matches are reported in
    ``skipped`` rather than dropped, so a short list is distinguishable from a
    missing schedule.
    """
    adapter = get_adapter(sport)
    matches = _upcoming(db, adapter.name, on_date, days)
    if not matches:
        return {
            "sport": adapter.name,
            "count": 0,
            "skipped": 0,
            "predictions": [],
        }

    from app.ml.dataset import load_finished_matches

    history = [m for m in load_finished_matches(db) if m.get("sport", adapter.name) == adapter.name]
    upcoming = [
        {
            "id": m.id,
            "home_team_id": m.home_team_id,
            "away_team_id": m.away_team_id,
            "kickoff_time": m.kickoff_time,
            "season": m.season,
            "odds_home": m.odds_home,
            "odds_draw": m.odds_draw,
            "odds_away": m.odds_away,
        }
        for m in matches
    ]
    frame = adapter.features_for_upcoming(history, upcoming)
    rows = (
        {int(row["match_id"]): {c: float(row[c]) for c in adapter.feature_columns}
         for _, row in frame.iterrows()}
        if not frame.empty
        else {}
    )

    predictions: list[dict[str, Any]] = []
    skipped = 0
    for match in matches:
        features = rows.get(match.id)
        if features is None or min(
            features["home_matches_played"], features["away_matches_played"]
        ) < adapter.min_team_history:
            skipped += 1
            continue
        predictions.append(_predict_match(adapter, match, features))

    predictions = [p for p in predictions if p["confidence"] >= min_confidence]
    predictions.sort(key=lambda p: p["confidence"], reverse=True)

    return {
        "sport": adapter.name,
        "count": len(predictions),
        "skipped": skipped,
        "skipped_reason": (
            f"fewer than {adapter.min_team_history} completed matches for one or both teams"
            if skipped
            else None
        ),
        "predictions": predictions,
    }
