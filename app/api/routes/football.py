"""Public prediction endpoints, matching lib/api-spec/openapi.yaml.

Mounted under /api so the generated predictor-web client works unchanged.
"""

from datetime import datetime, time, timedelta

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.api.deps import DbSession
from app.ml.features import FEATURE_COLUMNS, features_for_upcoming
from app.ml.dataset import load_finished_matches
from app.models.entities import League, Match
from app.schemas.football import LeagueOut, MatchWithPrediction
from app.services.ingest import FINISHED_STATUSES
from app.services.match_presenter import league_to_out, to_card

router = APIRouter(prefix="/api", tags=["football"])

LIVE_STATUSES = {"1H", "2H", "HT", "ET", "BT", "P", "LIVE"}


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/football/leagues", response_model=list[LeagueOut])
def list_leagues(db: DbSession) -> list[LeagueOut]:
    return [league_to_out(row) for row in db.scalars(select(League).order_by(League.name))]


def _cards_for(db: DbSession, matches: list[Match]) -> list[MatchWithPrediction]:
    """Build feature rows for the given fixtures, then render prediction cards."""
    if not matches:
        return []

    history = load_finished_matches(db)
    upcoming = [
        {
            "id": m.id,
            "home_team_id": m.home_team_id,
            "away_team_id": m.away_team_id,
            "kickoff_time": m.kickoff_time,
            "season": m.season,
        }
        for m in matches
    ]
    frame = features_for_upcoming(history, upcoming)
    if frame.empty:
        return []

    by_id = {row["match_id"]: row for _, row in frame.iterrows()}
    cards = []
    for match in matches:
        row = by_id.get(match.id)
        if row is None:
            continue
        cards.append(to_card(match, {c: float(row[c]) for c in FEATURE_COLUMNS}))
    return cards


def _todays_matches(db: DbSession, league_id: int | None) -> list[Match]:
    start = datetime.combine(datetime.utcnow().date(), time.min)
    stmt = select(Match).where(
        Match.kickoff_time >= start, Match.kickoff_time < start + timedelta(days=1)
    )
    if league_id is not None:
        stmt = stmt.where(Match.league_id == league_id)
    return list(db.scalars(stmt.order_by(Match.kickoff_time)))


@router.get("/football/matches/today", response_model=list[MatchWithPrediction])
def matches_today(db: DbSession, league_id: int | None = Query(default=None)) -> list[MatchWithPrediction]:
    return _cards_for(db, _todays_matches(db, league_id))


@router.get("/football/predictions/today", response_model=list[MatchWithPrediction])
def predictions_today(
    db: DbSession,
    league_id: int | None = Query(default=None),
    min_confidence: float = Query(default=0.0, ge=0, le=1),
) -> list[MatchWithPrediction]:
    """Prediction cards for today, most confident first.

    ``min_confidence`` is a calibrated probability in 0-1, not a marketing score:
    0.6 means the model expects that outcome to happen about 60% of the time.
    """
    upcoming = [m for m in _todays_matches(db, league_id) if m.status not in FINISHED_STATUSES]
    cards = _cards_for(db, upcoming)
    cards = [c for c in cards if c.prediction.confidence >= min_confidence]
    return sorted(cards, key=lambda c: c.prediction.confidence, reverse=True)


@router.get("/football/live", response_model=list[MatchWithPrediction])
def live_matches(db: DbSession) -> list[MatchWithPrediction]:
    matches = list(db.scalars(select(Match).where(Match.status.in_(LIVE_STATUSES))))
    return _cards_for(db, matches)


@router.get("/football/predictions/{fixture_id}", response_model=MatchWithPrediction)
def prediction_by_fixture(fixture_id: int, db: DbSession) -> MatchWithPrediction:
    match = db.scalar(select(Match).where(Match.external_id == fixture_id))
    if match is None:
        match = db.get(Match, fixture_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Fixture not found")

    cards = _cards_for(db, [match])
    if not cards:
        raise HTTPException(status_code=404, detail="Fixture not found")
    return cards[0]
