"""Public prediction endpoints, matching lib/api-spec/openapi.yaml.

Mounted under /api so the generated predictor-web client works unchanged.
"""

from datetime import date as date_type, datetime, time, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select

from app.api.deps import DbSession, get_current_user, oauth2_scheme
from app.ml.features import FEATURE_COLUMNS, features_for_upcoming
from app.ml.dataset import load_finished_matches
from app.models.entities import League, Match, SubscriptionType, User, UserRole
from app.schemas.football import LeagueOut, MatchWithPrediction
from app.services.ingest import FINISHED_STATUSES
from app.services.match_presenter import league_to_out, to_card
from app.services import results as results_service
from app.services.tips import (
    MARKETS,
    build_accumulator,
    build_tips,
    filter_by_market,
    select_banker,
)

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


def _matches_on(db: DbSession, on_date: date_type | None, league_id: int | None) -> list[Match]:
    start = datetime.combine(on_date or datetime.utcnow().date(), time.min)
    stmt = select(Match).where(
        Match.kickoff_time >= start, Match.kickoff_time < start + timedelta(days=1)
    )
    if league_id is not None:
        stmt = stmt.where(Match.league_id == league_id)
    return list(db.scalars(stmt.order_by(Match.kickoff_time)))


def _todays_matches(db: DbSession, league_id: int | None) -> list[Match]:
    return _matches_on(db, None, league_id)


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


# --- Tips ------------------------------------------------------------------


def _has_vip(user: User | None) -> bool:
    if user is None:
        return False
    return (
        user.subscription_type == SubscriptionType.PREMIUM
        or user.role in {UserRole.PREMIUM_USER, UserRole.ADMIN, UserRole.SUPER_ADMIN}
    )


async def optional_user(db: DbSession, request: Request) -> User | None:
    """Resolve the caller if they are signed in, without requiring it.

    Tips are browsable anonymously; only the VIP selections are withheld.
    """
    from app.api.deps import get_current_user as _resolve

    try:
        token = await oauth2_scheme(request)
        return _resolve(request, db, token)
    except HTTPException:
        return None


def _tip_cards(db: DbSession, matches: list[Match], market: str) -> list[dict]:
    cards = []
    for card in _cards_for(db, matches):
        match_row = next((m for m in matches if (m.external_id or m.id) == card.fixture_id), None)
        payload = card.model_dump()
        payload.update({
            "odds_home": match_row.odds_home if match_row else None,
            "odds_draw": match_row.odds_draw if match_row else None,
            "odds_away": match_row.odds_away if match_row else None,
            "home_team": card.home_team,
            "away_team": card.away_team,
        })
        candidates = filter_by_market(build_tips(card.prediction.model_dump(), payload), market)
        payload["tip"] = candidates[0].as_dict() if candidates else None
        payload["all_tips"] = [t.as_dict() for t in candidates]
        cards.append(payload)
    return cards


@router.get("/football/markets")
def list_markets() -> list[str]:
    return MARKETS


@router.get("/football/tips")
def tips(
    db: DbSession,
    viewer: Annotated[User | None, Depends(optional_user)],
    on_date: date_type | None = Query(default=None, alias="date"),
    market: str = Query(default="popular"),
    league_id: int | None = Query(default=None),
) -> dict:
    """Tip cards for a date and market.

    VIP selections are withheld from callers without a subscription: the count
    is disclosed so the paywall is visible, but the picks themselves are not.
    """
    if market not in MARKETS:
        raise HTTPException(status_code=422, detail=f"Unknown market '{market}'")

    fixtures = [m for m in _matches_on(db, on_date, league_id) if m.status not in FINISHED_STATUSES]
    cards = [c for c in _tip_cards(db, fixtures, market) if c["tip"]]
    cards.sort(key=lambda c: c["tip"]["probability"], reverse=True)

    # The strongest picks are the paid product; the rest are free.
    vip_cards = [c for c in cards if c["tip"]["value"] >= 0.04][:3]
    vip_ids = {c["fixture_id"] for c in vip_cards}
    free_cards = [c for c in cards if c["fixture_id"] not in vip_ids]

    unlocked = _has_vip(viewer)
    payload: dict = {
        "date": (on_date or datetime.utcnow().date()).isoformat(),
        "market": market,
        "matches": free_cards,
        "vip_locked": not unlocked,
        "vip_count": len(vip_cards),
    }
    if unlocked:
        payload["vip_matches"] = vip_cards
    if market == "banker":
        payload["banker"] = select_banker(free_cards)
    if market in ("2_odds", "acca"):
        payload["accumulator"] = build_accumulator(
            free_cards, target_odds=2.0 if market == "2_odds" else 5.0
        )
    return payload


@router.get("/football/vip/tips")
def vip_tips(
    db: DbSession,
    current_user: Annotated[User, Depends(get_current_user)],
    on_date: date_type | None = Query(default=None, alias="date"),
    market: str = Query(default="popular"),
) -> dict:
    if not _has_vip(current_user):
        raise HTTPException(status_code=402, detail="VIP subscription required")

    fixtures = [m for m in _matches_on(db, on_date, None) if m.status not in FINISHED_STATUSES]
    cards = [c for c in _tip_cards(db, fixtures, market) if c["tip"]]
    cards.sort(key=lambda c: c["tip"]["value"], reverse=True)
    return {"date": (on_date or datetime.utcnow().date()).isoformat(), "matches": cards}


# --- Verified results ------------------------------------------------------


@router.get("/football/results/performance")
def tracked_performance(db: DbSession, days: int = Query(default=30, ge=1, le=365)) -> dict:
    """Real settled performance. Every figure comes from tips recorded pre-kickoff."""
    return {
        "overall": results_service.performance(db, days=days),
        "by_market": results_service.performance_by_market(db, days=days),
    }


@router.get("/football/results/recent")
def recent_results(db: DbSession, limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
    rows = results_service.recent_settled(db, limit=limit)
    return [
        {
            "match_id": t.match_id,
            "home_team": t.match.home_team.name if t.match and t.match.home_team else None,
            "away_team": t.match.away_team.name if t.match and t.match.away_team else None,
            "kickoff": t.kickoff_time.isoformat(),
            "market": t.market,
            "selection": t.selection,
            "odds": t.odds,
            "probability": t.probability,
            "result": t.result.value,
            "score": (
                f"{t.match.home_goals}-{t.match.away_goals}"
                if t.match and t.match.home_goals is not None
                else None
            ),
        }
        for t in rows
    ]
