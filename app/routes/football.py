"""Public football prediction routes (no auth required)."""

from fastapi import APIRouter, HTTPException, Query

from app.football_api import (
    get_daily_pick,
    get_daily_picks,
    get_leagues,
    get_live_fixtures,
    get_today_fixtures,
)

router = APIRouter(prefix="/football", tags=["football"])

def _get_fixtures() -> list[dict]:
    # The data layer owns the refresh cache. Avoid a second route cache that
    # could hide newly fetched matches for up to two extra hours.
    return get_today_fixtures()


@router.get("/leagues")
def leagues():
    return get_leagues()


@router.get("/matches/today")
def matches_today(league_id: int | None = Query(None)):
    fixtures = _get_fixtures()
    if league_id is not None:
        fixtures = [f for f in fixtures if f.get("league_id") == league_id]
    return fixtures


@router.get("/predictions/today")
def predictions_today(league_id: int | None = Query(None)):
    return matches_today(league_id=league_id)


@router.get("/pick/today")
def daily_pick():
    return get_daily_pick()


@router.get("/picks/daily")
def daily_picks():
    return get_daily_picks()


@router.get("/predictions/{fixture_id}")
def prediction_by_id(fixture_id: int):
    fixtures = _get_fixtures()
    for f in fixtures:
        if f.get("fixture_id") == fixture_id:
            return f
    raise HTTPException(status_code=404, detail="Fixture not found")


@router.get("/live")
def live_matches():
    """Return matches currently in-play, refreshed every 60 s via the live API."""
    return get_live_fixtures()
