from datetime import date, datetime

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.api.deps import DbSession
from app.models.entities import Match
from app.schemas.common import MatchOut

router = APIRouter(prefix="/matches", tags=["matches"])


def _match_out(match: Match) -> MatchOut:
    return MatchOut(
        id=match.id,
        league_id=match.league_id,
        home_team_id=match.home_team_id,
        away_team_id=match.away_team_id,
        home_team_name=match.home_team.name if match.home_team else None,
        away_team_name=match.away_team.name if match.away_team else None,
        league_name=match.league.name if match.league else None,
        kickoff_time=match.kickoff_time,
        status=match.status,
    )


def _match_query():
    return select(Match).options(
        joinedload(Match.home_team),
        joinedload(Match.away_team),
        joinedload(Match.league),
    )


@router.get("/today", response_model=list[MatchOut])
def get_today_matches(db: DbSession) -> list[MatchOut]:
    today = date.today()
    start = datetime(today.year, today.month, today.day)
    end = datetime(today.year, today.month, today.day, 23, 59, 59)
    matches = list(
        db.scalars(_match_query().where(Match.kickoff_time >= start, Match.kickoff_time <= end))
    )
    return [_match_out(m) for m in matches]


@router.get("/live", response_model=list[MatchOut])
def get_live_matches(db: DbSession) -> list[MatchOut]:
    matches = list(db.scalars(_match_query().where(Match.status == "live")))
    return [_match_out(m) for m in matches]


@router.get("/{match_id}", response_model=MatchOut)
def get_match(match_id: int, db: DbSession) -> MatchOut:
    match = db.scalar(_match_query().where(Match.id == match_id))
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return _match_out(match)
