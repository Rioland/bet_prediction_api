from datetime import date, datetime

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.deps import DbSession
from app.models.entities import Match
from app.schemas.common import MatchOut

router = APIRouter(prefix="/matches", tags=["matches"])


@router.get("/today", response_model=list[MatchOut])
def get_today_matches(db: DbSession) -> list[Match]:
    today = date.today()
    start = datetime(today.year, today.month, today.day)
    end = datetime(today.year, today.month, today.day, 23, 59, 59)
    return list(db.scalars(select(Match).where(Match.kickoff_time >= start, Match.kickoff_time <= end)))


@router.get("/live", response_model=list[MatchOut])
def get_live_matches(db: DbSession) -> list[Match]:
    return list(db.scalars(select(Match).where(Match.status == "live")))


@router.get("/{match_id}", response_model=MatchOut)
def get_match(match_id: int, db: DbSession) -> Match:
    match = db.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match
