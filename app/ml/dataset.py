"""Load matches out of the database into the shape ``features`` expects."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Match
from app.services.ingest import FINISHED_STATUSES

_FIELDS = (
    "id", "league_id", "home_team_id", "away_team_id", "kickoff_time", "season",
    "home_goals", "away_goals", "home_shots_on_target", "away_shots_on_target",
    "home_possession", "away_possession", "odds_home", "odds_draw", "odds_away",
)


def _as_dict(match: Match) -> dict[str, Any]:
    return {field: getattr(match, field) for field in _FIELDS}


def load_finished_matches(db: Session, league_ids: list[int] | None = None) -> list[dict[str, Any]]:
    stmt = select(Match).where(
        Match.status.in_(FINISHED_STATUSES),
        Match.home_goals.is_not(None),
        Match.away_goals.is_not(None),
    )
    if league_ids:
        stmt = stmt.where(Match.league_id.in_(league_ids))
    return [_as_dict(m) for m in db.scalars(stmt.order_by(Match.kickoff_time))]


def load_upcoming_matches(db: Session, league_ids: list[int] | None = None) -> list[dict[str, Any]]:
    stmt = select(Match).where(Match.status.not_in(FINISHED_STATUSES))
    if league_ids:
        stmt = stmt.where(Match.league_id.in_(league_ids))
    return [_as_dict(m) for m in db.scalars(stmt.order_by(Match.kickoff_time))]
