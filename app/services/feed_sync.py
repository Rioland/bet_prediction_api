"""Persist live feed fixtures into the database.

football_api.py keeps fixtures in an in-memory cache, which is all the
Dixon-Coles path needs: it computes predictions from team ratings on the spot.
The learned pipeline needs something the cache cannot provide - a history of
results that survives a restart - so this mirrors the same fixtures into the
leagues/teams/matches tables.

One store, two readers: the daily-pick endpoints keep serving from the cache,
while training, backtesting and the results tracker read the database.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import League, Match, Team

FINISHED_STATUSES = {"FT", "AET", "PEN", "FINISHED"}


def _parse_kickoff(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _upsert_league(db: Session, fixture: dict[str, Any]) -> League:
    external_id = fixture.get("league_id")
    league = db.scalar(select(League).where(League.external_id == external_id))
    if league is None:
        league = League(external_id=external_id)
        db.add(league)
    league.name = fixture.get("league_name") or league.name or "Unknown"
    league.country = fixture.get("league_country") or league.country
    return league


def _upsert_team(db: Session, name: str, logo: str | None) -> Team:
    """The feed gives no stable team id, so names are the key.

    Team names from one provider are consistent within that provider, which is
    enough here; switching providers would need a mapping table.
    """
    team = db.scalar(select(Team).where(Team.name == name))
    if team is None:
        team = Team(name=name)
        db.add(team)
    team.logo_url = logo or team.logo_url
    return team


def sync_fixture(db: Session, fixture: dict[str, Any]) -> Match | None:
    external_id = fixture.get("fixture_id")
    home_name, away_name = fixture.get("home_team"), fixture.get("away_team")
    kickoff = _parse_kickoff(fixture.get("kickoff", ""))
    if not external_id or not home_name or not away_name or kickoff is None:
        return None

    league = _upsert_league(db, fixture)
    home = _upsert_team(db, home_name, fixture.get("home_logo"))
    away = _upsert_team(db, away_name, fixture.get("away_logo"))
    db.flush()

    match = db.scalar(select(Match).where(Match.external_id == external_id))
    if match is None:
        match = Match(external_id=external_id)
        db.add(match)

    match.league_id = league.id
    match.home_team_id = home.id
    match.away_team_id = away.id
    match.kickoff_time = kickoff
    match.status = fixture.get("status") or "NS"
    match.season = kickoff.year

    # Only record a result once the fixture is actually complete.
    if match.status in FINISHED_STATUSES:
        if fixture.get("home_score") is not None:
            match.home_goals = fixture["home_score"]
        if fixture.get("away_score") is not None:
            match.away_goals = fixture["away_score"]
    return match


def sync_fixtures(db: Session, fixtures: list[dict[str, Any]]) -> int:
    synced = sum(1 for fixture in fixtures if sync_fixture(db, fixture) is not None)
    db.commit()
    return synced
