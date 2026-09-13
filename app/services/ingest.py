"""Normalise api-football payloads into leagues / teams / matches.

Kept free of network calls so it can be unit-tested against recorded payloads.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import League, Match, Team

# api-football status codes that mean the 90 minutes are complete.
FINISHED_STATUSES = {"FT", "AET", "PEN"}


def _parse_kickoff(value: str) -> datetime:
    # Stored naive-UTC to match the existing DateTime columns.
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)


def upsert_league(db: Session, payload: dict[str, Any]) -> League:
    external_id = payload["id"]
    league = db.scalar(select(League).where(League.external_id == external_id))
    if league is None:
        league = League(external_id=external_id, name=payload.get("name", "Unknown"))
        db.add(league)
    league.name = payload.get("name", league.name)
    league.country = payload.get("country", league.country)
    return league


def upsert_team(db: Session, payload: dict[str, Any]) -> Team:
    external_id = payload["id"]
    team = db.scalar(select(Team).where(Team.external_id == external_id))
    if team is None:
        team = Team(external_id=external_id, name=payload.get("name", "Unknown"))
        db.add(team)
    team.name = payload.get("name", team.name)
    team.logo_url = payload.get("logo", team.logo_url)
    return team


def upsert_fixture(db: Session, item: dict[str, Any]) -> Match | None:
    """Upsert one api-football fixture entry. Returns None if it is unusable."""
    fixture = item.get("fixture") or {}
    league_payload = item.get("league") or {}
    teams = item.get("teams") or {}
    goals = item.get("goals") or {}

    external_id = fixture.get("id")
    home_payload, away_payload = teams.get("home"), teams.get("away")
    if not external_id or not home_payload or not away_payload or not fixture.get("date"):
        return None

    league = upsert_league(db, league_payload)
    home = upsert_team(db, home_payload)
    away = upsert_team(db, away_payload)
    db.flush()  # assign surrogate ids before referencing them

    status = ((fixture.get("status") or {}).get("short")) or "NS"
    match = db.scalar(select(Match).where(Match.external_id == external_id))
    if match is None:
        match = Match(external_id=external_id)
        db.add(match)

    match.league_id = league.id
    match.home_team_id = home.id
    match.away_team_id = away.id
    match.kickoff_time = _parse_kickoff(fixture["date"])
    match.status = status
    match.season = league_payload.get("season")

    # Only record a result once the match is actually complete.
    if status in FINISHED_STATUSES:
        match.home_goals = goals.get("home")
        match.away_goals = goals.get("away")
    return match


def ingest_fixtures(db: Session, items: list[dict[str, Any]]) -> int:
    count = 0
    for item in items:
        if upsert_fixture(db, item) is not None:
            count += 1
    db.commit()
    return count


def _stat(entries: list[dict[str, Any]], wanted: str) -> Any:
    for entry in entries:
        if entry.get("type") == wanted:
            return entry.get("value")
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_percent(value: Any) -> float | None:
    if isinstance(value, str) and value.endswith("%"):
        value = value[:-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def apply_statistics(db: Session, match: Match, response: list[dict[str, Any]]) -> None:
    """Attach post-match team statistics to a played fixture."""
    for side in response:
        team_external = ((side.get("team") or {}).get("id"))
        stats = side.get("statistics") or []
        team = db.scalar(select(Team).where(Team.external_id == team_external))
        if team is None:
            continue
        shots = _as_int(_stat(stats, "Shots on Goal"))
        possession = _as_percent(_stat(stats, "Ball Possession"))
        corners = _as_int(_stat(stats, "Corner Kicks"))
        if team.id == match.home_team_id:
            match.home_shots_on_target, match.home_possession, match.home_corners = shots, possession, corners
        elif team.id == match.away_team_id:
            match.away_shots_on_target, match.away_possession, match.away_corners = shots, possession, corners


def apply_odds(db: Session, match: Match, response: list[dict[str, Any]]) -> None:
    """Record 1X2 odds, averaged across bookmakers offering the Match Winner market."""
    home: list[float] = []
    draw: list[float] = []
    away: list[float] = []
    for entry in response:
        for bookmaker in entry.get("bookmakers") or []:
            for bet in bookmaker.get("bets") or []:
                if bet.get("name") != "Match Winner":
                    continue
                for value in bet.get("values") or []:
                    try:
                        odd = float(value.get("odd"))
                    except (TypeError, ValueError):
                        continue
                    label = str(value.get("value", "")).lower()
                    if label == "home":
                        home.append(odd)
                    elif label == "draw":
                        draw.append(odd)
                    elif label == "away":
                        away.append(odd)
    if home:
        match.odds_home = sum(home) / len(home)
    if draw:
        match.odds_draw = sum(draw) / len(draw)
    if away:
        match.odds_away = sum(away) / len(away)
