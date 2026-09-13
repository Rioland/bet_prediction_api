"""Ingestion tests against api-football's payload shape."""

from sqlalchemy.orm import Session

from app.models import Match, Team
from app.services.ingest import apply_odds, apply_statistics, ingest_fixtures, upsert_fixture

FIXTURE = {
    "fixture": {"id": 1001, "date": "2024-03-10T15:00:00+00:00", "status": {"short": "FT"}},
    "league": {"id": 39, "name": "Premier League", "country": "England", "season": 2023},
    "teams": {"home": {"id": 50, "name": "Man City"}, "away": {"id": 42, "name": "Arsenal"}},
    "goals": {"home": 2, "away": 1},
}

UPCOMING = {
    "fixture": {"id": 1002, "date": "2099-03-10T15:00:00+00:00", "status": {"short": "NS"}},
    "league": {"id": 39, "name": "Premier League", "country": "England", "season": 2023},
    "teams": {"home": {"id": 42, "name": "Arsenal"}, "away": {"id": 50, "name": "Man City"}},
    "goals": {"home": None, "away": None},
}


def test_ingest_creates_league_teams_and_match(db_session: Session) -> None:
    assert ingest_fixtures(db_session, [FIXTURE]) == 1

    match = db_session.query(Match).filter_by(external_id=1001).one()
    assert match.home_goals == 2 and match.away_goals == 1
    assert match.status == "FT"
    assert match.season == 2023
    assert db_session.query(Team).count() == 2


def test_ingest_is_idempotent(db_session: Session) -> None:
    ingest_fixtures(db_session, [FIXTURE])
    ingest_fixtures(db_session, [FIXTURE])
    assert db_session.query(Match).count() == 1
    assert db_session.query(Team).count() == 2


def test_unplayed_fixture_has_no_result(db_session: Session) -> None:
    ingest_fixtures(db_session, [UPCOMING])
    match = db_session.query(Match).filter_by(external_id=1002).one()
    assert match.home_goals is None and match.away_goals is None


def test_result_is_recorded_once_a_fixture_finishes(db_session: Session) -> None:
    ingest_fixtures(db_session, [UPCOMING])
    finished = {**UPCOMING, "fixture": {**UPCOMING["fixture"], "status": {"short": "FT"}},
                "goals": {"home": 0, "away": 3}}
    ingest_fixtures(db_session, [finished])

    match = db_session.query(Match).filter_by(external_id=1002).one()
    assert (match.home_goals, match.away_goals) == (0, 3)
    assert db_session.query(Match).count() == 1


def test_malformed_payload_is_skipped(db_session: Session) -> None:
    assert ingest_fixtures(db_session, [{"fixture": {"id": None}}]) == 0
    assert upsert_fixture(db_session, {"fixture": {"id": 5}, "teams": {}}) is None


def test_statistics_are_mapped_to_the_right_side(db_session: Session) -> None:
    ingest_fixtures(db_session, [FIXTURE])
    match = db_session.query(Match).filter_by(external_id=1001).one()

    apply_statistics(db_session, match, [
        {"team": {"id": 50}, "statistics": [
            {"type": "Shots on Goal", "value": 8},
            {"type": "Ball Possession", "value": "63%"},
            {"type": "Corner Kicks", "value": 7},
        ]},
        {"team": {"id": 42}, "statistics": [
            {"type": "Shots on Goal", "value": 3},
            {"type": "Ball Possession", "value": "37%"},
            {"type": "Corner Kicks", "value": 2},
        ]},
    ])
    db_session.commit()

    assert match.home_shots_on_target == 8
    assert match.home_possession == 63.0
    assert match.away_shots_on_target == 3
    assert match.away_possession == 37.0


def test_missing_statistic_values_stay_none(db_session: Session) -> None:
    ingest_fixtures(db_session, [FIXTURE])
    match = db_session.query(Match).filter_by(external_id=1001).one()
    apply_statistics(db_session, match, [
        {"team": {"id": 50}, "statistics": [{"type": "Shots on Goal", "value": None}]}
    ])
    assert match.home_shots_on_target is None


def test_odds_are_averaged_across_bookmakers(db_session: Session) -> None:
    ingest_fixtures(db_session, [FIXTURE])
    match = db_session.query(Match).filter_by(external_id=1001).one()

    apply_odds(db_session, match, [{"bookmakers": [
        {"bets": [{"name": "Match Winner", "values": [
            {"value": "Home", "odd": "1.50"}, {"value": "Draw", "odd": "4.00"},
            {"value": "Away", "odd": "6.00"}]}]},
        {"bets": [{"name": "Match Winner", "values": [
            {"value": "Home", "odd": "1.60"}, {"value": "Draw", "odd": "4.20"},
            {"value": "Away", "odd": "6.40"}]}]},
    ]}])
    db_session.commit()

    assert match.odds_home == 1.55
    assert match.odds_draw == 4.10
    assert round(match.odds_away, 2) == 6.20


def test_irrelevant_betting_markets_are_ignored(db_session: Session) -> None:
    ingest_fixtures(db_session, [FIXTURE])
    match = db_session.query(Match).filter_by(external_id=1001).one()
    apply_odds(db_session, match, [{"bookmakers": [
        {"bets": [{"name": "Both Teams Score", "values": [{"value": "Yes", "odd": "1.80"}]}]}
    ]}])
    assert match.odds_home is None
