"""The live feed must land in the database the pipeline reads."""

from sqlalchemy.orm import Session

from app.models import League, Match, Team
from app.services.feed_sync import sync_fixture, sync_fixtures

FIXTURE = {
    "fixture_id": 4001,
    "league_id": 2021,
    "league_name": "Premier League",
    "league_country": "England",
    "home_team": "Arsenal",
    "home_logo": "https://example.test/ars.png",
    "away_team": "Chelsea",
    "away_logo": "https://example.test/che.png",
    "kickoff": "2026-03-10T15:00:00Z",
    "status": "NS",
    "home_score": None,
    "away_score": None,
}


def test_sync_creates_league_teams_and_match(db_session: Session) -> None:
    assert sync_fixtures(db_session, [FIXTURE]) == 1
    match = db_session.query(Match).filter_by(external_id=4001).one()
    assert match.status == "NS"
    assert match.home_goals is None
    assert db_session.query(Team).count() == 2
    assert db_session.query(League).one().name == "Premier League"


def test_sync_is_idempotent(db_session: Session) -> None:
    sync_fixtures(db_session, [FIXTURE])
    sync_fixtures(db_session, [FIXTURE])
    assert db_session.query(Match).count() == 1
    assert db_session.query(Team).count() == 2


def test_result_recorded_only_once_finished(db_session: Session) -> None:
    sync_fixtures(db_session, [FIXTURE])
    # A score arriving while still marked NS must not be treated as final.
    sync_fixtures(db_session, [{**FIXTURE, "home_score": 1, "away_score": 0}])
    assert db_session.query(Match).one().home_goals is None

    sync_fixtures(db_session, [{**FIXTURE, "status": "FT", "home_score": 2, "away_score": 1}])
    match = db_session.query(Match).one()
    assert (match.home_goals, match.away_goals) == (2, 1)


def test_malformed_fixtures_are_skipped(db_session: Session) -> None:
    assert sync_fixture(db_session, {"fixture_id": None}) is None
    assert sync_fixture(db_session, {**FIXTURE, "kickoff": "not-a-date"}) is None
    assert sync_fixture(db_session, {**FIXTURE, "away_team": None}) is None
    assert db_session.query(Match).count() == 0


def test_kickoff_is_stored_as_naive_utc(db_session: Session) -> None:
    sync_fixtures(db_session, [FIXTURE])
    kickoff = db_session.query(Match).one().kickoff_time
    assert kickoff.tzinfo is None
    assert (kickoff.hour, kickoff.year, kickoff.month) == (15, 2026, 3)


# --- results must reach the database, or no team ever becomes analysable -----


def test_finished_fixtures_are_not_prediction_candidates() -> None:
    """The live cache excludes them - which is why they must be persisted separately."""
    from app.football_api import _is_prediction_candidate

    assert _is_prediction_candidate({"status": "NS"}) is True
    assert _is_prediction_candidate({"status": "1H"}) is True
    for status in ("FT", "AET", "PEN"):
        assert _is_prediction_candidate({"status": status}) is False, status


def test_espn_window_reaches_backwards_for_results() -> None:
    """Without a look-back the service only ever sees fixtures that have not happened."""
    import inspect

    from app.football_api import _RESULT_LOOKBACK_DAYS, _fetch_espn_fixtures, refresh_fixtures_loop

    assert _RESULT_LOOKBACK_DAYS > 0
    assert "lookback" in inspect.signature(_fetch_espn_fixtures).parameters

    source = inspect.getsource(refresh_fixtures_loop)
    assert "lookback=_RESULT_LOOKBACK_DAYS" in source
    # Finished fixtures must be persisted even though they never enter the cache.
    assert "_persist_fixtures(_fixture_cache + finished)" in source


def test_completed_fixtures_build_up_team_history(db_session: Session) -> None:
    """Five completed matches per side is what unlocks analysis."""
    from app.ml.dataset import load_finished_matches

    played = [
        {**FIXTURE, "fixture_id": 4100 + i, "status": "FT",
         "home_score": 2, "away_score": 1,
         "kickoff": f"2026-0{(i % 9) + 1}-1{i % 9}T15:00:00Z"}
        for i in range(6)
    ]
    sync_fixtures(db_session, played)

    finished = load_finished_matches(db_session)
    assert len(finished) == 6
    assert all(m["home_goals"] is not None for m in finished)
