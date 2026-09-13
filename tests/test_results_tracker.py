"""The results tracker must only ever report genuinely pre-recorded picks."""

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.models import League, Match, PublishedTip, Team
from app.services.results import performance, publish_tip, settle_pending_tips

TIP = {"market": "home_win", "selection": "1", "probability": 0.7, "odds": 1.8, "rationale": "x"}


def _match(db: Session, *, kickoff: datetime, status="NS", hg=None, ag=None) -> Match:
    league = League(external_id=1, name="L", country="C")
    home, away = Team(external_id=1, name="H"), Team(external_id=2, name="A")
    db.add_all([league, home, away])
    db.flush()
    match = Match(league_id=league.id, home_team_id=home.id, away_team_id=away.id,
                  kickoff_time=kickoff, status=status, home_goals=hg, away_goals=ag)
    db.add(match)
    db.commit()
    return match


def test_publishing_before_kickoff_records_the_tip(db_session: Session) -> None:
    match = _match(db_session, kickoff=datetime.utcnow() + timedelta(hours=3))
    assert publish_tip(db_session, match, TIP) is not None
    assert db_session.query(PublishedTip).count() == 1


def test_publishing_after_kickoff_is_refused(db_session: Session) -> None:
    """Otherwise the record could be back-filled once the result was known."""
    match = _match(db_session, kickoff=datetime.utcnow() - timedelta(hours=1))
    assert publish_tip(db_session, match, TIP) is None
    assert db_session.query(PublishedTip).count() == 0


def test_publishing_the_same_selection_twice_does_not_duplicate(db_session: Session) -> None:
    match = _match(db_session, kickoff=datetime.utcnow() + timedelta(hours=3))
    publish_tip(db_session, match, TIP)
    publish_tip(db_session, match, TIP)
    assert db_session.query(PublishedTip).count() == 1


def test_settlement_scores_finished_matches(db_session: Session) -> None:
    match = _match(db_session, kickoff=datetime.utcnow() + timedelta(hours=3))
    publish_tip(db_session, match, TIP)

    match.status, match.home_goals, match.away_goals = "FT", 2, 0
    db_session.commit()

    assert settle_pending_tips(db_session) == {"won": 1, "lost": 0, "void": 0, "settled": 1}
    assert db_session.query(PublishedTip).one().result == "won"


def test_unfinished_matches_stay_pending(db_session: Session) -> None:
    match = _match(db_session, kickoff=datetime.utcnow() + timedelta(hours=3))
    publish_tip(db_session, match, TIP)
    assert settle_pending_tips(db_session)["settled"] == 0
    assert db_session.query(PublishedTip).one().result == "pending"


def test_performance_reports_nothing_when_there_is_nothing(db_session: Session) -> None:
    stats = performance(db_session)
    assert stats["settled"] == 0
    assert stats["roi_percent"] is None
    assert "note" in stats


def test_performance_computes_strike_rate_and_roi(db_session: Session) -> None:
    now = datetime.utcnow()
    # A real match: PostgreSQL enforces the foreign key that SQLite ignores.
    match = _match(db_session, kickoff=now)
    # Two winners at 1.8, two losers: staked 4, returned 3.6 -> -10% ROI.
    for i, (result, odds) in enumerate(
        [("won", 1.8), ("won", 1.8), ("lost", 1.8), ("lost", 1.8)]
    ):
        db_session.add(PublishedTip(
            match_id=match.id, market="home_win", selection=str(i), probability=0.6, odds=odds,
            published_at=now, kickoff_time=now, result=result,
        ))
    db_session.commit()

    stats = performance(db_session)
    assert stats["settled"] == 4
    assert stats["strike_rate"] == 50.0
    assert stats["roi_percent"] == -10.0
    assert stats["profit_units"] == -0.4


def test_pending_tips_are_excluded_from_performance(db_session: Session) -> None:
    now = datetime.utcnow()
    match = _match(db_session, kickoff=now)
    db_session.add(PublishedTip(match_id=match.id, market="m", selection="1", probability=0.6,
                                odds=2.0, published_at=now, kickoff_time=now,
                                result="pending"))
    db_session.commit()
    assert performance(db_session)["settled"] == 0
