"""Insert demo leagues, teams, matches, and predictions when the database is empty."""

from datetime import date, datetime, timedelta

from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.models.entities import League, Match, Prediction, Team


def seed_if_empty() -> int:
    db = SessionLocal()
    try:
        count = db.scalar(select(func.count()).select_from(Match)) or 0
        if count > 0:
            return 0

        league = League(id=1, external_id=39, name="Premier League", country="England")
        teams = [
            Team(id=1, external_id=33, name="Manchester United"),
            Team(id=2, external_id=40, name="Liverpool"),
            Team(id=3, external_id=49, name="Chelsea"),
            Team(id=4, external_id=50, name="Manchester City"),
        ]
        db.add(league)
        db.add_all(teams)
        db.flush()

        today = date.today()
        kickoffs = [
            datetime(today.year, today.month, today.day, 15, 0),
            datetime(today.year, today.month, today.day, 17, 30),
            datetime(today.year, today.month, today.day, 20, 0),
        ]
        matches = [
            Match(
                league_id=1,
                home_team_id=1,
                away_team_id=2,
                kickoff_time=kickoffs[0],
                status="scheduled",
            ),
            Match(
                league_id=1,
                home_team_id=3,
                away_team_id=4,
                kickoff_time=kickoffs[1],
                status="live",
            ),
            Match(
                league_id=1,
                home_team_id=2,
                away_team_id=3,
                kickoff_time=kickoffs[2],
                status="scheduled",
            ),
        ]
        db.add_all(matches)
        db.flush()

        predictions = [
            Prediction(
                match_id=matches[0].id,
                prediction_type="match_result",
                prediction="home_win",
                confidence=62.5,
                probabilities={"home_win": 0.625, "draw": 0.22, "away_win": 0.155},
            ),
            Prediction(
                match_id=matches[1].id,
                prediction_type="match_result",
                prediction="draw",
                confidence=48.0,
                probabilities={"home_win": 0.31, "draw": 0.48, "away_win": 0.21},
            ),
            Prediction(
                match_id=matches[2].id,
                prediction_type="match_result",
                prediction="away_win",
                confidence=55.0,
                probabilities={"home_win": 0.25, "draw": 0.2, "away_win": 0.55},
            ),
        ]
        db.add_all(predictions)
        db.commit()
        return len(matches)
    finally:
        db.close()


if __name__ == "__main__":
    created = seed_if_empty()
    print(f"Seeded {created} demo matches" if created else "Database already has matches")
