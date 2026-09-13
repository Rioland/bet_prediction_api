"""GET /predictions/{sport} and POST /operations/retrain."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.auth import create_access_token, create_refresh_token, hash_password
from app.database import get_db
from app.models import League, Match, Team, User
from main import app


@pytest.fixture()
def client(db_session: Session) -> TestClient:
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _user(db: Session, email: str, role: str = "user") -> User:
    user = User(name=email, email=email, password_hash=hash_password("password1234"), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _bearer(user: User) -> dict:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


@pytest.fixture()
def football_card(db_session: Session):
    league = League(external_id=1, name="Premier League", country="England", sport="football")
    db_session.add(league)
    teams = [Team(name=f"F{i}") for i in range(6)]
    db_session.add_all(teams)
    db_session.flush()

    base = datetime.utcnow() - timedelta(days=60)
    for i in range(40):
        db_session.add(Match(
            external_id=10_000 + i, sport="football", league_id=league.id,
            home_team_id=teams[i % 6].id, away_team_id=teams[(i + 2) % 6].id,
            kickoff_time=base + timedelta(days=i), status="FT", season=2026,
            home_goals=i % 4, away_goals=(i + 1) % 3,
            odds_home=2.0, odds_draw=3.4, odds_away=3.6,
        ))
    db_session.add(Match(
        external_id=11_111, sport="football", league_id=league.id,
        home_team_id=teams[0].id, away_team_id=teams[1].id, status="NS", season=2026,
        kickoff_time=datetime.utcnow() + timedelta(hours=6),
        odds_home=1.9, odds_draw=3.5, odds_away=4.1,
    ))
    db_session.commit()


# --- sport routing -----------------------------------------------------------


def test_sport_index_lists_both_sports(client: TestClient) -> None:
    body = client.get("/predictions").json()
    names = {s["name"] for s in body["sports"]}
    assert names == {"football", "basketball"}

    football = next(s for s in body["sports"] if s["name"] == "football")
    basketball = next(s for s in body["sports"] if s["name"] == "basketball")
    assert football["has_draw"] is True
    assert basketball["has_draw"] is False


def test_unknown_sport_is_404_not_500(client: TestClient) -> None:
    response = client.get("/predictions/tennis")
    assert response.status_code == 404
    assert "football" in response.json()["detail"]


def test_predictions_return_outcome_and_confidence(
    client: TestClient, football_card, trained_models
) -> None:
    body = client.get("/predictions/football").json()
    assert body["sport"] == "football"

    for prediction in body["predictions"]:
        assert prediction["predicted_outcome"] in {"H", "D", "A"}
        # Confidence is a probability, not a rating.
        assert 0 < prediction["confidence"] <= 1
        assert prediction["confidence"] == pytest.approx(
            max(prediction["probabilities"].values()), abs=0.01
        )
        assert prediction["predicted_label"]


def test_predictions_are_sorted_by_confidence(
    client: TestClient, football_card, trained_models
) -> None:
    confidences = [p["confidence"] for p in client.get("/predictions/football").json()["predictions"]]
    assert confidences == sorted(confidences, reverse=True)


def test_confidence_filter_is_applied(client: TestClient, football_card, trained_models) -> None:
    assert client.get("/predictions/football?min_confidence=0.999").json()["predictions"] == []


def test_thin_history_is_reported_not_hidden(client: TestClient, db_session, trained_models) -> None:
    """A short list must be distinguishable from a missing schedule."""
    league = League(external_id=9, name="New", country="X", sport="football")
    db_session.add(league)
    teams = [Team(name="New A"), Team(name="New B")]
    db_session.add_all(teams)
    db_session.flush()
    db_session.add(Match(
        external_id=12_345, sport="football", league_id=league.id,
        home_team_id=teams[0].id, away_team_id=teams[1].id, status="NS", season=2026,
        kickoff_time=datetime.utcnow() + timedelta(hours=4),
    ))
    db_session.commit()

    body = client.get("/predictions/football").json()
    assert body["skipped"] == 1
    assert "completed matches" in body["skipped_reason"]


def test_basketball_returns_empty_without_data(client: TestClient, trained_models) -> None:
    body = client.get("/predictions/basketball").json()
    assert body["sport"] == "basketball"
    assert body["count"] == 0


def test_football_fixtures_do_not_leak_into_basketball(
    client: TestClient, football_card, trained_models
) -> None:
    assert client.get("/predictions/basketball").json()["predictions"] == []


# --- retrain -----------------------------------------------------------------


def test_retrain_requires_authentication(client: TestClient) -> None:
    assert client.post("/operations/retrain").status_code == 401


def test_retrain_rejects_non_admins(client: TestClient, db_session: Session) -> None:
    plain = _user(db_session, "plain@test.com", role="user")
    assert client.post("/operations/retrain", headers=_bearer(plain)).status_code == 403


def test_retrain_rejects_a_refresh_token(client: TestClient, db_session: Session) -> None:
    admin = _user(db_session, "admin1@test.com", role="admin")
    headers = {"Authorization": f"Bearer {create_refresh_token(admin.id)}"}
    assert client.post("/operations/retrain", headers=headers).status_code == 401


def test_retrain_refuses_when_there_is_too_little_history(
    client: TestClient, db_session: Session, football_card
) -> None:
    """40 completed matches is far below the training minimum."""
    admin = _user(db_session, "admin2@test.com", role="admin")
    response = client.post("/operations/retrain", headers=_bearer(admin))

    assert response.status_code == 409
    assert "are needed" in response.json()["detail"]


def test_retrain_rejects_an_unknown_sport(client: TestClient, db_session: Session) -> None:
    admin = _user(db_session, "admin3@test.com", role="admin")
    assert client.post(
        "/operations/retrain?sport=chess", headers=_bearer(admin)
    ).status_code == 404


def test_operations_sports_reports_trainability(client: TestClient, db_session, football_card) -> None:
    admin = _user(db_session, "admin4@test.com", role="admin")
    body = client.get("/operations/sports", headers=_bearer(admin)).json()

    football = next(s for s in body["sports"] if s["sport"] == "football")
    assert football["completed_matches"] == 40
    assert football["trainable"] is False
    assert football["minimum_required"] > 40
