"""Integration tests for the /api/football endpoints the frontend consumes."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.entities import League, Match, Team


@pytest.fixture()
def league_with_fixtures(db_session: Session) -> League:
    """A league with played history plus one fixture kicking off today."""
    league = League(external_id=39, name="Premier League", country="England")
    db_session.add(league)
    teams = [Team(external_id=100 + i, name=f"Team {i}") for i in range(6)]
    db_session.add_all(teams)
    db_session.flush()

    base = datetime.utcnow() - timedelta(days=60)
    for i in range(40):
        home, away = teams[i % 6], teams[(i + 3) % 6]
        db_session.add(Match(
            external_id=9000 + i, league_id=league.id, home_team_id=home.id, away_team_id=away.id,
            kickoff_time=base + timedelta(days=i), status="FT", season=2024,
            home_goals=(i % 4), away_goals=((i + 1) % 3),
        ))

    db_session.add(Match(
        external_id=7777, league_id=league.id, home_team_id=teams[0].id, away_team_id=teams[1].id,
        kickoff_time=datetime.utcnow().replace(hour=20, minute=0, second=0, microsecond=0),
        status="NS", season=2024,
    ))
    db_session.commit()
    return league


def test_healthz(client: TestClient) -> None:
    assert client.get("/api/healthz").json() == {"status": "ok"}


def test_leagues_endpoint(client: TestClient, league_with_fixtures: League) -> None:
    body = client.get("/api/football/leagues").json()
    assert len(body) == 1
    assert body[0]["name"] == "Premier League"
    assert body[0]["country"] == "England"


def test_predictions_today_matches_the_frontend_contract(
    client: TestClient, league_with_fixtures: League, trained_models
) -> None:
    response = client.get("/api/football/predictions/today")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1

    card = body[0]
    # Exactly the fields lib/api-spec/openapi.yaml declares required.
    for field in ("fixture_id", "league_id", "league_name", "home_team", "away_team", "kickoff", "status", "prediction"):
        assert field in card, f"missing {field}"

    prediction = card["prediction"]
    for field in ("home_win_prob", "draw_prob", "away_win_prob", "btts_prob", "over_25_prob",
                  "predicted_winner", "confidence", "home_xg", "away_xg", "predicted_score"):
        assert field in prediction, f"missing prediction.{field}"

    total = prediction["home_win_prob"] + prediction["draw_prob"] + prediction["away_win_prob"]
    assert total == pytest.approx(1.0, abs=0.01)
    assert prediction["predicted_winner"] in {"home", "draw", "away"}
    # 0-1: the card renders this as `confidence * 10` out of 10.
    assert 0 < prediction["confidence"] <= 1
    assert prediction["confidence"] == pytest.approx(max(
        prediction["home_win_prob"], prediction["draw_prob"], prediction["away_win_prob"]
    ), abs=0.01)


def test_confidence_filter_excludes_low_probability_picks(
    client: TestClient, league_with_fixtures: League, trained_models
) -> None:
    assert client.get("/api/football/predictions/today?min_confidence=0.999").json() == []
    assert len(client.get("/api/football/predictions/today?min_confidence=0").json()) == 1


def test_prediction_by_fixture_id(client: TestClient, league_with_fixtures: League, trained_models) -> None:
    card = client.get("/api/football/predictions/7777").json()
    assert card["fixture_id"] == 7777
    assert card["home_team"] == "Team 0"


def test_unknown_fixture_returns_404(client: TestClient, trained_models) -> None:
    assert client.get("/api/football/predictions/424242").status_code == 404


def test_league_filter_is_applied(client: TestClient, league_with_fixtures: League, trained_models) -> None:
    other = client.get("/api/football/predictions/today?league_id=99999").json()
    assert other == []
    same = client.get(f"/api/football/predictions/today?league_id={league_with_fixtures.id}").json()
    assert len(same) == 1


def test_endpoints_report_503_when_models_are_missing(
    client: TestClient, league_with_fixtures: League, tmp_path, monkeypatch
) -> None:
    from app.core.config import settings
    from app.services import prediction_service

    monkeypatch.setattr(settings, "model_dir", str(tmp_path / "empty"))
    prediction_service.clear_model_cache()
    response = client.get("/api/football/predictions/today")
    assert response.status_code == 503
    assert "not trained" in response.json()["detail"]
    prediction_service.clear_model_cache()
