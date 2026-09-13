"""Integration tests for the tips, VIP, results and news routes."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.auth import create_access_token, create_refresh_token, hash_password
from app.database import get_db
from app.models import Article, League, Match, Team, User
from main import app


@pytest.fixture()
def client(db_session: Session) -> TestClient:
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _user(db: Session, email: str, **kwargs) -> User:
    user = User(name=email, email=email, password_hash=hash_password("password1234"), **kwargs)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _bearer(user: User) -> dict:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


@pytest.fixture()
def fixtures(db_session: Session) -> League:
    league = League(external_id=2021, name="Premier League", country="England")
    db_session.add(league)
    teams = [Team(name=f"Team {i}") for i in range(6)]
    db_session.add_all(teams)
    db_session.flush()

    base = datetime.utcnow() - timedelta(days=50)
    for i in range(40):
        db_session.add(Match(
            external_id=3000 + i, league_id=league.id,
            home_team_id=teams[i % 6].id, away_team_id=teams[(i + 2) % 6].id,
            kickoff_time=base + timedelta(days=i), status="FT", season=2025,
            home_goals=i % 3, away_goals=(i + 1) % 2,
            odds_home=2.1, odds_draw=3.4, odds_away=3.2,
        ))
    db_session.add(Match(
        external_id=5555, league_id=league.id, home_team_id=teams[0].id,
        away_team_id=teams[1].id, status="NS", season=2025,
        kickoff_time=datetime.utcnow().replace(hour=20, minute=0, second=0, microsecond=0),
        odds_home=1.9, odds_draw=3.5, odds_away=4.2,
    ))
    db_session.commit()
    return league


def test_markets_are_listed(client: TestClient) -> None:
    markets = client.get("/football/markets").json()
    assert "popular" in markets and "btts" in markets


def test_tips_returns_cards_for_today(client: TestClient, fixtures, trained_models) -> None:
    body = client.get("/football/tips?market=popular").json()
    assert body["market"] == "popular"
    assert isinstance(body["matches"], list)
    for card in body["matches"]:
        assert card["tip"]["probability"] >= 0.55
        assert card["home_team"] and card["away_team"]


def test_unknown_market_is_rejected(client: TestClient, trained_models) -> None:
    assert client.get("/football/tips?market=nope").status_code == 422


def test_anonymous_sees_paywall_not_picks(client: TestClient, fixtures, trained_models) -> None:
    body = client.get("/football/tips").json()
    assert body["vip_locked"] is True
    assert "vip_matches" not in body
    assert "vip_count" in body


def test_free_user_stays_locked(client: TestClient, fixtures, db_session, trained_models) -> None:
    user = _user(db_session, "free@test.com", role="user", subscription_type="free")
    body = client.get("/football/tips", headers=_bearer(user)).json()
    assert body["vip_locked"] is True


def test_premium_user_is_unlocked(client: TestClient, fixtures, db_session, trained_models) -> None:
    user = _user(db_session, "vip@test.com", role="user", subscription_type="premium")
    body = client.get("/football/tips", headers=_bearer(user)).json()
    assert body["vip_locked"] is False


def test_vip_endpoint_needs_auth_then_subscription(
    client: TestClient, fixtures, db_session, trained_models
) -> None:
    assert client.get("/football/vip/tips").status_code == 401
    free = _user(db_session, "free2@test.com", role="user", subscription_type="free")
    assert client.get("/football/vip/tips", headers=_bearer(free)).status_code == 402


def test_refresh_token_cannot_unlock_vip(
    client: TestClient, fixtures, db_session, trained_models
) -> None:
    """A refresh token is long lived and must not authenticate a request."""
    user = _user(db_session, "vip2@test.com", role="user", subscription_type="premium")
    headers = {"Authorization": f"Bearer {create_refresh_token(user.id)}"}
    assert client.get("/football/vip/tips", headers=headers).status_code == 401


def test_performance_reports_empty_record_honestly(client: TestClient) -> None:
    body = client.get("/football/results/performance").json()
    assert body["overall"]["settled"] == 0
    assert body["overall"]["roi_percent"] is None


def test_news_lists_only_published(client: TestClient, db_session: Session) -> None:
    db_session.add_all([
        Article(slug="live", title="Live", body="b", published=True, published_at=datetime.utcnow()),
        Article(slug="draft", title="Draft", body="b", published=False),
    ])
    db_session.commit()
    assert [a["slug"] for a in client.get("/news").json()] == ["live"]
    assert client.get("/news/draft").status_code == 404


def test_creating_news_requires_admin(client: TestClient, db_session: Session) -> None:
    payload = {"slug": "s", "title": "t", "body": "b", "published": True}
    assert client.post("/news", json=payload).status_code == 401

    plain = _user(db_session, "plain@test.com", role="user")
    assert client.post("/news", json=payload, headers=_bearer(plain)).status_code == 403

    admin = _user(db_session, "admin@test.com", role="admin")
    assert client.post("/news", json=payload, headers=_bearer(admin)).status_code == 201


def test_fixtures_with_no_team_history_are_not_tipped(
    client: TestClient, db_session: Session, trained_models
) -> None:
    """Unknown teams get neutral priors, which would yield a confident-looking
    identical tip for every such fixture."""
    league = League(external_id=9999, name="New League", country="Nowhere")
    db_session.add(league)
    unknown = [Team(name="Unknown A"), Team(name="Unknown B")]
    db_session.add_all(unknown)
    db_session.flush()
    db_session.add(Match(
        external_id=7001, league_id=league.id, home_team_id=unknown[0].id,
        away_team_id=unknown[1].id, status="NS", season=2025,
        kickoff_time=datetime.utcnow().replace(hour=19, minute=0, second=0, microsecond=0),
        odds_home=2.0, odds_draw=3.4, odds_away=3.6,
    ))
    db_session.commit()

    fixture_ids = [c["fixture_id"] for c in client.get("/football/tips").json()["matches"]]
    assert 7001 not in fixture_ids
