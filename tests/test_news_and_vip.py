"""News CRUD and VIP gating."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.entities import Article, League, Match, SubscriptionType, Team, User
from app.core.security import hash_password
from tests.conftest import TEST_PASSWORD

ARTICLE = {"slug": "home-and-away-records", "title": "Home and Away Records",
           "body": "Full text.", "excerpt": "Why the split matters.", "published": True}


def _auth(client: TestClient, email: str) -> dict:
    tokens = client.post("/auth/login", json={"email": email, "password": TEST_PASSWORD}).json()
    client.cookies.clear()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.fixture()
def free_user(db_session: Session) -> User:
    user = User(name="Free", email="free@test.com", password_hash=hash_password(TEST_PASSWORD))
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture()
def vip_user(db_session: Session) -> User:
    user = User(name="Vip", email="vip@test.com", password_hash=hash_password(TEST_PASSWORD),
                subscription_type=SubscriptionType.PREMIUM)
    db_session.add(user)
    db_session.commit()
    return user


# --- News ------------------------------------------------------------------


def test_only_published_articles_are_listed(client: TestClient, db_session: Session) -> None:
    db_session.add_all([
        Article(slug="live", title="Live", body="b", published=True, published_at=datetime.utcnow()),
        Article(slug="draft", title="Draft", body="b", published=False),
    ])
    db_session.commit()

    slugs = [a["slug"] for a in client.get("/api/news").json()]
    assert slugs == ["live"]


def test_article_detail_returns_body(client: TestClient, db_session: Session) -> None:
    db_session.add(Article(slug="live", title="Live", body="The full text.",
                           published=True, published_at=datetime.utcnow()))
    db_session.commit()
    assert client.get("/api/news/live").json()["body"] == "The full text."


def test_unpublished_article_is_not_readable(client: TestClient, db_session: Session) -> None:
    db_session.add(Article(slug="draft", title="Draft", body="b", published=False))
    db_session.commit()
    assert client.get("/api/news/draft").status_code == 404


def test_creating_an_article_requires_admin(client: TestClient, free_user: User) -> None:
    assert client.post("/api/news", json=ARTICLE).status_code == 401
    response = client.post("/api/news", json=ARTICLE, headers=_auth(client, free_user.email))
    assert response.status_code == 403


def test_admin_can_create_and_publish(client: TestClient, admin_user: User) -> None:
    response = client.post("/api/news", json=ARTICLE, headers=_auth(client, admin_user.email))
    assert response.status_code == 201
    assert response.json()["published_at"] is not None
    assert client.get("/api/news/home-and-away-records").status_code == 200


def test_duplicate_slug_is_rejected(client: TestClient, admin_user: User) -> None:
    headers = _auth(client, admin_user.email)
    client.post("/api/news", json=ARTICLE, headers=headers)
    assert client.post("/api/news", json=ARTICLE, headers=headers).status_code == 400


def test_publish_date_is_stamped_once(client: TestClient, admin_user: User, db_session: Session) -> None:
    headers = _auth(client, admin_user.email)
    client.post("/api/news", json={**ARTICLE, "published": False}, headers=headers)
    first = client.patch(f"/api/news/{ARTICLE['slug']}", json=ARTICLE, headers=headers).json()
    second = client.patch(f"/api/news/{ARTICLE['slug']}", json=ARTICLE, headers=headers).json()
    assert first["published_at"] == second["published_at"]


# --- VIP gating ------------------------------------------------------------


@pytest.fixture()
def fixtures_today(db_session: Session) -> None:
    league = League(external_id=39, name="PL", country="England")
    db_session.add(league)
    teams = [Team(external_id=100 + i, name=f"T{i}") for i in range(4)]
    db_session.add_all(teams)
    db_session.flush()

    base = datetime.utcnow() - timedelta(days=40)
    for i in range(30):
        db_session.add(Match(external_id=500 + i, league_id=league.id,
                             home_team_id=teams[i % 4].id, away_team_id=teams[(i + 1) % 4].id,
                             kickoff_time=base + timedelta(days=i), status="FT",
                             season=2024, home_goals=i % 3, away_goals=(i + 1) % 2))
    db_session.add(Match(external_id=8888, league_id=league.id, home_team_id=teams[0].id,
                         away_team_id=teams[1].id, status="NS", season=2024,
                         kickoff_time=datetime.utcnow().replace(hour=21, minute=0, second=0, microsecond=0)))
    db_session.commit()


def test_anonymous_callers_see_the_paywall_not_the_picks(
    client: TestClient, fixtures_today, trained_models
) -> None:
    body = client.get("/api/football/tips?market=popular").json()
    assert body["vip_locked"] is True
    assert "vip_matches" not in body
    assert "vip_count" in body  # the count is disclosed, the picks are not


def test_free_user_is_still_locked(client: TestClient, fixtures_today, free_user, trained_models) -> None:
    body = client.get("/api/football/tips", headers=_auth(client, free_user.email)).json()
    assert body["vip_locked"] is True
    assert "vip_matches" not in body


def test_premium_subscriber_is_unlocked(client: TestClient, fixtures_today, vip_user, trained_models) -> None:
    body = client.get("/api/football/tips", headers=_auth(client, vip_user.email)).json()
    assert body["vip_locked"] is False


def test_vip_endpoint_rejects_free_users_with_402(
    client: TestClient, fixtures_today, free_user, trained_models
) -> None:
    response = client.get("/api/football/vip/tips", headers=_auth(client, free_user.email))
    assert response.status_code == 402


def test_vip_endpoint_requires_authentication(client: TestClient, trained_models) -> None:
    assert client.get("/api/football/vip/tips").status_code == 401


def test_unknown_market_is_rejected(client: TestClient, trained_models) -> None:
    assert client.get("/api/football/tips?market=nonsense").status_code == 422
