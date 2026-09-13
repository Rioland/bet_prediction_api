"""Regression tests for the auth findings fixed in the security review."""

import pyotp
import pytest
from fastapi.testclient import TestClient

from app.core.rate_limit import limiter
from app.models.entities import User
from tests.conftest import TEST_PASSWORD


@pytest.fixture()
def rate_limited():
    """Re-enable the limiter (disabled globally in conftest) for one test."""
    limiter.reset()
    limiter.enabled = True
    yield
    limiter.enabled = False
    limiter.reset()


def _login(client: TestClient, email: str, otp: str | None = None):
    payload = {"email": email, "password": TEST_PASSWORD}
    if otp:
        payload["otp_code"] = otp
    return client.post("/auth/login", json=payload)


# --- Finding 1: 2FA was bypassable through the public login endpoint ---------


def test_public_login_requires_otp_when_2fa_enabled(
    client: TestClient, admin_user_2fa: tuple[User, str]
) -> None:
    user, _ = admin_user_2fa
    response = _login(client, user.email)
    assert response.status_code == 401
    assert response.json()["detail"] == "OTP code required"


def test_public_login_rejects_wrong_otp(
    client: TestClient, admin_user_2fa: tuple[User, str]
) -> None:
    user, _ = admin_user_2fa
    response = _login(client, user.email, otp="000000")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid OTP code"


def test_public_login_succeeds_with_valid_otp(
    client: TestClient, admin_user_2fa: tuple[User, str]
) -> None:
    user, secret = admin_user_2fa
    response = _login(client, user.email, otp=pyotp.TOTP(secret).now())
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_public_login_unaffected_when_2fa_disabled(
    client: TestClient, admin_user: User
) -> None:
    assert _login(client, admin_user.email).status_code == 200


# --- Finding 2: refresh tokens authenticated ordinary requests ---------------


def test_refresh_token_is_rejected_as_a_bearer_token(
    client: TestClient, admin_user: User
) -> None:
    tokens = _login(client, admin_user.email).json()
    client.cookies.clear()  # force bearer auth, not the cookie set by login

    response = client.post(
        "/notifications/register-device",
        json={"token": "device-abc", "platform": "ios"},
        headers={"Authorization": f"Bearer {tokens['refresh_token']}"},
    )
    assert response.status_code == 401


def test_access_token_still_authenticates(client: TestClient, admin_user: User) -> None:
    tokens = _login(client, admin_user.email).json()
    client.cookies.clear()

    response = client.post(
        "/notifications/register-device",
        json={"token": "device-abc", "platform": "ios"},
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert response.status_code == 200


def test_refresh_endpoint_still_accepts_refresh_tokens(
    client: TestClient, admin_user: User
) -> None:
    tokens = _login(client, admin_user.email).json()
    response = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert response.status_code == 200
    assert response.json()["access_token"]


# --- Finding 4: CSRF middleware only covered /admin/* paths ------------------


def test_cookie_auth_on_non_admin_route_requires_csrf_token(
    client: TestClient, admin_user: User
) -> None:
    client.post("/admin/auth/login", json={"email": admin_user.email, "password": TEST_PASSWORD})
    assert client.cookies.get("admin_access_token")

    response = client.post(
        "/notifications/register-device",
        json={"token": "device-xyz", "platform": "android"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "CSRF validation failed"


def test_cookie_auth_on_non_admin_route_succeeds_with_csrf_token(
    client: TestClient, admin_user: User
) -> None:
    login = client.post(
        "/admin/auth/login", json={"email": admin_user.email, "password": TEST_PASSWORD}
    )
    csrf = login.json()["csrf_token"]

    response = client.post(
        "/notifications/register-device",
        json={"token": "device-xyz", "platform": "android"},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200


def test_mismatched_csrf_token_is_rejected(client: TestClient, admin_user: User) -> None:
    client.post("/admin/auth/login", json={"email": admin_user.email, "password": TEST_PASSWORD})

    response = client.post(
        "/notifications/register-device",
        json={"token": "device-xyz", "platform": "android"},
        headers={"X-CSRF-Token": "not-the-cookie-value"},
    )
    assert response.status_code == 403


# --- Finding 3: no rate limit on credential endpoints ------------------------


def test_login_is_rate_limited(client: TestClient, admin_user: User, rate_limited) -> None:
    statuses = [
        client.post(
            "/auth/login", json={"email": admin_user.email, "password": "wrong-password"}
        ).status_code
        for _ in range(7)
    ]
    assert 429 in statuses, f"expected a 429 among {statuses}"


# --- Finding 10: case-variant signup raised a 500 instead of a 400 -----------


def test_duplicate_email_differing_in_case_is_rejected_cleanly(client: TestClient) -> None:
    first = client.post(
        "/auth/register",
        json={"name": "A", "email": "Case@Example.com", "password": TEST_PASSWORD},
    )
    assert first.status_code == 200

    second = client.post(
        "/auth/register",
        json={"name": "B", "email": "case@example.com", "password": TEST_PASSWORD},
    )
    assert second.status_code == 400
    assert second.json()["detail"] == "Email already in use"
