import pyotp
from fastapi.testclient import TestClient

from app.models.entities import User
from tests.conftest import admin_login


def test_admin_login_without_otp(client: TestClient, admin_user: User) -> None:
    response = client.post(
        "/admin/auth/login",
        json={"email": admin_user.email, "password": "password123"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["csrf_token"]
    assert body["user"]["email"] == admin_user.email
    assert client.cookies.get("admin_access_token")
    assert client.cookies.get("admin_refresh_token")
    assert client.cookies.get("admin_csrf_token")


def test_admin_login_requires_otp_when_2fa_enabled(
    client: TestClient, admin_user_2fa: tuple[User, str]
) -> None:
    user, _ = admin_user_2fa
    response = client.post(
        "/admin/auth/login",
        json={"email": user.email, "password": "password123"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "OTP code required"


def test_admin_login_with_valid_otp(client: TestClient, admin_user_2fa: tuple[User, str]) -> None:
    user, secret = admin_user_2fa
    otp = pyotp.TOTP(secret).now()
    response = client.post(
        "/admin/auth/login",
        json={"email": user.email, "password": "password123", "otp_code": otp},
    )
    assert response.status_code == 200
    assert response.json()["user"]["email"] == user.email


def test_admin_login_rejects_invalid_otp(client: TestClient, admin_user_2fa: tuple[User, str]) -> None:
    user, _ = admin_user_2fa
    response = client.post(
        "/admin/auth/login",
        json={"email": user.email, "password": "password123", "otp_code": "000000"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid OTP code"


def test_cookie_session_me_lifecycle(client: TestClient, admin_user: User) -> None:
    admin_login(client, admin_user.email, "password123")

    me = client.get("/admin/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == admin_user.email

    logout = client.post(
        "/admin/auth/logout",
        headers={"x-csrf-token": client.cookies.get("admin_csrf_token")},
    )
    assert logout.status_code == 200

    me_after = client.get("/admin/auth/me")
    assert me_after.status_code == 401


def test_refresh_rotates_session_cookies(client: TestClient, admin_user: User) -> None:
    admin_login(client, admin_user.email, "password123")
    old_access = client.cookies.get("admin_access_token")

    refresh = client.post("/admin/auth/refresh")
    assert refresh.status_code == 200
    assert client.cookies.get("admin_access_token")
    assert client.cookies.get("admin_csrf_token")

    me = client.get("/admin/auth/me")
    assert me.status_code == 200


def test_csrf_rejects_unsafe_cookie_request_without_header(
    client: TestClient, admin_user: User
) -> None:
    admin_login(client, admin_user.email, "password123")

    response = client.post(
        "/admin/notifications/send",
        json={"title": "Test", "body": "Hello", "audience": "all"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "CSRF validation failed"


def test_csrf_accepts_unsafe_cookie_request_with_valid_header(
    client: TestClient, admin_user: User
) -> None:
    admin_login(client, admin_user.email, "password123")
    csrf = client.cookies.get("admin_csrf_token")

    response = client.post(
        "/admin/notifications/send",
        json={"title": "Test", "body": "Hello", "audience": "all"},
        headers={"x-csrf-token": csrf},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "queued"


def test_csrf_exempts_login_and_refresh(client: TestClient, admin_user: User) -> None:
    response = client.post(
        "/admin/auth/login",
        json={"email": admin_user.email, "password": "password123"},
    )
    assert response.status_code == 200

    refresh = client.post("/admin/auth/refresh")
    assert refresh.status_code == 200
