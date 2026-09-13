"""Customer accounts, checkout, callbacks and code gating, end to end."""

from datetime import datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.auth import create_access_token, create_refresh_token, hash_password
from app.database import get_db
from app.models import BettingSlip, Payment, Subscription, User
from app.rate_limit import limiter
from app.routes import account
from app.services.opay import OPayClient, OPayConfig, compute_callback_signature
from main import app

SECRET = "OPAYPRV-route-tests"


def _gateway(status="SUCCESS", total=500_000):
    def handler(request):
        if request.url.path.endswith("/create"):
            return httpx.Response(200, json={"code": "00000", "data": {
                "orderNo": "ORD1", "cashierUrl": "https://sandboxcashier.opaycheckout.com/c?t=1"}})
        return httpx.Response(200, json={"code": "00000", "data": {
            "status": status, "amount": {"total": total, "currency": "NGN"}}})
    return OPayClient(OPayConfig("256612345678901", "OPAYPUB-test", SECRET),
                      transport=httpx.MockTransport(handler))


@pytest.fixture()
def client(db_session: Session):
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[account.opay_client] = lambda: _gateway()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _register(client, email="fan@test.com", password="goodpassword"):
    return client.post("/auth/register", json={"name": "Fan", "email": email, "password": password})


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# --- registration and login --------------------------------------------------


def test_registration_returns_a_session(client) -> None:
    body = _register(client).json()
    assert body["access_token"] and body["refresh_token"]
    assert body["user"]["role"] == "user"


def test_registration_cannot_self_assign_a_role(client, db_session) -> None:
    client.post("/auth/register", json={
        "name": "Sneaky", "email": "sneaky@test.com", "password": "goodpassword",
        "role": "admin", "subscription_type": "premium"})
    user = db_session.query(User).filter_by(email="sneaky@test.com").one()
    assert user.role == "user"
    assert user.subscription_type == "free"


def test_duplicate_email_is_refused_case_insensitively(client) -> None:
    _register(client, "Case@Test.com")
    assert _register(client, "case@test.com").status_code == 409


@pytest.mark.parametrize("payload", [
    {"name": "A", "email": "not-an-email", "password": "goodpassword"},
    {"name": "A", "email": "a@test.com", "password": "short"},
    {"name": "   ", "email": "a@test.com", "password": "goodpassword"},
])
def test_invalid_registrations_are_rejected(client, payload) -> None:
    assert client.post("/auth/register", json=payload).status_code == 422


def test_login_succeeds_with_correct_credentials(client) -> None:
    _register(client)
    assert client.post("/auth/login", json={"email": "fan@test.com",
                                           "password": "goodpassword"}).status_code == 200


def test_login_gives_one_message_whether_or_not_the_account_exists(client) -> None:
    """Otherwise the endpoint reveals which emails are registered."""
    _register(client)
    wrong_password = client.post("/auth/login", json={"email": "fan@test.com", "password": "nope"})
    no_account = client.post("/auth/login", json={"email": "ghost@test.com", "password": "nope"})
    assert wrong_password.status_code == no_account.status_code == 401
    assert wrong_password.json()["detail"] == no_account.json()["detail"]


def test_suspended_accounts_cannot_sign_in(client, db_session) -> None:
    _register(client)
    user = db_session.query(User).filter_by(email="fan@test.com").one()
    user.status = "suspended"
    db_session.commit()
    assert client.post("/auth/login", json={"email": "fan@test.com",
                                           "password": "goodpassword"}).status_code == 403


def test_me_requires_an_access_token_not_a_refresh_token(client, db_session) -> None:
    token = _register(client).json()
    assert client.get("/auth/me", headers=_auth(token["access_token"])).status_code == 200
    assert client.get("/auth/me", headers=_auth(token["refresh_token"])).status_code == 401
    assert client.get("/auth/me").status_code == 401


def test_login_is_rate_limited(client) -> None:
    limiter.reset()
    limiter.enabled = True
    try:
        codes = [client.post("/auth/login", json={"email": "x@test.com", "password": "p"}).status_code
                 for _ in range(15)]
    finally:
        limiter.enabled = False
        limiter.reset()
    assert 429 in codes


# --- plan and checkout -------------------------------------------------------


def test_plan_is_public_and_defaults_to_five_thousand(client) -> None:
    body = client.get("/subscription/plan").json()
    assert body["price_naira"] == 5000
    assert body["currency"] == "NGN"
    assert body["period_days"] == 30


def test_checkout_requires_sign_in(client) -> None:
    assert client.post("/subscription/checkout").status_code == 401


def test_checkout_returns_an_opay_url(client, db_session) -> None:
    token = _register(client).json()["access_token"]
    body = client.post("/subscription/checkout", headers=_auth(token)).json()
    assert body["checkout_url"].startswith("https://")
    assert body["amount_naira"] == 5000
    assert db_session.query(Payment).one().amount_kobo == 500_000


def test_checkout_reports_when_payments_are_not_configured(client) -> None:
    app.dependency_overrides[account.opay_client] = lambda: OPayClient(OPayConfig("", "", ""))
    token = _register(client).json()["access_token"]
    assert client.post("/subscription/checkout", headers=_auth(token)).status_code == 503


# --- verification and callbacks ----------------------------------------------


def _paid_reference(client) -> tuple[str, str]:
    token = _register(client).json()["access_token"]
    reference = client.post("/subscription/checkout", headers=_auth(token)).json()["reference"]
    return token, reference


def test_verify_activates_the_subscription(client) -> None:
    token, reference = _paid_reference(client)
    body = client.post(f"/subscription/verify/{reference}", headers=_auth(token)).json()
    assert body["status"] == "success"
    assert body["subscription"]["active"] is True


def test_a_user_cannot_verify_someone_elses_payment(client) -> None:
    _, reference = _paid_reference(client)
    other = _register(client, "other@test.com").json()["access_token"]
    assert client.post(f"/subscription/verify/{reference}", headers=_auth(other)).status_code == 404


def test_callback_with_a_valid_signature_activates(client, db_session) -> None:
    _, reference = _paid_reference(client)
    payload = {"amount": "500000", "currency": "NGN", "reference": reference, "refunded": False,
               "status": "SUCCESS", "timestamp": "2026-09-13T10:00:00Z", "token": "T",
               "transactionId": "TX"}
    response = client.post("/payments/opay/callback", json={
        "payload": payload, "sha512": compute_callback_signature(payload, SECRET)})

    assert response.status_code == 200
    assert db_session.query(Subscription).count() == 1


def test_forged_callback_is_rejected_and_grants_nothing(client, db_session) -> None:
    _, reference = _paid_reference(client)
    payload = {"amount": "500000", "currency": "NGN", "reference": reference, "refunded": False,
               "status": "SUCCESS", "timestamp": "2026-09-13T10:00:00Z", "token": "T",
               "transactionId": "TX"}
    response = client.post("/payments/opay/callback", json={
        "payload": payload, "sha512": compute_callback_signature(payload, "attacker")})

    assert response.status_code == 400
    assert db_session.query(Subscription).count() == 0


def test_callback_with_garbage_body_is_rejected(client) -> None:
    assert client.post("/payments/opay/callback", content=b"not json",
                       headers={"Content-Type": "application/json"}).status_code == 400


# --- code gating -------------------------------------------------------------


@pytest.fixture()
def coded_slip(db_session: Session) -> BettingSlip:
    slip = BettingSlip(sport="football", slip_date=datetime.utcnow().date(), tier="banker",
                       label="Banker", legs=[{"fixture_id": 1, "market": "m", "selection": "s"}],
                       total_odds=1.5, combined_probability=0.7, booking_code="REALCODE1")
    db_session.add(slip)
    db_session.commit()
    return slip


def test_anonymous_visitors_see_a_code_exists_but_not_the_code(client, coded_slip) -> None:
    slip = client.get("/slips").json()["slips"][0]
    assert slip["has_code"] is True
    assert slip["code_locked"] is True
    assert slip["booking_code"] is None


def test_free_accounts_do_not_see_codes(client, coded_slip) -> None:
    token = _register(client).json()["access_token"]
    body = client.get("/slips", headers=_auth(token)).json()
    assert body["codes_unlocked"] is False
    assert body["slips"][0]["booking_code"] is None


def test_subscribers_see_codes(client, coded_slip) -> None:
    token, reference = _paid_reference(client)
    client.post(f"/subscription/verify/{reference}", headers=_auth(token))

    body = client.get("/slips", headers=_auth(token)).json()
    assert body["codes_unlocked"] is True
    assert body["slips"][0]["booking_code"] == "REALCODE1"


def test_an_expired_subscriber_loses_the_codes(client, db_session, coded_slip) -> None:
    token = _register(client).json()["access_token"]
    user = db_session.query(User).filter_by(email="fan@test.com").one()
    db_session.add(Subscription(user_id=user.id, started_at=datetime.utcnow() - timedelta(days=40),
                                expires_at=datetime.utcnow() - timedelta(minutes=1)))
    db_session.commit()
    assert client.get("/slips", headers=_auth(token)).json()["slips"][0]["booking_code"] is None


def test_a_refresh_token_does_not_unlock_codes(client, db_session, coded_slip) -> None:
    token, reference = _paid_reference(client)
    client.post(f"/subscription/verify/{reference}", headers=_auth(token))
    user = db_session.query(User).filter_by(email="fan@test.com").one()

    body = client.get("/slips", headers=_auth(create_refresh_token(user.id))).json()
    assert body["slips"][0]["booking_code"] is None


# --- admin price control -----------------------------------------------------


def _admin_token(db_session) -> str:
    admin = User(name="Admin", email="boss@test.com", password_hash=hash_password("adminpass12"),
                 role="admin")
    db_session.add(admin)
    db_session.commit()
    db_session.refresh(admin)
    return create_access_token(admin.id)


def test_admin_can_change_the_price(client, db_session) -> None:
    token = _admin_token(db_session)
    response = client.put("/admin/subscription-price", json={"price_naira": "7000"},
                          headers=_auth(token))
    assert response.status_code == 200
    assert client.get("/subscription/plan").json()["price_naira"] == 7000


def test_customers_cannot_change_the_price(client) -> None:
    token = _register(client).json()["access_token"]
    assert client.put("/admin/subscription-price", json={"price_naira": "1"},
                      headers=_auth(token)).status_code == 403


def test_an_implausible_price_is_refused(client, db_session) -> None:
    token = _admin_token(db_session)
    assert client.put("/admin/subscription-price", json={"price_naira": "5"},
                      headers=_auth(token)).status_code == 422


def test_admin_subscriptions_reports_real_expiry(client, db_session) -> None:
    token, reference = _paid_reference(client)
    client.post(f"/subscription/verify/{reference}", headers=_auth(token))

    rows = client.get("/admin/subscriptions", headers=_auth(_admin_token(db_session))).json()
    assert len(rows) == 1
    assert rows[0]["provider"] == "opay"
    assert rows[0]["status"] == "active"


def test_revenue_counts_only_verified_payments(client, db_session) -> None:
    token, reference = _paid_reference(client)
    _register(client, "unpaid@test.com")  # a checkout started and abandoned
    client.post(f"/subscription/verify/{reference}", headers=_auth(token))

    body = client.get("/admin/analytics/dashboard", headers=_auth(_admin_token(db_session))).json()
    assert body["revenue"] == 5000


# --- moderators ----------------------------------------------------------------


def _staff_token(db_session, role: str) -> str:
    staff = User(name=role, email=f"{role}@staff.test", password_hash=hash_password("staffpass123"),
                 role=role)
    db_session.add(staff)
    db_session.commit()
    db_session.refresh(staff)
    return create_access_token(staff.id)


@pytest.mark.parametrize("method,path,body", [
    ("put", "/admin/subscription-price", {"price_naira": "100"}),
    ("get", "/admin/subscription-price", None),
    ("get", "/admin/payments", None),
    ("patch", "/admin/subscriptions/1/cancel", None),
])
def test_moderators_cannot_touch_money(client, db_session, method, path, body) -> None:
    """Moderators pass the general admin check; pricing and payments need more."""
    token = _staff_token(db_session, "moderator")
    kwargs = {"headers": _auth(token)}
    if body is not None:
        kwargs["json"] = body
    assert getattr(client, method)(path, **kwargs).status_code == 403


def test_moderators_cannot_publish_booking_codes(client, db_session, coded_slip) -> None:
    token = _staff_token(db_session, "moderator")
    assert client.post(f"/admin/slips/{coded_slip.id}/code", json={"booking_code": "HIJACK01"},
                       headers=_auth(token)).status_code == 403
    assert client.delete(f"/admin/slips/{coded_slip.id}/code",
                         headers=_auth(token)).status_code == 403


@pytest.mark.parametrize("role", ["admin", "super_admin"])
def test_full_admins_can_set_the_price(client, db_session, role) -> None:
    token = _staff_token(db_session, role)
    assert client.put("/admin/subscription-price", json={"price_naira": "6000"},
                      headers=_auth(token)).status_code == 200
