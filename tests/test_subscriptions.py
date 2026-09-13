"""Subscription activation.

Every path that could grant access without a verified, correctly-priced
payment is exercised here.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import httpx
import pytest
from sqlalchemy.orm import Session

from app.auth import hash_password
from app.models import Payment, Subscription, User
from app.services import subscriptions as subs
from app.services.opay import OPayClient, OPayConfig, OPayError, compute_callback_signature

SECRET = "OPAYPRV-subscription-tests"
CONFIG = OPayConfig("256612345678901", "OPAYPUB-test", SECRET)


def _user(db: Session, email: str = "buyer@test.com", role: str = "user") -> User:
    user = User(name="Buyer", email=email, password_hash=hash_password("password1234"), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _gateway(status: str = "SUCCESS", total: int = 500_000, currency: str = "NGN",
             calls: list | None = None) -> OPayClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url.path)
        if request.url.path.endswith("/create"):
            return httpx.Response(200, json={"code": "00000", "data": {
                "orderNo": "ORD1", "status": "INITIAL",
                "cashierUrl": "https://sandboxcashier.opaycheckout.com/c?t=1"}})
        return httpx.Response(200, json={"code": "00000", "data": {
            "status": status, "amount": {"total": total, "currency": currency}}})

    return OPayClient(CONFIG, transport=httpx.MockTransport(handler))


def _checkout(db: Session, user: User, client: OPayClient | None = None) -> Payment:
    return subs.start_checkout(
        db, user, client or _gateway(),
        return_url="https://site/return", callback_url="https://api/cb", cancel_url="https://site/x",
    )


def _callback(reference: str, *, status="SUCCESS", amount="500000", secret=SECRET) -> dict:
    payload = {
        "amount": amount, "currency": "NGN", "reference": reference, "refunded": False,
        "status": status, "timestamp": "2026-09-13T10:00:00Z", "token": "T1",
        "transactionId": "TX1",
    }
    return {"payload": payload, "sha512": compute_callback_signature(payload, secret)}


# --- price -------------------------------------------------------------------


def test_price_defaults_to_five_thousand_naira(db_session: Session) -> None:
    assert subs.get_price_naira(db_session) == Decimal("5000")


def test_admin_price_change_persists(db_session: Session) -> None:
    """Settings used to live in a dict that reset on every restart."""
    subs.set_price_naira(db_session, "7500", "admin@test.com")
    db_session.expire_all()
    assert subs.get_price_naira(db_session) == Decimal("7500.00")


@pytest.mark.parametrize("bad", ["50", "0", "-10", "2000000", "abc"])
def test_implausible_prices_are_refused(db_session: Session, bad) -> None:
    with pytest.raises(ValueError):
        subs.set_price_naira(db_session, bad, "admin@test.com")


def test_checkout_charges_the_current_price_in_kobo(db_session: Session) -> None:
    subs.set_price_naira(db_session, "6000", "admin@test.com")
    payment = _checkout(db_session, _user(db_session))
    assert payment.amount_kobo == 600_000
    assert payment.checkout_url.startswith("https://")
    assert payment.status == "pending"


def test_a_price_change_mid_checkout_does_not_affect_that_payment(db_session: Session) -> None:
    """The customer pays, and is verified against, the price they were shown."""
    user = _user(db_session)
    payment = _checkout(db_session, user)            # created at 5000
    subs.set_price_naira(db_session, "9000", "admin")  # admin raises the price

    result = subs.confirm_payment(db_session, payment.reference, _gateway(total=500_000))
    assert result.status == "success"
    assert subs.is_active(db_session, user)


# --- activation --------------------------------------------------------------


def test_no_access_before_payment(db_session: Session) -> None:
    user = _user(db_session)
    _checkout(db_session, user)
    assert subs.is_active(db_session, user) is False


def test_verified_payment_grants_thirty_days(db_session: Session) -> None:
    user = _user(db_session)
    payment = _checkout(db_session, user)
    subs.confirm_payment(db_session, payment.reference, _gateway())

    subscription = subs.get_subscription(db_session, user)
    assert subs.is_active(db_session, user)
    remaining = subscription.expires_at - datetime.utcnow()
    assert timedelta(days=29, hours=23) < remaining <= timedelta(days=30)


def test_an_underpayment_grants_nothing(db_session: Session) -> None:
    """A genuine SUCCESS for the wrong amount must not unlock the subscription."""
    user = _user(db_session)
    payment = _checkout(db_session, user)
    result = subs.confirm_payment(db_session, payment.reference, _gateway(total=100))

    assert result.status == "amount_mismatch"
    assert "Expected 500000" in result.failure_reason
    assert subs.is_active(db_session, user) is False


def test_the_wrong_currency_grants_nothing(db_session: Session) -> None:
    user = _user(db_session)
    payment = _checkout(db_session, user)
    subs.confirm_payment(db_session, payment.reference, _gateway(currency="USD"))
    assert subs.is_active(db_session, user) is False


@pytest.mark.parametrize("state", ["FAIL", "CLOSE", "PENDING", "INITIAL"])
def test_non_successful_states_grant_nothing(db_session: Session, state) -> None:
    user = _user(db_session)
    payment = _checkout(db_session, user)
    subs.confirm_payment(db_session, payment.reference, _gateway(status=state))
    assert subs.is_active(db_session, user) is False


def test_repeated_confirmation_extends_only_once(db_session: Session) -> None:
    """Gateways retry callbacks and customers refresh the return page."""
    user = _user(db_session)
    payment = _checkout(db_session, user)
    client = _gateway()

    for _ in range(5):
        subs.confirm_payment(db_session, payment.reference, client)

    subscription = subs.get_subscription(db_session, user)
    assert subscription.expires_at - datetime.utcnow() <= timedelta(days=30)


def test_an_applied_payment_is_not_re_queried(db_session: Session) -> None:
    user = _user(db_session)
    payment = _checkout(db_session, user)
    subs.confirm_payment(db_session, payment.reference, _gateway())

    calls: list = []
    subs.confirm_payment(db_session, payment.reference, _gateway(calls=calls))
    assert calls == []


def test_the_atomic_claim_refuses_a_second_application(db_session: Session) -> None:
    user = _user(db_session)
    payment = _checkout(db_session, user)
    assert subs.apply_successful_payment(db_session, payment) is True
    db_session.refresh(payment)
    assert subs.apply_successful_payment(db_session, payment) is False


def test_renewing_early_stacks_onto_remaining_time(db_session: Session) -> None:
    user = _user(db_session)
    first = _checkout(db_session, user)
    subs.confirm_payment(db_session, first.reference, _gateway())
    second = _checkout(db_session, user)
    subs.confirm_payment(db_session, second.reference, _gateway())

    remaining = subs.get_subscription(db_session, user).expires_at - datetime.utcnow()
    assert remaining > timedelta(days=59), "early renewal must not forfeit the first month"


def test_renewing_after_expiry_starts_from_now(db_session: Session) -> None:
    user = _user(db_session)
    db_session.add(Subscription(user_id=user.id, started_at=datetime.utcnow() - timedelta(days=90),
                                expires_at=datetime.utcnow() - timedelta(days=50)))
    db_session.commit()

    payment = _checkout(db_session, user)
    subs.confirm_payment(db_session, payment.reference, _gateway())
    remaining = subs.get_subscription(db_session, user).expires_at - datetime.utcnow()
    assert timedelta(days=29) < remaining <= timedelta(days=30)


def test_an_expired_subscription_grants_no_access(db_session: Session) -> None:
    user = _user(db_session)
    db_session.add(Subscription(user_id=user.id, started_at=datetime.utcnow() - timedelta(days=40),
                                expires_at=datetime.utcnow() - timedelta(seconds=1)))
    db_session.commit()
    assert subs.is_active(db_session, user) is False


def test_staff_have_access_without_paying(db_session: Session) -> None:
    assert subs.is_active(db_session, _user(db_session, "admin@t.com", role="admin"))


def test_unknown_reference_is_rejected(db_session: Session) -> None:
    with pytest.raises(LookupError):
        subs.confirm_payment(db_session, "SUB-DOESNOTEXIST", _gateway())


# --- callbacks ---------------------------------------------------------------


def test_a_genuine_callback_activates_after_re_verifying(db_session: Session) -> None:
    user = _user(db_session)
    payment = _checkout(db_session, user)
    calls: list = []

    subs.handle_callback(db_session, _callback(payment.reference), _gateway(calls=calls))

    assert subs.is_active(db_session, user)
    assert any(path.endswith("/status") for path in calls), "the callback must be cross-checked"


def test_a_forged_callback_grants_nothing(db_session: Session) -> None:
    user = _user(db_session)
    payment = _checkout(db_session, user)

    with pytest.raises(OPayError):
        subs.handle_callback(db_session, _callback(payment.reference, secret="attacker-key"),
                             _gateway())
    assert subs.is_active(db_session, user) is False


def test_a_genuine_callback_cannot_override_the_gateways_own_answer(db_session: Session) -> None:
    """The callback says SUCCESS; asked directly, the gateway says FAIL. FAIL wins."""
    user = _user(db_session)
    payment = _checkout(db_session, user)
    subs.handle_callback(db_session, _callback(payment.reference), _gateway(status="FAIL"))
    assert subs.is_active(db_session, user) is False


def test_a_failed_checkout_leaves_an_errored_record(db_session: Session) -> None:
    def handler(request):
        return httpx.Response(200, json={"code": "00004", "message": "Invalid request parameters"})

    user = _user(db_session)
    with pytest.raises(OPayError):
        _checkout(db_session, user, OPayClient(CONFIG, transport=httpx.MockTransport(handler)))
    assert db_session.query(Payment).one().status == "error"
