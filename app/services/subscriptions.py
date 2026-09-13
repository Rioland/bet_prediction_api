"""Paid subscriptions, backed by verified gateway payments.

Access is granted in exactly one place, apply_successful_payment, and only
after the gateway has confirmed the payment directly. Three things are checked
before a subscription is extended:

1. The gateway says SUCCESS when asked itself - not merely a callback, and never
   the browser redirect, which a customer can simply type into the address bar.
2. The amount and currency match what this payment was created for. A valid
   signature proves a callback is genuine; it does not prove the customer paid
   the right amount.
3. The payment has not already been applied. Gateways retry callbacks, and
   customers refresh the return page, so the same success can arrive many
   times.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.models import AppSetting, Payment, Subscription, User
from app.services.opay import (
    CURRENCY,
    OPayClient,
    OPayError,
    naira_to_kobo,
    new_reference,
    verify_callback,
)

PRICE_KEY = "subscription_price_naira"
DEFAULT_PRICE_NAIRA = Decimal("5000")
PERIOD_DAYS = 30

# Guard rails on an admin-entered price, so a slip of the keyboard cannot put
# the subscription at five naira or five million.
MIN_PRICE_NAIRA = Decimal("100")
MAX_PRICE_NAIRA = Decimal("1000000")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── price ────────────────────────────────────────────────────────────────────


def get_price_naira(db: Session) -> Decimal:
    setting = db.get(AppSetting, PRICE_KEY)
    if setting is None:
        return DEFAULT_PRICE_NAIRA
    try:
        return Decimal(setting.value)
    except InvalidOperation:
        return DEFAULT_PRICE_NAIRA


def set_price_naira(db: Session, amount: str | int | float, updated_by: str) -> Decimal:
    """Change the price for future checkouts.

    Payments already started keep the price they were created with; see
    Payment.amount_kobo.
    """
    try:
        price = Decimal(str(amount)).quantize(Decimal("0.01"))
    except InvalidOperation as exc:
        raise ValueError("Price must be a number.") from exc

    if price < MIN_PRICE_NAIRA or price > MAX_PRICE_NAIRA:
        raise ValueError(
            f"Price must be between ₦{MIN_PRICE_NAIRA:,} and ₦{MAX_PRICE_NAIRA:,}."
        )

    setting = db.get(AppSetting, PRICE_KEY)
    if setting is None:
        setting = AppSetting(key=PRICE_KEY, value=str(price))
        db.add(setting)
    setting.value = str(price)
    setting.updated_by = updated_by
    setting.updated_at = _now()
    db.commit()
    return price


# ── access ───────────────────────────────────────────────────────────────────


def get_subscription(db: Session, user: User) -> Subscription | None:
    return db.query(Subscription).filter(Subscription.user_id == user.id).first()


def is_active(db: Session, user: User | None) -> bool:
    if user is None:
        return False
    # Staff can see everything without paying.
    if user.role in {"admin", "super_admin", "moderator"}:
        return True
    subscription = get_subscription(db, user)
    return bool(subscription and subscription.expires_at > _now())


def describe(db: Session, user: User) -> dict[str, Any]:
    subscription = get_subscription(db, user)
    active = is_active(db, user)
    days_left = None
    if subscription and subscription.expires_at > _now():
        days_left = max(0, (subscription.expires_at - _now()).days)
    return {
        "active": active,
        "expires_at": subscription.expires_at.isoformat() if subscription else None,
        "days_left": days_left,
        "price_naira": float(get_price_naira(db)),
        "period_days": PERIOD_DAYS,
        "staff": user.role in {"admin", "super_admin", "moderator"},
    }


# ── checkout ─────────────────────────────────────────────────────────────────


def start_checkout(
    db: Session,
    user: User,
    client: OPayClient,
    *,
    return_url: str,
    callback_url: str,
    cancel_url: str,
) -> Payment:
    """Create a payment at today's price and open a hosted checkout for it."""
    amount_kobo = naira_to_kobo(get_price_naira(db))
    payment = Payment(
        user_id=user.id,
        reference=new_reference(),
        amount_kobo=amount_kobo,
        currency=CURRENCY,
        status="initial",
    )
    db.add(payment)
    db.commit()
    db.refresh(payment)

    try:
        data = client.create_checkout(
            reference=payment.reference,
            amount_kobo=amount_kobo,
            return_url=return_url,
            callback_url=callback_url,
            cancel_url=cancel_url,
            user_id=user.id,
            user_email=user.email,
            user_name=user.name,
            product_name="Monthly subscription",
            product_description=f"{PERIOD_DAYS} days of booking codes and premium picks",
        )
    except OPayError as exc:
        payment.status = "error"
        payment.failure_reason = str(exc)[:500]
        db.commit()
        raise

    payment.provider_order_no = data.get("orderNo")
    payment.checkout_url = data["cashierUrl"]
    payment.status = "pending"
    db.commit()
    db.refresh(payment)
    return payment


# ── confirmation ─────────────────────────────────────────────────────────────


def apply_successful_payment(db: Session, payment: Payment) -> bool:
    """Extend the subscription for a verified payment, at most once.

    The claim on the payment is an atomic conditional update. Two concurrent
    callbacks can both read applied_at as empty; only one of them can change
    it, so only one of them extends the subscription.
    """
    now = _now()
    claimed = db.execute(
        update(Payment)
        .where(Payment.id == payment.id, Payment.applied_at.is_(None))
        .values(applied_at=now, status="success", paid_at=now)
    ).rowcount
    if claimed != 1:
        db.rollback()
        return False

    subscription = db.query(Subscription).filter(Subscription.user_id == payment.user_id).first()
    if subscription is None:
        subscription = Subscription(
            user_id=payment.user_id, started_at=now, expires_at=now + timedelta(days=PERIOD_DAYS)
        )
        db.add(subscription)
    else:
        # Renewing early stacks onto the remaining time rather than forfeiting it.
        base = max(subscription.expires_at, now)
        subscription.expires_at = base + timedelta(days=PERIOD_DAYS)
        subscription.cancelled_at = None
    subscription.last_payment_id = payment.id

    user = db.get(User, payment.user_id)
    if user is not None:
        user.subscription_type = "premium"

    db.commit()
    return True


def confirm_payment(db: Session, reference: str, client: OPayClient) -> Payment:
    """Ask the gateway for the truth about a payment, and act on it.

    Safe to call any number of times for the same reference.
    """
    payment = db.query(Payment).filter(Payment.reference == reference).first()
    if payment is None:
        raise LookupError("Unknown payment reference.")
    if payment.applied_at is not None:
        return payment

    status = client.query_status(reference)
    state = str(status.get("status", "")).upper()
    amount = status.get("amount") or {}

    if state == "SUCCESS":
        paid_kobo = int(amount.get("total", -1))
        paid_currency = str(amount.get("currency", ""))
        if paid_kobo != payment.amount_kobo or paid_currency != payment.currency:
            payment.status = "amount_mismatch"
            payment.failure_reason = (
                f"Expected {payment.amount_kobo} {payment.currency}, "
                f"gateway reported {paid_kobo} {paid_currency}."
            )
            db.commit()
            return payment
        apply_successful_payment(db, payment)
        db.refresh(payment)
        return payment

    if state in {"FAIL", "CLOSE"}:
        payment.status = state.lower()
        payment.failure_reason = status.get("failureReason") or payment.failure_reason
        db.commit()
    return payment


def handle_callback(db: Session, body: dict[str, Any], client: OPayClient) -> Payment:
    """Verify a gateway callback, then confirm the payment independently.

    The signature proves the callback is from the gateway. The status query
    that follows is what actually decides access, so a callback with a genuine
    signature but a stale or unexpected state cannot grant anything by itself.
    """
    payload = verify_callback(body, client.config.secret_key)
    return confirm_payment(db, str(payload["reference"]), client)
