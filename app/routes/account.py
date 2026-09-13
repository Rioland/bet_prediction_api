"""Subscriber accounts: registration, login, subscription and payment.

Separate from admin authentication. Customers and staff share the users table
and the same token format, but staff sign in through /admin/auth, and nothing
here can grant a role.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from jwt import PyJWTError
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.orm import Session

from app.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.config import (
    OPAY_MERCHANT_ID,
    OPAY_PRODUCTION,
    OPAY_PUBLIC_KEY,
    OPAY_SECRET_KEY,
    PUBLIC_API_URL,
    PUBLIC_SITE_URL,
)
from app.database import get_db
from app.models import Payment, User
from app.rate_limit import limiter
from app.services import subscriptions as subs
from app.services.opay import OPayClient, OPayConfig, OPayError, OPayNotConfigured

logger = logging.getLogger(__name__)

router = APIRouter(tags=["account"])
DbSession = Annotated[Session, Depends(get_db)]

MIN_PASSWORD_LENGTH = 8


def opay_client() -> OPayClient:
    return OPayClient(OPayConfig(OPAY_MERCHANT_ID, OPAY_PUBLIC_KEY, OPAY_SECRET_KEY, OPAY_PRODUCTION))


OPay = Annotated[OPayClient, Depends(opay_client)]


# ── auth ─────────────────────────────────────────────────────────────────────


class RegisterInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=128)

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name cannot be blank.")
        return value


class LoginInput(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


def _session(user: User) -> dict[str, Any]:
    return {
        "access_token": create_access_token(user.id),
        "refresh_token": create_refresh_token(user.id),
        "token_type": "bearer",
        "user": {"id": user.id, "name": user.name, "email": user.email, "role": user.role},
    }


def current_user(db: DbSession, authorization: str | None = Header(default=None)) -> User:
    """Bearer access token only; a refresh token is rejected."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not signed in.")
    try:
        data = decode_token(authorization.split(" ", 1)[1].strip())
        if data.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid session.")
        user = db.get(User, int(data["sub"]))
    except (PyJWTError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=401, detail="Invalid session.") from exc

    if user is None:
        raise HTTPException(status_code=401, detail="Invalid session.")
    if user.status in {"suspended", "banned"}:
        raise HTTPException(status_code=403, detail="This account is not active.")
    return user


CurrentUser = Annotated[User, Depends(current_user)]


@router.post("/auth/register", status_code=201)
@limiter.limit("5/minute")
def register(request: Request, body: RegisterInput, db: DbSession) -> dict[str, Any]:
    email = body.email.lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    # Always a plain customer. Role and subscription are never taken from input.
    user = User(name=body.name, email=email, password_hash=hash_password(body.password),
                role="user", status="active", subscription_type="free")
    db.add(user)
    db.commit()
    db.refresh(user)
    return _session(user)


@router.post("/auth/login")
@limiter.limit("10/minute")
def login(request: Request, body: LoginInput, db: DbSession) -> dict[str, Any]:
    user = db.query(User).filter(User.email == body.email.lower()).first()
    # One message for both cases, so the endpoint cannot be used to discover
    # which emails have accounts.
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    if user.status in {"suspended", "banned"}:
        raise HTTPException(status_code=403, detail="This account is not active.")
    return _session(user)


@router.get("/auth/me")
def me(user: CurrentUser, db: DbSession) -> dict[str, Any]:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "subscription": subs.describe(db, user),
    }


# ── subscription ─────────────────────────────────────────────────────────────


@router.get("/subscription/plan")
def plan(db: DbSession) -> dict[str, Any]:
    """Public: what a subscription costs and how long it lasts."""
    return {
        "price_naira": float(subs.get_price_naira(db)),
        "currency": "NGN",
        "period_days": subs.PERIOD_DAYS,
        "payments_enabled": opay_client().config.configured,
    }


@router.post("/subscription/checkout")
@limiter.limit("10/minute")
def checkout(request: Request, user: CurrentUser, db: DbSession, client: OPay) -> dict[str, Any]:
    """Open an OPay checkout for one subscription period."""
    try:
        payment = subs.start_checkout(
            db, user, client,
            return_url=f"{PUBLIC_SITE_URL}/account?payment=return",
            callback_url=f"{PUBLIC_API_URL}/payments/opay/callback",
            cancel_url=f"{PUBLIC_SITE_URL}/account?payment=cancelled",
        )
    except OPayNotConfigured as exc:
        raise HTTPException(status_code=503, detail="Payments are not set up yet.") from exc
    except OPayError as exc:
        logger.warning("OPay checkout failed for user %s: %s", user.id, exc)
        raise HTTPException(status_code=502, detail="Could not start the payment. Try again.") from exc

    return {
        "reference": payment.reference,
        "checkout_url": payment.checkout_url,
        "amount_naira": payment.amount_kobo / 100,
    }


@router.post("/subscription/verify/{reference}")
@limiter.limit("20/minute")
def verify(
    request: Request, reference: str, user: CurrentUser, db: DbSession, client: OPay
) -> dict[str, Any]:
    """Check a payment after returning from checkout.

    The return redirect is not evidence of payment - anyone can load that URL.
    This asks OPay directly, and only the customer who made the payment may ask.
    """
    payment = db.query(Payment).filter(Payment.reference == reference).first()
    if payment is None or payment.user_id != user.id:
        raise HTTPException(status_code=404, detail="Payment not found.")

    try:
        payment = subs.confirm_payment(db, reference, client)
    except OPayNotConfigured as exc:
        raise HTTPException(status_code=503, detail="Payments are not set up yet.") from exc
    except OPayError as exc:
        raise HTTPException(status_code=502, detail="Could not reach OPay. Try again shortly.") from exc

    return {"status": payment.status, "subscription": subs.describe(db, user)}


@router.get("/subscription/payments")
def my_payments(user: CurrentUser, db: DbSession) -> list[dict[str, Any]]:
    rows = (
        db.query(Payment).filter(Payment.user_id == user.id)
        .order_by(Payment.created_at.desc()).limit(20).all()
    )
    return [
        {"reference": p.reference, "amount_naira": p.amount_kobo / 100, "status": p.status,
         "created_at": p.created_at.isoformat(),
         "paid_at": p.paid_at.isoformat() if p.paid_at else None}
        for p in rows
    ]


@router.post("/payments/opay/callback")
async def opay_callback(request: Request, db: DbSession, client: OPay) -> dict[str, str]:
    """OPay's server-to-server notification.

    Public by necessity. Access is only granted after the signature verifies
    and a fresh status query agrees. OPay retries anything that is not
    acknowledged within five seconds, so a rejected or failed callback still
    answers quickly; confirm_payment is idempotent, so retries are harmless.
    """
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid body.") from None

    try:
        payment = subs.handle_callback(db, body, client)
    except OPayNotConfigured:
        logger.error("OPay callback received but payments are not configured.")
        raise HTTPException(status_code=503, detail="Not configured.") from None
    except LookupError:
        logger.warning("OPay callback for an unknown reference.")
        raise HTTPException(status_code=404, detail="Unknown reference.") from None
    except OPayError as exc:
        # Includes signature failures. Logged without the payload.
        logger.warning("Rejected OPay callback: %s", exc)
        raise HTTPException(status_code=400, detail="Rejected.") from None

    return {"status": payment.status}
