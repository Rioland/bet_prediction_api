from datetime import timedelta

import pyotp
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import create_token, hash_password, verify_password
from app.models.entities import User
from app.schemas.common import LoginRequest, RegisterRequest, TokenPair


def register_user(db: Session, payload: RegisterRequest) -> TokenPair:
    email = str(payload.email).lower()
    existing = db.scalar(select(User).where(User.email == email))
    if existing:
        raise HTTPException(status_code=400, detail="Email already in use")
    user = User(
        name=payload.name,
        email=email,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return issue_tokens(user.id)


def login_user(db: Session, payload: LoginRequest) -> TokenPair:
    user = db.scalar(select(User).where(User.email == str(payload.email).lower()))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    # This endpoint issues the same tokens as the admin login, so it must apply
    # the same second factor - otherwise 2FA is bypassable by logging in here.
    verify_second_factor(user, payload.otp_code)
    return issue_tokens(user.id)


def verify_second_factor(user: User, otp_code: str | None) -> None:
    """Raise unless the account's second factor is satisfied (no-op when disabled)."""
    if not user.two_factor_enabled:
        return
    if not user.two_factor_secret:
        raise HTTPException(status_code=500, detail="2FA misconfigured for this account")
    if not otp_code:
        raise HTTPException(status_code=401, detail="OTP code required")
    if not pyotp.TOTP(user.two_factor_secret).verify(otp_code, valid_window=1):
        raise HTTPException(status_code=401, detail="Invalid OTP code")


def issue_tokens(user_id: int) -> TokenPair:
    subject = str(user_id)
    access_token = create_token(
        subject=subject,
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
        token_type="access",
    )
    refresh_token = create_token(
        subject=subject,
        expires_delta=timedelta(minutes=settings.refresh_token_expire_minutes),
        token_type="refresh",
    )
    return TokenPair(access_token=access_token, refresh_token=refresh_token)
