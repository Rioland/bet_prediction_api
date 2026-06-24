import secrets

import pyotp
from fastapi import APIRouter, HTTPException, Request, Response, status
from jose import JWTError, jwt
from sqlalchemy import select

from typing import Annotated

from fastapi import Depends

from app.api.deps import DbSession, get_current_user
from app.core.config import settings
from app.core.security import verify_password
from app.models.entities import LoginHistory, User, UserRole
from app.schemas.admin import (
    AdminLoginRequest,
    AdminSessionResponse,
    AdminUserOut,
    TwoFASetupResponse,
    TwoFAVerifyRequest,
)
from app.schemas.common import RefreshRequest, TokenPair
from app.services.auth_service import issue_tokens
from app.services.audit import log_admin_action

router = APIRouter(prefix="/admin/auth", tags=["admin-auth"])


def _set_admin_cookies(response: Response, access_token: str, refresh_token: str, csrf_token: str) -> None:
    response.set_cookie(
        "admin_access_token",
        access_token,
        httponly=True,
        secure=settings.admin_cookie_secure,
        samesite="lax",
        max_age=settings.access_token_expire_minutes * 60,
        path="/",
    )
    response.set_cookie(
        "admin_refresh_token",
        refresh_token,
        httponly=True,
        secure=settings.admin_cookie_secure,
        samesite="lax",
        max_age=settings.refresh_token_expire_minutes * 60,
        path="/",
    )
    response.set_cookie(
        "admin_csrf_token",
        csrf_token,
        httponly=False,
        secure=settings.admin_cookie_secure,
        samesite="lax",
        max_age=settings.refresh_token_expire_minutes * 60,
        path="/",
    )


@router.post("/login", response_model=AdminSessionResponse)
def admin_login(payload: AdminLoginRequest, db: DbSession, request: Request, response: Response) -> AdminSessionResponse:
    user = db.scalar(select(User).where(User.email == str(payload.email).lower()))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if user.role not in {UserRole.ADMIN, UserRole.SUPER_ADMIN}:
        raise HTTPException(status_code=403, detail="Admin role required")
    if user.two_factor_enabled:
        if not user.two_factor_secret:
            raise HTTPException(status_code=500, detail="2FA misconfigured for admin account")
        if not payload.otp_code:
            raise HTTPException(status_code=401, detail="OTP code required")
        totp = pyotp.TOTP(user.two_factor_secret)
        if not totp.verify(payload.otp_code, valid_window=1):
            raise HTTPException(status_code=401, detail="Invalid OTP code")
    db.add(
        LoginHistory(
            user_id=user.id,
            ip_address=request.client.host if request.client else None,
            device=request.headers.get("user-agent"),
            location=None,
        )
    )
    db.commit()
    log_admin_action(
        db,
        admin=user,
        action="admin_login",
        ip_address=request.client.host if request.client else None,
    )
    tokens = issue_tokens(user.id)
    csrf_token = secrets.token_urlsafe(32)
    _set_admin_cookies(response, tokens.access_token, tokens.refresh_token, csrf_token)
    return AdminSessionResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        csrf_token=csrf_token,
        user=AdminUserOut.model_validate(user),
    )


@router.post("/refresh", response_model=TokenPair)
def admin_refresh(
    request: Request, response: Response, payload: RefreshRequest | None = None
) -> TokenPair:
    refresh_token = payload.refresh_token if payload else None
    if not refresh_token:
        refresh_token = request.cookies.get("admin_refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token")
    try:
        data = jwt.decode(
            refresh_token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        if data.get("type") != "refresh":
            raise JWTError("invalid token type")
        user_id = int(data["sub"])
    except (JWTError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid refresh token") from exc
    tokens = issue_tokens(user_id)
    csrf_token = secrets.token_urlsafe(32)
    _set_admin_cookies(response, tokens.access_token, tokens.refresh_token, csrf_token)
    return tokens


@router.get("/me", response_model=AdminUserOut)
def admin_me(db: DbSession, request: Request) -> AdminUserOut:
    token = request.cookies.get("admin_access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id = int(payload["sub"])
    except (JWTError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid session") from exc
    user = db.get(User, user_id)
    if not user or user.role not in {UserRole.ADMIN, UserRole.SUPER_ADMIN}:
        raise HTTPException(status_code=403, detail="Admin access required")
    return AdminUserOut.model_validate(user)


@router.post("/logout")
def admin_logout(response: Response) -> dict:
    response.delete_cookie("admin_access_token", path="/")
    response.delete_cookie("admin_refresh_token", path="/")
    response.delete_cookie("admin_csrf_token", path="/")
    return {"status": "logged_out"}


@router.post("/2fa/setup", response_model=TwoFASetupResponse)
def setup_admin_2fa(current_user: Annotated[User, Depends(get_current_user)], db: DbSession) -> TwoFASetupResponse:
    if current_user.role not in {UserRole.ADMIN, UserRole.SUPER_ADMIN}:
        raise HTTPException(status_code=403, detail="Admin access required")
    secret = pyotp.random_base32()
    current_user.two_factor_secret = secret
    current_user.two_factor_enabled = False
    db.commit()
    otp_url = pyotp.totp.TOTP(secret).provisioning_uri(
        name=current_user.email, issuer_name=settings.app_name
    )
    return TwoFASetupResponse(secret=secret, otpauth_url=otp_url)


@router.post("/2fa/verify")
def verify_admin_2fa(
    payload: TwoFAVerifyRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: DbSession,
) -> dict:
    if current_user.role not in {UserRole.ADMIN, UserRole.SUPER_ADMIN}:
        raise HTTPException(status_code=403, detail="Admin access required")
    if not current_user.two_factor_secret:
        raise HTTPException(status_code=400, detail="2FA is not configured")
    totp = pyotp.TOTP(current_user.two_factor_secret)
    if not totp.verify(payload.otp_code, valid_window=1):
        raise HTTPException(status_code=401, detail="Invalid OTP code")
    current_user.two_factor_enabled = True
    db.commit()
    return {"status": "2fa_enabled"}
