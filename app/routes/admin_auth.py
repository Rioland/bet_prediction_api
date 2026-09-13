"""
Admin auth routes: login, refresh, logout, me.

Auth model: the access token is a bearer token. The frontend stores it and
sends `Authorization: Bearer <token>` on every request — this is the sole
mechanism used for authorizing state-changing admin requests, and it is
inherently CSRF-safe (a malicious site cannot set custom headers on a
cross-origin request). We also set httpOnly cookies as a convenience for the
refresh flow only; cookies are NEVER accepted for authorizing admin actions.
"""

from datetime import datetime

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Response, status
from app.auth import JWTError
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from app.database import get_db
from app.models import User

router = APIRouter(prefix="/admin/auth", tags=["admin-auth"])

ADMIN_ROLES = {"admin", "super_admin", "moderator"}


class LoginRequest(BaseModel):
    email: str
    password: str
    otp_code: str | None = None


def _user_dict(user: User) -> dict:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "status": user.status,
        "subscription_type": user.subscription_type,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
        "two_factor_enabled": user.two_factor_enabled,
    }


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    # Only the refresh token lives in a cookie, and it is never accepted for
    # authorizing admin actions (see module docstring) — only for minting a
    # new access token via /refresh. httpOnly + samesite=lax limits exposure.
    response.set_cookie("admin_refresh_token", refresh_token, httponly=True, samesite="lax", path="/")


@router.post("/login")
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower().strip()).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if user.role not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Admin access required")
    if user.status != "active":
        raise HTTPException(status_code=403, detail="Account is not active")

    user.last_login_at = datetime.utcnow()
    db.commit()

    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)
    _set_refresh_cookie(response, refresh_token)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": _user_dict(user),
    }


@router.post("/refresh")
def refresh(
    response: Response,
    db: Session = Depends(get_db),
    admin_refresh_token: str | None = Cookie(default=None),
):
    token = admin_refresh_token
    if not token:
        raise HTTPException(status_code=401, detail="Missing refresh token")
    try:
        data = decode_token(token)
        if data.get("type") != "refresh":
            raise JWTError("wrong type")
        user_id = int(data["sub"])
    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    access_token = create_access_token(user.id)
    refresh_token_new = create_refresh_token(user.id)
    _set_refresh_cookie(response, refresh_token_new)
    return {"access_token": access_token, "refresh_token": refresh_token_new}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("admin_refresh_token", path="/")
    return {"status": "logged_out"}


# ── Shared auth dependency ───────────────────────────────────────────────────

def get_current_admin(
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
) -> User:
    """
    Authorize admin requests using the Bearer access token ONLY. Cookies are
    never accepted here — that would reopen the CSRF hole this design avoids.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ", 1)[1].strip()
    try:
        data = decode_token(token)
        if data.get("type") != "access":
            raise JWTError("wrong type")
        user_id = int(data["sub"])
    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.get(User, user_id)
    if not user or user.role not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# Moderators can pass get_current_admin, which is right for moderation work
# but not for anything that moves money or decides what subscribers are sold.
FULL_ADMIN_ROLES = {"admin", "super_admin"}


def get_full_admin(current: User = Depends(get_current_admin)) -> User:
    """An admin or super admin - never a moderator."""
    if current.role not in FULL_ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current


@router.get("/me")
def me(current: User = Depends(get_current_admin)):
    return _user_dict(current)
