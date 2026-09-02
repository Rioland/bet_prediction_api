from fastapi import APIRouter, HTTPException, Request, status
from jose import JWTError, jwt

from app.api.deps import DbSession
from app.core.config import settings
from app.core.rate_limit import limiter
from app.schemas.common import LoginRequest, RefreshRequest, RegisterRequest, TokenPair
from app.services.auth_service import issue_tokens, login_user, register_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenPair)
@limiter.limit("5/minute")
def register(request: Request, payload: RegisterRequest, db: DbSession) -> TokenPair:
    return register_user(db, payload)


@router.post("/login", response_model=TokenPair)
@limiter.limit("5/minute")
def login(request: Request, payload: LoginRequest, db: DbSession) -> TokenPair:
    return login_user(db, payload)


@router.post("/refresh", response_model=TokenPair)
@limiter.limit("20/minute")
def refresh(request: Request, payload: RefreshRequest) -> TokenPair:
    try:
        data = jwt.decode(
            payload.refresh_token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        if data.get("type") != "refresh":
            raise JWTError("invalid token type")
        user_id = int(data["sub"])
    except (JWTError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token") from exc
    return issue_tokens(user_id)
