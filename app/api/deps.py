from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models.entities import User, UserRole, UserStatus

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(
    request: Request, db: DbSession, token: Annotated[str | None, Depends(oauth2_scheme)]
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    auth_token = token or request.cookies.get("admin_access_token")
    if not auth_token:
        raise credentials_exception
    try:
        payload = jwt.decode(auth_token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id = int(payload.get("sub", "0"))
    except (JWTError, ValueError) as exc:
        raise credentials_exception from exc
    user = db.get(User, user_id)
    if not user:
        raise credentials_exception
    if user.status in {UserStatus.SUSPENDED, UserStatus.BANNED}:
        raise HTTPException(status_code=403, detail="Account restricted")
    return user


def require_admin(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    if current_user.role not in {UserRole.ADMIN, UserRole.SUPER_ADMIN}:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user
