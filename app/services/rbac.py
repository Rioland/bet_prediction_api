from collections.abc import Callable

from fastapi import Depends, HTTPException

from app.api.deps import get_current_user
from app.models.entities import User, UserRole

ROLE_LEVEL: dict[UserRole, int] = {
    UserRole.USER: 1,
    UserRole.PREMIUM_USER: 2,
    UserRole.MODERATOR: 3,
    UserRole.ADMIN: 4,
    UserRole.SUPER_ADMIN: 5,
}


def require_role(minimum_role: UserRole) -> Callable:
    def _guard(current_user: User = Depends(get_current_user)) -> User:
        if ROLE_LEVEL[current_user.role] < ROLE_LEVEL[minimum_role]:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user

    return _guard

