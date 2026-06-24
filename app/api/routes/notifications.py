from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import DbSession, get_current_user
from app.models.entities import DeviceToken, User
from app.schemas.common import DeviceRegisterRequest

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.post("/register-device")
def register_device(
    payload: DeviceRegisterRequest,
    db: DbSession,
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    token = DeviceToken(user_id=current_user.id, token=payload.token, platform=payload.platform)
    db.add(token)
    db.commit()
    return {"status": "ok"}
