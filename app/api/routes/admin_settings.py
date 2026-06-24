from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select

from app.api.deps import DbSession
from app.models.entities import SystemSetting, User, UserRole
from app.schemas.admin import SettingPatchRequest
from app.services.audit import log_admin_action
from app.services.crypto import decrypt_secret, encrypt_secret
from app.services.rbac import require_role

router = APIRouter(prefix="/admin/settings", tags=["admin-settings"])


@router.get("")
def get_settings(
    db: DbSession, _: Annotated[User, Depends(require_role(UserRole.SUPER_ADMIN))]
) -> list[dict]:
    rows = list(db.scalars(select(SystemSetting).order_by(SystemSetting.key.asc())))
    return [{"key": r.key, "value": decrypt_secret(r.encrypted_value)} for r in rows]


@router.patch("")
def update_setting(
    payload: SettingPatchRequest,
    db: DbSession,
    request: Request,
    current_admin: Annotated[User, Depends(require_role(UserRole.SUPER_ADMIN))],
) -> dict:
    row = db.scalar(select(SystemSetting).where(SystemSetting.key == payload.key))
    encrypted = encrypt_secret(payload.value)
    if not row:
        row = SystemSetting(
            key=payload.key,
            encrypted_value=encrypted,
            updated_by=current_admin.id,
            updated_at=datetime.utcnow(),
        )
        db.add(row)
    else:
        row.encrypted_value = encrypted
        row.updated_by = current_admin.id
        row.updated_at = datetime.utcnow()
    db.commit()
    log_admin_action(
        db,
        current_admin,
        "update_system_setting",
        ip_address=request.client.host if request.client else None,
        metadata={"key": payload.key},
    )
    return {"status": "updated"}
