from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select

from app.api.deps import DbSession
from app.models.entities import Notification, SubscriptionType, User, UserRole
from app.schemas.admin import NotificationSendRequest
from app.services.audit import log_admin_action
from app.services.rbac import require_role

router = APIRouter(prefix="/admin/notifications", tags=["admin-notifications"])


@router.post("/send")
def send_notifications(
    payload: NotificationSendRequest,
    db: DbSession,
    request: Request,
    current_admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
) -> dict:
    stmt = select(User.id)
    if payload.audience == "premium":
        stmt = stmt.where(User.subscription_type == SubscriptionType.PREMIUM)
    elif payload.audience == "selected" and payload.user_ids:
        stmt = stmt.where(User.id.in_(payload.user_ids))
    recipients = list(db.scalars(stmt))
    for user_id in recipients:
        db.add(Notification(user_id=user_id, title=payload.title, body=payload.body))
    db.commit()
    log_admin_action(
        db,
        current_admin,
        "send_notification",
        ip_address=request.client.host if request.client else None,
        metadata={"audience": payload.audience, "count": len(recipients), "channel": payload.channel},
    )
    return {"status": "queued", "recipients": len(recipients)}
