from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select

from app.api.deps import DbSession, require_admin
from app.models.entities import Notification, User
from app.workers.tasks import retrain_models_task, sync_live_matches_task, sync_upcoming_fixtures_task

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/sync/live")
def trigger_live_sync(_: Annotated[object, Depends(require_admin)]) -> dict:
    task = sync_live_matches_task.delay()
    return {"task_id": task.id, "status": "queued"}


@router.post("/sync/fixtures")
def trigger_fixture_sync(_: Annotated[object, Depends(require_admin)]) -> dict:
    task = sync_upcoming_fixtures_task.delay()
    return {"task_id": task.id, "status": "queued"}


@router.post("/ml/retrain")
def retrain(_: Annotated[object, Depends(require_admin)]) -> dict:
    task = retrain_models_task.delay()
    return {"task_id": task.id, "status": "queued"}


@router.get("/users")
def list_users(db: DbSession, _: Annotated[object, Depends(require_admin)]) -> list[dict]:
    users = db.scalars(select(User).order_by(User.created_at.desc()).limit(200))
    return [
        {
            "id": u.id,
            "name": u.name,
            "email": u.email,
            "subscription_type": u.subscription_type.value,
            "created_at": u.created_at.isoformat(),
        }
        for u in users
    ]


@router.get("/analytics")
def analytics(db: DbSession, _: Annotated[object, Depends(require_admin)]) -> dict:
    total_users = db.scalar(select(func.count()).select_from(User)) or 0
    sent_notifications = db.scalar(
        select(func.count()).select_from(Notification).where(Notification.sent.is_(True))
    ) or 0
    return {"total_users": total_users, "sent_notifications": sent_notifications}


@router.post("/notifications/broadcast")
def broadcast_notification(
    db: DbSession, _: Annotated[object, Depends(require_admin)], title: str, body: str
) -> dict:
    users = list(db.scalars(select(User.id)))
    for user_id in users:
        db.add(Notification(user_id=user_id, title=title, body=body))
    db.commit()
    return {"status": "queued", "users": len(users)}
