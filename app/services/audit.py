from sqlalchemy.orm import Session

from app.models.entities import AdminLog, User


def log_admin_action(
    db: Session,
    admin: User,
    action: str,
    ip_address: str | None = None,
    target_user_id: int | None = None,
    metadata: dict | None = None,
) -> None:
    entry = AdminLog(
        admin_id=admin.id,
        action=action,
        target_user_id=target_user_id,
        metadata_json=metadata or {},
        ip_address=ip_address,
    )
    db.add(entry)
    db.commit()
