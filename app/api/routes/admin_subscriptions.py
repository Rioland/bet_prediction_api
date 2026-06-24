from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select

from app.api.deps import DbSession
from app.models.entities import Subscription, User, UserRole
from app.services.audit import log_admin_action
from app.services.rbac import require_role

router = APIRouter(prefix="/admin/subscriptions", tags=["admin-subscriptions"])


@router.get("")
def list_subscriptions(
    db: DbSession, _: Annotated[User, Depends(require_role(UserRole.ADMIN))]
) -> list[dict]:
    rows = list(db.scalars(select(Subscription).order_by(Subscription.created_at.desc())))
    return [
        {
            "id": r.id,
            "user_id": r.user_id,
            "provider": r.provider,
            "status": r.status,
            "expires_at": r.expires_at,
        }
        for r in rows
    ]


@router.post("/{subscription_id}/extend")
def extend_subscription(
    subscription_id: int,
    days: int,
    db: DbSession,
    request: Request,
    current_admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
) -> dict:
    row = db.get(Subscription, subscription_id)
    if not row:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if row.expires_at is None:
        raise HTTPException(status_code=400, detail="Subscription has no expiry")
    row.expires_at = row.expires_at + timedelta(days=days)
    db.commit()
    log_admin_action(
        db,
        current_admin,
        "extend_subscription",
        target_user_id=row.user_id,
        ip_address=request.client.host if request.client else None,
        metadata={"subscription_id": subscription_id, "days": days},
    )
    return {"status": "extended"}


@router.post("/{subscription_id}/cancel")
def cancel_subscription(
    subscription_id: int,
    db: DbSession,
    request: Request,
    current_admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
) -> dict:
    row = db.get(Subscription, subscription_id)
    if not row:
        raise HTTPException(status_code=404, detail="Subscription not found")
    row.status = "cancelled"
    db.commit()
    log_admin_action(
        db,
        current_admin,
        "cancel_subscription",
        target_user_id=row.user_id,
        ip_address=request.client.host if request.client else None,
        metadata={"subscription_id": subscription_id},
    )
    return {"status": "cancelled"}


@router.post("/{subscription_id}/refund")
def refund_subscription(
    subscription_id: int,
    provider: str,
    db: DbSession,
    request: Request,
    current_admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
) -> dict:
    row = db.get(Subscription, subscription_id)
    if not row:
        raise HTTPException(status_code=404, detail="Subscription not found")
    # TODO: wire Stripe/Paystack/Flutterwave APIs.
    log_admin_action(
        db,
        current_admin,
        "refund_subscription",
        target_user_id=row.user_id,
        ip_address=request.client.host if request.client else None,
        metadata={"subscription_id": subscription_id, "provider": provider},
    )
    return {"status": "refund_requested", "provider": provider}
