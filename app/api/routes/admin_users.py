import csv
import io
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import Select, or_, select

from app.api.deps import DbSession, get_current_user
from app.models.entities import (
    LoginHistory,
    Prediction,
    Referral,
    Subscription,
    SubscriptionType,
    Transaction,
    User,
    UserRole,
    UserStatus,
    Wallet,
)
from app.schemas.admin import AdminUserOut, AdminUserPatch, UserActionRequest
from app.services.audit import log_admin_action
from app.services.rbac import require_role

router = APIRouter(prefix="/admin/users", tags=["admin-users"])


@router.get("")
def list_users(
    db: DbSession,
    current_admin: Annotated[User, Depends(require_role(UserRole.MODERATOR))],
    q: str | None = None,
    role: UserRole | None = None,
    subscription: SubscriptionType | None = None,
    status: UserStatus | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict:
    stmt: Select = select(User)
    if q:
        stmt = stmt.where(or_(User.name.ilike(f"%{q}%"), User.email.ilike(f"%{q}%")))
    if role:
        stmt = stmt.where(User.role == role)
    if subscription:
        stmt = stmt.where(User.subscription_type == subscription)
    if status:
        stmt = stmt.where(User.status == status)
    users = list(db.scalars(stmt.offset((page - 1) * page_size).limit(page_size)))
    log_admin_action(db, current_admin, "list_users")
    return {"items": [AdminUserOut.model_validate(u).model_dump() for u in users], "page": page}


@router.get("/export")
def export_users_csv(
    db: DbSession, _: Annotated[User, Depends(require_role(UserRole.ADMIN))]
) -> StreamingResponse:
    users = list(db.scalars(select(User).order_by(User.id.asc())))
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["id", "name", "email", "role", "status", "subscription_type", "created_at"])
    for u in users:
        writer.writerow([u.id, u.name, u.email, u.role.value, u.status.value, u.subscription_type.value, u.created_at.isoformat()])
    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=users.csv"},
    )


@router.get("/{user_id}")
def get_user_details(
    user_id: int,
    db: DbSession,
    _: Annotated[User, Depends(require_role(UserRole.MODERATOR))],
) -> dict:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    wallet = db.scalar(select(Wallet).where(Wallet.user_id == user_id))
    subs = list(db.scalars(select(Subscription).where(Subscription.user_id == user_id)))
    login_history = list(db.scalars(select(LoginHistory).where(LoginHistory.user_id == user_id).limit(20)))
    referrals = list(
        db.scalars(select(Referral).where(Referral.referrer_user_id == user_id).limit(20))
    )
    prediction_history = list(
        db.scalars(select(Prediction).order_by(Prediction.created_at.desc()).limit(20))
    )
    wallet_history = []
    if wallet:
        wallet_history = list(
            db.scalars(
                select(Transaction).where(Transaction.wallet_id == wallet.id).order_by(Transaction.id.desc())
            )
        )
    return {
        "user": AdminUserOut.model_validate(user).model_dump(),
        "prediction_history": [p.id for p in prediction_history],
        "wallet_history": [t.id for t in wallet_history],
        "subscriptions": [{"id": s.id, "status": s.status, "expires_at": s.expires_at} for s in subs],
        "login_history": [{"id": l.id, "ip_address": l.ip_address, "created_at": l.created_at} for l in login_history],
        "referral_history": [{"id": r.id, "referred_user_id": r.referred_user_id} for r in referrals],
    }


@router.patch("/{user_id}", response_model=AdminUserOut)
def update_user(
    user_id: int,
    payload: AdminUserPatch,
    db: DbSession,
    request: Request,
    current_admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
) -> User:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(user, key, value)
    db.commit()
    db.refresh(user)
    log_admin_action(
        db,
        current_admin,
        "update_user",
        target_user_id=user_id,
        ip_address=request.client.host if request.client else None,
        metadata={"changes": payload.model_dump(exclude_none=True)},
    )
    return user


@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    db: DbSession,
    request: Request,
    current_admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
) -> dict:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    log_admin_action(
        db,
        current_admin,
        "delete_user",
        target_user_id=user_id,
        ip_address=request.client.host if request.client else None,
    )
    return {"status": "deleted"}


def _update_user_status(
    db: DbSession, user_id: int, status_value: UserStatus, current_admin: User, request: Request, action: str
) -> dict:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.status = status_value
    db.commit()
    log_admin_action(
        db,
        current_admin,
        action,
        target_user_id=user_id,
        ip_address=request.client.host if request.client else None,
    )
    return {"status": user.status.value}


@router.post("/{user_id}/suspend")
def suspend_user(
    user_id: int,
    _: UserActionRequest,
    db: DbSession,
    request: Request,
    current_admin: Annotated[User, Depends(require_role(UserRole.MODERATOR))],
) -> dict:
    return _update_user_status(db, user_id, UserStatus.SUSPENDED, current_admin, request, "suspend_user")


@router.post("/{user_id}/ban")
def ban_user(
    user_id: int,
    _: UserActionRequest,
    db: DbSession,
    request: Request,
    current_admin: Annotated[User, Depends(require_role(UserRole.MODERATOR))],
) -> dict:
    return _update_user_status(db, user_id, UserStatus.BANNED, current_admin, request, "ban_user")


@router.post("/{user_id}/grant-premium")
def grant_premium(
    user_id: int,
    db: DbSession,
    request: Request,
    current_admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
) -> dict:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.subscription_type = SubscriptionType.PREMIUM
    user.role = UserRole.PREMIUM_USER if user.role == UserRole.USER else user.role
    db.add(
        Subscription(
            user_id=user_id,
            provider="admin",
            provider_reference=f"grant-{user_id}-{int(datetime.utcnow().timestamp())}",
            status="active",
            expires_at=datetime.utcnow() + timedelta(days=30),
        )
    )
    db.commit()
    log_admin_action(
        db,
        current_admin,
        "grant_premium",
        ip_address=request.client.host if request.client else None,
        target_user_id=user_id,
    )
    return {"status": "premium_granted"}


@router.post("/{user_id}/revoke-premium")
def revoke_premium(
    user_id: int,
    db: DbSession,
    request: Request,
    current_admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
) -> dict:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.subscription_type = SubscriptionType.FREE
    if user.role == UserRole.PREMIUM_USER:
        user.role = UserRole.USER
    db.commit()
    log_admin_action(
        db,
        current_admin,
        "revoke_premium",
        ip_address=request.client.host if request.client else None,
        target_user_id=user_id,
    )
    return {"status": "premium_revoked"}


@router.post("/{user_id}/reset-password")
def reset_password(
    user_id: int,
    _: UserActionRequest,
    db: DbSession,
    request: Request,
    current_admin: Annotated[User, Depends(require_role(UserRole.ADMIN))],
) -> dict:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # In production this should create a reset token and send secure email.
    log_admin_action(
        db,
        current_admin,
        "reset_password",
        target_user_id=user_id,
        ip_address=request.client.host if request.client else None,
    )
    return {"status": "reset_initiated"}
