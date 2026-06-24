from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select

from app.api.deps import DbSession
from app.models.entities import Match, Prediction, SubscriptionType, Transaction, User, UserStatus
from app.services.rbac import require_role
from app.models.entities import UserRole

router = APIRouter(prefix="/admin/analytics", tags=["admin-analytics"])


@router.get("/dashboard")
def dashboard(
    db: DbSession, _: Annotated[User, Depends(require_role(UserRole.ADMIN))]
) -> dict:
    total_users = db.scalar(select(func.count()).select_from(User)) or 0
    active_users = db.scalar(
        select(func.count()).select_from(User).where(User.status == UserStatus.ACTIVE)
    ) or 0
    premium_users = db.scalar(
        select(func.count()).select_from(User).where(User.subscription_type == SubscriptionType.PREMIUM)
    ) or 0
    live_matches = db.scalar(select(func.count()).select_from(Match).where(Match.status == "live")) or 0
    today = datetime.utcnow().date()
    predictions_today = db.scalar(
        select(func.count()).select_from(Prediction).where(func.date(Prediction.created_at) == today)
    ) or 0
    revenue = db.scalar(select(func.coalesce(func.sum(Transaction.amount), 0.0))) or 0.0
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    prev_month_start = (month_start - timedelta(days=1)).replace(day=1)
    current_count = db.scalar(
        select(func.count()).select_from(User).where(User.created_at >= month_start)
    ) or 0
    prev_count = db.scalar(
        select(func.count())
        .select_from(User)
        .where(User.created_at >= prev_month_start, User.created_at < month_start)
    ) or 1
    growth = round((current_count / prev_count) * 100, 2)
    return {
        "total_users": total_users,
        "active_users": active_users,
        "premium_users": premium_users,
        "live_matches": live_matches,
        "predictions_today": predictions_today,
        "revenue": revenue,
        "monthly_growth": growth,
    }


@router.get("/revenue")
def revenue_analytics(
    db: DbSession, _: Annotated[User, Depends(require_role(UserRole.ADMIN))]
) -> dict:
    rows = db.execute(
        select(func.date(Transaction.created_at), func.coalesce(func.sum(Transaction.amount), 0))
        .group_by(func.date(Transaction.created_at))
        .order_by(func.date(Transaction.created_at))
    ).all()
    return {"series": [{"date": str(r[0]), "value": float(r[1])} for r in rows]}


@router.get("/users")
def user_growth(
    db: DbSession, _: Annotated[User, Depends(require_role(UserRole.ADMIN))]
) -> dict:
    rows = db.execute(
        select(func.date(User.created_at), func.count(User.id))
        .group_by(func.date(User.created_at))
        .order_by(func.date(User.created_at))
    ).all()
    return {"series": [{"date": str(r[0]), "value": int(r[1])} for r in rows]}
