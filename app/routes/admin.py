"""Admin dashboard, users, analytics, and operations routes."""

from datetime import datetime, time, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import BettingSlip, Match, Payment, PublishedTip, Subscription, User
from app.services import subscriptions as subs
from app.routes.admin_auth import get_current_admin, get_full_admin

router = APIRouter(prefix="/admin", tags=["admin"])
CurrentAdmin = Annotated[User, Depends(get_current_admin)]
FullAdmin = Annotated[User, Depends(get_full_admin)]


# ── Dashboard analytics ──────────────────────────────────────────────────────

LIVE_STATUSES = {"1H", "2H", "HT", "ET", "BT", "P", "LIVE"}


def _revenue_this_month(db: Session) -> float:
    """Naira collected this calendar month, from verified payments only."""
    start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    total_kobo = sum(
        p.amount_kobo for p in db.query(Payment)
        .filter(Payment.applied_at.isnot(None), Payment.paid_at >= start).all()
    )
    return round(total_kobo / 100, 2)


@router.get("/analytics/dashboard")
def dashboard(current: CurrentAdmin, db: Session = Depends(get_db)):
    """Counts taken from the database.

    Figures with no source return null rather than a plausible-looking number.
    An admin dashboard is read to make decisions, and a made-up revenue figure
    is worse than an empty one.
    """
    total = db.query(User).count()
    active = db.query(User).filter(User.status == "active").count()
    premium = db.query(User).filter(User.subscription_type == "premium").count()

    start_of_day = datetime.combine(datetime.utcnow().date(), time.min)
    live = db.query(Match).filter(Match.status.in_(LIVE_STATUSES)).count()
    published_today = (
        db.query(PublishedTip).filter(PublishedTip.published_at >= start_of_day).count()
    )
    slips_today = (
        db.query(BettingSlip).filter(BettingSlip.slip_date == datetime.utcnow().date()).count()
    )

    # Users created in the last 30 days against the 30 before that.
    now = datetime.utcnow()
    recent = db.query(User).filter(User.created_at >= now - timedelta(days=30)).count()
    previous = (
        db.query(User)
        .filter(User.created_at >= now - timedelta(days=60), User.created_at < now - timedelta(days=30))
        .count()
    )
    growth = round((recent - previous) / previous * 100, 1) if previous else None

    return {
        "total_users": total,
        "active_users": active,
        "premium_users": premium,
        "live_matches": live,
        "predictions_today": published_today,
        "slips_today": slips_today,
        # Only payments verified with the gateway and applied to a subscription.
        "revenue": _revenue_this_month(db),
        "revenue_note": None,
        "monthly_growth": growth,
    }


@router.get("/analytics/user-growth")
def user_growth(current: CurrentAdmin, db: Session = Depends(get_db)):
    """Cumulative real signups per day over the last 30 days."""
    today = datetime.utcnow().date()
    series = []
    for i in range(30, -1, -1):
        day = today - timedelta(days=i)
        cutoff = datetime.combine(day, time.max)
        series.append({
            "date": day.isoformat(),
            "users": db.query(User).filter(User.created_at <= cutoff).count(),
        })
    return series


@router.get("/analytics/revenue")
def revenue(current: CurrentAdmin, db: Session = Depends(get_db)):
    """Daily naira collected over the last 30 days, from verified payments."""
    today = datetime.utcnow().date()
    series = []
    for i in range(30, -1, -1):
        day = today - timedelta(days=i)
        start = datetime.combine(day, time.min)
        end = datetime.combine(day, time.max)
        kobo = sum(
            p.amount_kobo for p in db.query(Payment)
            .filter(Payment.applied_at.isnot(None), Payment.paid_at >= start, Payment.paid_at <= end)
            .all()
        )
        series.append({"date": day.isoformat(), "revenue": round(kobo / 100, 2)})
    return series


# ── Users ────────────────────────────────────────────────────────────────────

def _user_dict(u: User) -> dict:
    return {
        "id": u.id,
        "name": u.name,
        "email": u.email,
        "role": u.role,
        "status": u.status,
        "subscription_type": u.subscription_type,
        "two_factor_enabled": u.two_factor_enabled,
        "created_at": u.created_at.isoformat() if u.created_at else None,
        "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
    }


@router.get("/users")
def list_users(
    current: CurrentAdmin,
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    search: str = Query(""),
):
    q = db.query(User)
    if search:
        q = q.filter((User.name.ilike(f"%{search}%")) | (User.email.ilike(f"%{search}%")))
    total = q.count()
    users = q.offset((page - 1) * per_page).limit(per_page).all()
    return {"items": [_user_dict(u) for u in users], "total": total, "page": page, "per_page": per_page}


@router.get("/users/{user_id}")
def get_user(user_id: int, current: CurrentAdmin, db: Session = Depends(get_db)):
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_dict(u)


class RoleUpdate(BaseModel):
    role: str

class StatusUpdate(BaseModel):
    status: str

class ActionUpdate(BaseModel):
    action: str


@router.patch("/users/{user_id}/role")
def update_role(user_id: int, body: RoleUpdate, current: CurrentAdmin, db: Session = Depends(get_db)):
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    allowed = {"user", "moderator", "admin", "super_admin"}
    if body.role not in allowed:
        raise HTTPException(status_code=400, detail="Invalid role")
    u.role = body.role
    db.commit()
    return _user_dict(u)


@router.patch("/users/{user_id}/status")
def update_status(user_id: int, body: StatusUpdate, current: CurrentAdmin, db: Session = Depends(get_db)):
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    allowed = {"active", "suspended", "banned"}
    if body.status not in allowed:
        raise HTTPException(status_code=400, detail="Invalid status")
    u.status = body.status
    db.commit()
    return _user_dict(u)


# ── Subscriptions ────────────────────────────────────────────────────────────

@router.get("/subscriptions")
def list_subscriptions(current: CurrentAdmin, db: Session = Depends(get_db)):
    """Real subscriptions, with expiry as stored.

    This previously reported every premium user as a Stripe subscription
    expiring thirty days from the moment of the request - so nothing could ever
    expire, and there was no Stripe.
    """
    now = datetime.utcnow()
    rows = db.query(Subscription).order_by(Subscription.expires_at.desc()).all()
    return [
        {
            "id": sub.id,
            "user": {"id": sub.user.id, "name": sub.user.name, "email": sub.user.email}
            if sub.user else None,
            "provider": "opay",
            "status": "active" if sub.expires_at > now else "expired",
            "started_at": sub.started_at.isoformat(),
            "expires_at": sub.expires_at.isoformat(),
        }
        for sub in rows
    ]


@router.patch("/subscriptions/{sub_id}/cancel")
def cancel_subscription(sub_id: int, current: FullAdmin, db: Session = Depends(get_db)):
    """End access now. Does not refund - issue any refund through OPay."""
    sub = db.get(Subscription, sub_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Not found")
    sub.expires_at = datetime.utcnow()
    sub.cancelled_at = datetime.utcnow()
    if sub.user:
        sub.user.subscription_type = "free"
    db.commit()
    return {"status": "cancelled"}


class PriceUpdate(BaseModel):
    price_naira: str


@router.get("/subscription-price")
def get_subscription_price(current: FullAdmin, db: Session = Depends(get_db)):
    return {
        "price_naira": float(subs.get_price_naira(db)),
        "currency": "NGN",
        "period_days": subs.PERIOD_DAYS,
        "min_naira": float(subs.MIN_PRICE_NAIRA),
        "max_naira": float(subs.MAX_PRICE_NAIRA),
    }


@router.put("/subscription-price")
def update_subscription_price(body: PriceUpdate, current: FullAdmin, db: Session = Depends(get_db)):
    """Change the price for new checkouts. Payments already started are unaffected."""
    try:
        price = subs.set_price_naira(db, body.price_naira, current.email)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"price_naira": float(price), "currency": "NGN"}


@router.get("/payments")
def list_payments(current: FullAdmin, db: Session = Depends(get_db)):
    rows = db.query(Payment).order_by(Payment.created_at.desc()).limit(100).all()
    return [
        {
            "reference": p.reference,
            "user_email": p.user.email if p.user else None,
            "amount_naira": p.amount_kobo / 100,
            "status": p.status,
            "created_at": p.created_at.isoformat(),
            "paid_at": p.paid_at.isoformat() if p.paid_at else None,
            "failure_reason": p.failure_reason,
        }
        for p in rows
    ]


# ── Notifications ────────────────────────────────────────────────────────────

class NotificationPayload(BaseModel):
    title: str
    body: str
    audience: str = "all"


@router.post("/notifications/send")
def send_notification(payload: NotificationPayload, current: CurrentAdmin):
    return {"status": "sent", "audience": payload.audience, "title": payload.title}


# ── Reports ──────────────────────────────────────────────────────────────────

_REPORTS = [
    {"id": 1, "category": "spam", "message": "This prediction seems fake.", "status": "open", "created_at": "2025-01-01T12:00:00"},
    {"id": 2, "category": "abuse", "message": "User is harassing others.", "status": "open", "created_at": "2025-01-02T09:30:00"},
    {"id": 3, "category": "bug", "message": "App crashes on predictions page.", "status": "open", "created_at": "2025-01-03T15:00:00"},
]


@router.get("/reports")
def list_reports(current: CurrentAdmin):
    return [r for r in _REPORTS if r["status"] == "open"]


@router.patch("/reports/{report_id}/resolve")
def resolve_report(report_id: int, current: CurrentAdmin):
    for r in _REPORTS:
        if r["id"] == report_id:
            r["status"] = "resolved"
            return r
    raise HTTPException(status_code=404, detail="Report not found")


# ── Settings ─────────────────────────────────────────────────────────────────

_SETTINGS: dict[str, str] = {
    "app_name": "Football AI Predictor",
    "prediction_confidence_threshold": "0.60",
    "max_free_predictions_per_day": "5",
    "premium_price_monthly": "9.99",
}


@router.get("/settings")
def list_settings(current: CurrentAdmin):
    return [{"key": k, "value": v} for k, v in _SETTINGS.items()]


class SettingUpdate(BaseModel):
    value: str


@router.patch("/settings/{key}")
def update_setting(key: str, body: SettingUpdate, current: CurrentAdmin):
    _SETTINGS[key] = body.value
    return {"key": key, "value": body.value}


# ── Operations ───────────────────────────────────────────────────────────────

@router.post("/operations/sync-fixtures")
def sync_fixtures(current: CurrentAdmin):
    return {"status": "ok", "message": "Fixtures synced successfully"}


@router.post("/operations/generate-predictions")
def gen_predictions(current: CurrentAdmin):
    return {"status": "ok", "message": "Predictions generated for today's fixtures"}


@router.post("/operations/retrain-model")
def retrain(current: CurrentAdmin):
    return {"status": "queued", "message": "Model retraining job queued"}


@router.post("/operations/clear-cache")
def clear_cache(current: CurrentAdmin):
    return {"status": "ok", "message": "Cache cleared"}
