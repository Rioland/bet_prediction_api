"""Daily betting slips, and the admin screen for attaching booking codes."""

from __future__ import annotations

from datetime import date as date_type, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from jwt import PyJWTError
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import BettingSlip, User
from app.routes.admin_auth import get_current_admin, get_full_admin
from app.auth import decode_token
from app.services import slips as slips_service
from app.services import subscriptions as subs

router = APIRouter(tags=["slips"])

DbSession = Annotated[Session, Depends(get_db)]
CurrentAdmin = Annotated[User, Depends(get_current_admin)]
# Publishing a code decides what paying subscribers receive.
FullAdmin = Annotated[User, Depends(get_full_admin)]

# Bookmaker booking codes are short alphanumeric strings.
CODE_PATTERN = r"^[A-Za-z0-9\-]{4,40}$"


class CodeInput(BaseModel):
    booking_code: str = Field(pattern=CODE_PATTERN)


def _resolve(db: Session, slip_id: int) -> BettingSlip:
    slip = db.get(BettingSlip, slip_id)
    if slip is None:
        raise HTTPException(status_code=404, detail="Slip not found")
    return slip


def optional_viewer(db: DbSession, authorization: str | None = Header(default=None)) -> User | None:
    """The signed-in viewer if there is one; slips are browsable without an account."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    try:
        data = decode_token(authorization.split(" ", 1)[1].strip())
        if data.get("type") != "access":
            return None
        user = db.get(User, int(data["sub"]))
    except (PyJWTError, ValueError, KeyError):
        return None
    if user is None or user.status in {"suspended", "banned"}:
        return None
    return user


@router.get("/slips")
def list_slips(
    db: DbSession,
    viewer: Annotated[User | None, Depends(optional_viewer)],
    on_date: date_type | None = Query(default=None, alias="date"),
    sport: str = Query(default="football"),
) -> dict[str, Any]:
    """The day's slips.

    Each carries a booking code only if an admin created that slip on the
    bookmaker and recorded the real code. The rest publish their selections so
    a reader can build the slip themselves; none of them invent a code.
    """
    slip_date = on_date or datetime.utcnow().date()
    slips = slips_service.list_slips(db, sport, slip_date)

    # Build on first request for a date so a card is always published, even
    # with nobody around to create slips on the bookmaker. Existing slips are
    # left alone, so any attached code keeps describing its own legs.
    if not slips and sport == "football":
        from app.routes.tips import select_for_date

        selection = select_for_date(db, slip_date)
        slips = slips_service.build_slips(db, selection["pool"], sport, slip_date)

    unlocked = subs.is_active(db, viewer)
    payload = [slips_service.to_dict(s, can_see_code=unlocked) for s in slips]

    return {
        "date": slip_date.isoformat(),
        "sport": sport,
        "count": len(payload),
        "with_codes": sum(1 for s in payload if s["has_code"]),
        "codes_unlocked": unlocked,
        "slips": payload,
    }


@router.post("/admin/slips/{slip_id}/code")
def set_booking_code(
    slip_id: int, payload: CodeInput, db: DbSession, current_admin: FullAdmin
) -> dict[str, Any]:
    """Record a booking code created on the bookmaker for this slip.

    The code is stored exactly as given. It is not validated against the
    bookmaker - we cannot check it resolves - so entering a code for the wrong
    slip publishes the wrong bet. Check it before saving.
    """
    slip = _resolve(db, slip_id)
    updated = slips_service.attach_code(db, slip, payload.booking_code, current_admin.email)
    return slips_service.to_dict(updated)


@router.delete("/admin/slips/{slip_id}/code")
def clear_booking_code(slip_id: int, db: DbSession, current_admin: FullAdmin) -> dict[str, Any]:
    """Remove a code, reverting the slip to publishing its selections only."""
    return slips_service.to_dict(slips_service.remove_code(db, _resolve(db, slip_id)))


@router.get("/admin/slips")
def admin_list_slips(
    db: DbSession,
    current_admin: CurrentAdmin,
    on_date: date_type | None = Query(default=None, alias="date"),
    sport: str = Query(default="football"),
) -> dict[str, Any]:
    """Every slip for a date, including which still need a code."""
    slip_date = on_date or datetime.utcnow().date()
    slips = [slips_service.to_dict(s) for s in slips_service.list_slips(db, sport, slip_date)]
    return {
        "date": slip_date.isoformat(),
        "slips": slips,
        "awaiting_code": [s["id"] for s in slips if not s["has_code"]],
    }


@router.post("/admin/slips/settle")
def settle(db: DbSession, current_admin: CurrentAdmin, sport: str = Query(default="football")) -> dict:
    """Score slips whose legs have all finished."""
    return slips_service.settle_slips(db, sport)
