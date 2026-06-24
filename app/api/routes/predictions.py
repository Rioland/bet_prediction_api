from datetime import date, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from app.api.deps import DbSession, get_current_user
from app.models.entities import Match, Prediction, SubscriptionType, User
from app.schemas.common import PredictionOut

router = APIRouter(tags=["predictions"])


@router.get("/predictions/today", response_model=list[PredictionOut])
def get_today_predictions(db: DbSession) -> list[Prediction]:
    today = date.today()
    start = datetime(today.year, today.month, today.day)
    end = datetime(today.year, today.month, today.day, 23, 59, 59)
    stmt = (
        select(Prediction)
        .join(Match, Prediction.match_id == Match.id)
        .where(Match.kickoff_time >= start, Match.kickoff_time <= end)
    )
    return list(db.scalars(stmt))


@router.get("/predictions/{match_id}", response_model=list[PredictionOut])
def get_match_predictions(match_id: int, db: DbSession) -> list[Prediction]:
    return list(db.scalars(select(Prediction).where(Prediction.match_id == match_id)))


@router.get("/premium/predictions", response_model=list[PredictionOut])
def get_premium_predictions(
    db: DbSession, current_user: Annotated[User, Depends(get_current_user)]
) -> list[Prediction]:
    if current_user.subscription_type != SubscriptionType.PREMIUM:
        raise HTTPException(status_code=402, detail="Premium subscription required")
    return list(db.scalars(select(Prediction).order_by(Prediction.created_at.desc()).limit(50)))
