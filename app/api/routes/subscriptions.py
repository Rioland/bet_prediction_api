from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import DbSession, get_current_user
from app.models.entities import Subscription, SubscriptionType, User

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])


@router.post("/verify")
def verify_subscription(
    db: DbSession, current_user: Annotated[User, Depends(get_current_user)], provider_ref: str
) -> dict:
    # Placeholder verification flow for Stripe/Paystack/Flutterwave callback confirmation.
    sub = Subscription(
        user_id=current_user.id,
        provider="manual",
        provider_reference=provider_ref,
        status="active",
    )
    db.add(sub)
    current_user.subscription_type = SubscriptionType.PREMIUM
    db.commit()
    return {"status": "verified", "subscription": "premium"}
