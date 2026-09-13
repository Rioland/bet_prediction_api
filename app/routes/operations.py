"""Admin operations.

Retraining is slow and writes model artifacts that inference reads, so it runs
in the background and is guarded by the existing admin RBAC.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.config import MODEL_DIR
from app.ml.dataset import load_finished_matches
from app.ml.model_store import publish_models
from app.ml.train import MIN_ROWS, train_for_sport
from app.models import User
from app.routes.admin_auth import get_current_admin
from app.services.prediction_service import clear_model_cache
from app.sports.registry import available_sports, get_adapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/operations", tags=["operations"])

DbSession = Annotated[Session, Depends(get_db)]
CurrentAdmin = Annotated[User, Depends(get_current_admin)]


def _training_rows(db: Session, sport: str) -> list[dict[str, Any]]:
    return [m for m in load_finished_matches(db) if m.get("sport", "football") == sport]


def _retrain(sport: str) -> None:
    """Run training on its own session; background tasks outlive the request."""
    adapter = get_adapter(sport)
    db = SessionLocal()
    try:
        frame = adapter.build_dataset(_training_rows(db, sport))
        summary = train_for_sport(adapter, frame)
        publish_models(MODEL_DIR, sport)  # survive a restart on ephemeral disks
        clear_model_cache()  # so the next request serves the new models
        logger.info("Retrained %s on %d rows", sport, summary["rows"])
    except Exception:
        logger.exception("Retraining %s failed", sport)
    finally:
        db.close()


@router.post("/retrain", status_code=202)
def retrain(
    db: DbSession,
    current_admin: CurrentAdmin,
    background: BackgroundTasks,
    sport: str = Query(default="football"),
) -> dict[str, Any]:
    """Retrain a sport's models. Admin only.

    Returns 202 and trains in the background: a full run takes minutes and
    would otherwise hold the connection open past most proxy timeouts.

    The row count is checked up front so an obviously doomed run fails loudly
    here rather than silently in a background task.
    """
    adapter = get_adapter(sport)
    rows = len(_training_rows(db, adapter.name))
    if rows < MIN_ROWS:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Only {rows} completed {adapter.name} matches; {MIN_ROWS} are needed "
                "for a temporal split to mean anything. Ingest more history first."
            ),
        )

    background.add_task(_retrain, adapter.name)
    logger.info("Retrain of %s requested by %s", adapter.name, current_admin.email)

    return {
        "status": "accepted",
        "sport": adapter.name,
        "rows": rows,
        "targets": adapter.targets,
        "requested_by": current_admin.email,
        "note": "Training runs in the background; check the training report for results.",
    }


@router.get("/sports")
def operations_sports(current_admin: CurrentAdmin) -> dict[str, Any]:
    """Sports available to retrain, with how much history each has."""
    db = SessionLocal()
    try:
        return {
            "sports": [
                {
                    "sport": name,
                    "completed_matches": len(_training_rows(db, name)),
                    "trainable": len(_training_rows(db, name)) >= MIN_ROWS,
                    "minimum_required": MIN_ROWS,
                }
                for name in available_sports()
            ]
        }
    finally:
        db.close()
