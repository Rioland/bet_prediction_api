"""Orchestration shared by the CLI scripts and the Celery workers."""

from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.ml.dataset import load_finished_matches, load_upcoming_matches
from app.ml.features import build_dataset, features_for_upcoming
from app.ml.train import train_models


def build_training_frame(db: Session, league_ids: list[int] | None = None) -> pd.DataFrame:
    return build_dataset(load_finished_matches(db, league_ids))


def upcoming_frame(db: Session, league_ids: list[int] | None = None) -> pd.DataFrame:
    return features_for_upcoming(
        load_finished_matches(db, league_ids), load_upcoming_matches(db, league_ids)
    )


def retrain(db: Session, league_ids: list[int] | None = None) -> dict[str, Any]:
    from app.services.prediction_service import clear_model_cache

    summary = train_models(build_training_frame(db, league_ids))
    clear_model_cache()  # so the API serves the newly written models immediately
    return summary
