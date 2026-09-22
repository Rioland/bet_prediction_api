"""Fill an empty deployment with match history and a first model, unattended.

A fresh database has no completed matches, so no team has the history the
model needs and there is nothing to train on: every fixture reads "not enough
history" and the tips endpoint answers 503 until someone runs the backfill and
training scripts from a workstation. This does both on the server at startup.

It only acts on what is missing. Once history is loaded and a model is stored
in the database, a restart finds both and returns straight away, so it costs a
count and a lookup on every boot after the first.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

from app.config import MODEL_DIR
from app.database import SessionLocal
from app.football_api import _fetch_espn_fixtures
from app.ml.model_store import restore_model
from app.ml.train import MIN_ROWS, model_filename
from app.routes.operations import _retrain, _training_rows
from app.services.feed_sync import sync_fixtures
from app.sports.registry import get_adapter

logger = logging.getLogger(__name__)

# About four months: enough completed matches to clear MIN_ROWS and to give
# most teams in the covered leagues the five results analysis needs.
BOOTSTRAP_DAYS = 120
_CHUNK_DAYS = 14


def _completed_count(sport: str) -> int:
    db = SessionLocal()
    try:
        return len(_training_rows(db, sport))
    finally:
        db.close()


def _model_stored(sport: str) -> bool:
    """True when every model the sport serves is on disk or in the database."""
    adapter = get_adapter(sport)
    for target in adapter.targets:
        name = model_filename(adapter.name, target)
        if not (Path(MODEL_DIR) / name).exists() and not restore_model(name, MODEL_DIR):
            return False
    return True


def _store(fixtures: list[dict]) -> int:
    db = SessionLocal()
    try:
        return sync_fixtures(db, fixtures)
    finally:
        db.close()


async def _backfill(days: int) -> int:
    """Load completed fixtures for the last ``days`` days, oldest first."""
    today = date.today()
    stored = 0
    for offset in range(days, 0, -_CHUNK_DAYS):
        window_end = today - timedelta(days=max(offset - _CHUNK_DAYS, 0))
        fixtures = await _fetch_espn_fixtures(window_end, days=0, lookback=_CHUNK_DAYS)
        if fixtures:
            # Database writes are blocking; keep them off the event loop so the
            # API goes on answering while history loads.
            stored += await asyncio.to_thread(_store, fixtures)
    return stored


async def bootstrap(sport: str = "football", days: int = BOOTSTRAP_DAYS) -> None:
    """Backfill and train if needed. Logs failures; never raises."""
    try:
        completed = await asyncio.to_thread(_completed_count, sport)
        if completed < MIN_ROWS:
            logger.info(
                "Only %d completed %s matches; backfilling %d days of history",
                completed, sport, days,
            )
            stored = await _backfill(days)
            completed = await asyncio.to_thread(_completed_count, sport)
            logger.info("Backfill stored %d fixtures; %d completed", stored, completed)

        if await asyncio.to_thread(_model_stored, sport):
            return

        if completed < MIN_ROWS:
            logger.warning(
                "No %s model and only %d completed matches (%d needed); "
                "training will wait for more results",
                sport, completed, MIN_ROWS,
            )
            return

        logger.info("No stored %s model; training on %d matches", sport, completed)
        await asyncio.to_thread(_retrain, sport)
    except Exception:
        logger.exception("Startup bootstrap for %s failed", sport)
