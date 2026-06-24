from datetime import date

from app.workers.celery_app import celery


@celery.task
def sync_live_matches_task() -> dict:
    # Integrate provider pull and DB upsert in production.
    return {"status": "ok", "synced": "live_matches"}


@celery.task
def sync_upcoming_fixtures_task() -> dict:
    # Integrate provider pull and DB upsert in production.
    return {"status": "ok", "synced": f"fixtures_{date.today()}"}


@celery.task
def retrain_models_task() -> dict:
    from app.ml.train import train_models

    artifacts = train_models("data/historical_matches.csv")
    return {"status": "ok", "artifacts": artifacts}
