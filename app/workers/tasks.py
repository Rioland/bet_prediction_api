import asyncio
from datetime import date, timedelta

from app.db.session import SessionLocal
from app.workers.celery_app import celery


@celery.task
def sync_live_matches_task() -> dict:
    from app.services.football_api import FootballApiClient
    from app.services.ingest import ingest_fixtures

    db = SessionLocal()
    try:
        payload = asyncio.run(FootballApiClient().get_live())
        count = ingest_fixtures(db, payload.get("response", []))
    finally:
        db.close()
    return {"status": "ok", "synced": count}


@celery.task
def sync_upcoming_fixtures_task(days_ahead: int = 3) -> dict:
    from app.services.football_api import FootballApiClient
    from app.services.ingest import ingest_fixtures

    client = FootballApiClient()
    db = SessionLocal()
    total = 0
    try:
        for offset in range(days_ahead + 1):
            payload = asyncio.run(client.get_fixtures(date.today() + timedelta(days=offset)))
            total += ingest_fixtures(db, payload.get("response", []))
    finally:
        db.close()
    return {"status": "ok", "synced": total}


@celery.task
def retrain_models_task() -> dict:
    from app.ml.pipeline import retrain

    db = SessionLocal()
    try:
        summary = retrain(db)
    finally:
        db.close()
    return {"status": "ok", "summary": summary}
