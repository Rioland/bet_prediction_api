from celery import Celery

from app.core.config import settings

celery = Celery("football_ai", broker=settings.redis_url, backend=settings.redis_url)
celery.conf.beat_schedule = {
    "sync-live-matches-every-5-minutes": {
        "task": "app.workers.tasks.sync_live_matches_task",
        "schedule": 300.0,
    },
    "sync-upcoming-fixtures-daily": {
        "task": "app.workers.tasks.sync_upcoming_fixtures_task",
        "schedule": 86400.0,
    },
}
