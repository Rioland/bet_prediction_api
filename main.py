"""
Football AI Predictor - Python FastAPI backend
Port: 5000  (proxied through Express api-server at /api)

Admin seeding is opt-in: set DEFAULT_ADMIN_EMAIL and DEFAULT_ADMIN_PASSWORD.
There is deliberately no default admin credential.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.database import SessionLocal, engine
from app.migrations import run_migrations
from app.config import CORS_ORIGINS, FIXTURE_REFRESH_ENABLED
from app.seed import seed_admin, seed_demo_users
from app.routes.admin_auth import router as admin_auth_router
from app.routes.admin import router as admin_router
from app.routes.football import router as football_router
from app.routes.tips import router as tips_router
from app.routes.news import router as news_router
from app.routes.operations import router as operations_router
from app.routes.predictions import router as predictions_router
from app.routes.slips import router as slips_router
from app.routes.account import router as account_router
from app.rate_limit import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.football_api import refresh_fixtures_loop


# Set when database setup fails, so /healthz can report why instead of the
# service simply not answering.
STARTUP_ERROR: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global STARTUP_ERROR

    # Database setup must not be able to stop the app from serving. A hosted
    # database that is unreachable used to block startup here, so the platform
    # accepted connections and the app never replied - a silent hang with
    # nothing in the logs to act on. Now it starts, and says what is wrong.
    try:
        for change in run_migrations(engine):
            logging.getLogger(__name__).info("Schema migration: %s", change)
        db = SessionLocal()
        try:
            seed_admin(db)
            seed_demo_users(db)
        finally:
            db.close()
    except Exception as exc:
        STARTUP_ERROR = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
        logging.getLogger(__name__).exception("Database setup failed; serving in a degraded state")

    refresh_task = (
        asyncio.create_task(refresh_fixtures_loop()) if FIXTURE_REFRESH_ENABLED else None
    )
    try:
        yield
    finally:
        if refresh_task is not None:
            refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title="Football AI Predictor API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # A wildcard origin cannot be combined with credentials; browsers reject it,
    # and it would expose authenticated endpoints to any site if they did not.
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_auth_router)
app.include_router(admin_router)
app.include_router(football_router)
app.include_router(tips_router)
app.include_router(news_router)
app.include_router(predictions_router)
app.include_router(operations_router)
app.include_router(slips_router)
app.include_router(account_router)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/healthz")
def healthz():
    """Liveness, plus whether the database is actually usable.

    Deliberately does no database work of its own beyond a trivial probe, so it
    still answers when the database is down - that answer is the diagnosis.
    """
    database = "ok"
    detail = None
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        database = "unavailable"
        detail = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"

    healthy = database == "ok" and STARTUP_ERROR is None
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "degraded",
            "service": "football-ai-predictor-api",
            "database": database,
            "database_error": detail,
            "startup_error": STARTUP_ERROR,
        },
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PREDICTOR_PORT", "5000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
