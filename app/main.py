import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes import (
    admin,
    admin_analytics,
    admin_auth,
    admin_notifications,
    admin_reports,
    admin_settings,
    admin_subscriptions,
    admin_users,
    auth,
    matches,
    notifications,
    predictions,
    subscriptions,
)
from app.core.config import settings
from app.core.csrf import CSRFMiddleware
from app.db.session import Base, engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("football_ai")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=settings.jwt_secret_key, https_only=settings.admin_cookie_secure)
app.add_middleware(CSRFMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(auth.router)
app.include_router(matches.router)
app.include_router(predictions.router)
app.include_router(subscriptions.router)
app.include_router(notifications.router)
app.include_router(admin.router)
app.include_router(admin_auth.router)
app.include_router(admin_users.router)
app.include_router(admin_analytics.router)
app.include_router(admin_notifications.router)
app.include_router(admin_reports.router)
app.include_router(admin_settings.router)
app.include_router(admin_subscriptions.router)


@app.get("/health")
@limiter.limit("20/minute")
def health(request: Request) -> dict:
    return {"status": "ok", "service": settings.app_name}
