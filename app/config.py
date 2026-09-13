import os
import secrets

# In dev, fall back to a randomly generated secret (invalidates sessions on
# restart, but never a guessable well-known string). In production, set the
# JWT_SECRET env var so tokens survive restarts.
JWT_SECRET = os.getenv("JWT_SECRET") or os.getenv("SESSION_SECRET") or secrets.token_hex(32)
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

def _normalise_database_url(url: str) -> str:
    """Render and Heroku hand out postgres:// URLs, which SQLAlchemy 2 rejects."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


DATABASE_URL = _normalise_database_url(
    os.getenv("PREDICTOR_DATABASE_URL") or os.getenv("DATABASE_URL") or "sqlite:///./football_ai.db"
)

# Where trained model artifacts are written and read.
MODEL_DIR = os.getenv("MODEL_DIR", "./models")

# The background fixture refresh reaches the network and writes to the
# database. Tests must be able to switch it off so they are deterministic and
# do not depend on a live feed.
FIXTURE_REFRESH_ENABLED = os.getenv("FIXTURE_REFRESH_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY", "").strip()
FOOTBALL_API_BASE = "https://api.football-data.org/v4"

# Admin seeding is opt-in and credential-free by default. A hardcoded default
# password in a public repository is a published super-admin login for every
# deployment that runs this code, which needs no exploit to use.
DEFAULT_ADMIN_EMAIL = os.getenv("DEFAULT_ADMIN_EMAIL", "").strip()
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "").strip()
DEFAULT_ADMIN_NAME = os.getenv("DEFAULT_ADMIN_NAME", "Super Admin").strip()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
IS_PRODUCTION = ENVIRONMENT == "production"

# Demo accounts share one published password and include admin and moderator
# roles, so they must never exist outside local development.
SEED_DEMO_USERS = os.getenv("SEED_DEMO_USERS", "").strip().lower() in {"1", "true", "yes"}

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]


# ── OPay ─────────────────────────────────────────────────────────────────────
# From the OPay merchant dashboard under API Keys & Webhooks. Never commit these
# and never expose OPAY_SECRET_KEY to the browser.
OPAY_MERCHANT_ID = os.getenv("OPAY_MERCHANT_ID", "").strip()
OPAY_PUBLIC_KEY = os.getenv("OPAY_PUBLIC_KEY", "").strip()
OPAY_SECRET_KEY = os.getenv("OPAY_SECRET_KEY", "").strip()
# Staging unless explicitly switched. Real money only moves when this is set.
OPAY_PRODUCTION = os.getenv("OPAY_PRODUCTION", "").strip().lower() in {"1", "true", "yes"}

# Where the customer lands after paying, and where OPay posts the callback.
# The callback must be a public HTTPS URL: OPay cannot reach localhost.
PUBLIC_SITE_URL = os.getenv("PUBLIC_SITE_URL", "http://localhost:5173").rstrip("/")
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost:8000").rstrip("/")
