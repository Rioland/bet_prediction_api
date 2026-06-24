#!/bin/sh
# Local dev without Docker — uses SQLite (no Postgres/Redis required for API testing)

set -e
cd "$(dirname "$0")/.."

echo "==> Installing Python dependencies (if needed)..."
python3 -m pip install -q \
  fastapi "uvicorn[standard]" pydantic pydantic-settings sqlalchemy alembic \
  python-jose bcrypt pyotp cryptography slowapi httpx itsdangerous email-validator \
  redis celery 2>/dev/null || true

export DATABASE_URL="sqlite:///./football_ai.db"
export REDIS_URL="redis://localhost:6379/0"
# Other vars load from .env (JWT, API keys, etc.)

echo "==> Creating database tables..."
PYTHONPATH=. python3 -c "from app.db.session import Base, engine; Base.metadata.create_all(bind=engine)"

echo "==> Starting API at http://localhost:8000"
echo "    Docs:  http://localhost:8000/docs"
echo "    Health: http://localhost:8000/health"
echo ""
PYTHONPATH=. python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
