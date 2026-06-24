#!/bin/sh
set -e

export PYTHONPATH=/code
cd /code

if [ -z "$DATABASE_URL" ]; then
  echo "ERROR: DATABASE_URL is not set."
  echo "On Render: Dashboard -> your Postgres -> Connect -> Internal Database URL"
  echo "Paste it as DATABASE_URL on the web service Environment tab."
  exit 1
fi

case "$DATABASE_URL" in
  *@postgres:*|*@postgres/*)
    echo "ERROR: DATABASE_URL points to Docker host 'postgres' (local docker-compose only)."
    echo "Use your Render Postgres Internal Database URL instead."
    exit 1
    ;;
esac

echo "Running database migrations..."
alembic upgrade head

echo "Ensuring all ORM tables exist..."
python3 -c "from app.db.session import Base, engine; Base.metadata.create_all(bind=engine)"

echo "Seeding demo data if database is empty..."
python3 scripts/seed_demo_data.py

echo "Starting API on port ${PORT:-8000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
