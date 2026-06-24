#!/bin/sh
set -e

export PYTHONPATH=/code
cd /code

echo "Running database migrations..."
alembic upgrade head

echo "Ensuring all ORM tables exist..."
python3 -c "from app.db.session import Base, engine; Base.metadata.create_all(bind=engine)"

echo "Starting API on port ${PORT:-8000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
