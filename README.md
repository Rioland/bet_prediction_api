# Football AI Predictor API

FastAPI backend for the prediction site: fixtures, model-based tips, per-fixture
analysis, results tracking, SportyBet booking codes, customer accounts and OPay
subscriptions.

This repository is published from `artifacts/predictor-api` in
[bet_prediction_project](https://github.com/Rioland/bet_prediction_project),
where development happens. The website lives there too.

## Run locally

```bash
cp .env.example .env.local   # fill in
pip install -r requirements.txt
set -a; . ./.env.local; set +a
uvicorn main:app --reload --port 8000
```

## Deploy (Render)

Docker web service building `./Dockerfile`, health check `/healthz`. See
`render.yaml` for every environment variable and what it does. Set at least
`DATABASE_URL` (PostgreSQL) and `JWT_SECRET`.

The schema is created and migrated automatically at startup. To load match
history and train models, run from your machine against the production
database's external URL:

```bash
export PREDICTOR_DATABASE_URL="<external database URL>"
python scripts/backfill_history.py --days 240
python scripts/train_model.py
```

## Tests

```bash
python -m pytest
TEST_DATABASE_URL=postgresql://localhost/predictor_test python -m pytest   # on Postgres
```
