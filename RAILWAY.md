# Deploy Backend to Railway

## 1. Push code to GitHub

Railway deploys from a Git repository. Push this repo to GitHub first.

## 2. Create Railway project

1. Go to [railway.app](https://railway.app) and sign in.
2. **New Project** → **Deploy from GitHub repo**.
3. Select your repository.
4. Open the new service → **Settings** → set **Root Directory** to `backend`.

## 3. Add PostgreSQL

1. In the project, click **+ New** → **Database** → **PostgreSQL**.
2. Open your API service → **Variables** → **Add Reference** → select `DATABASE_URL` from Postgres.

Railway provides `postgresql://...`; the app auto-converts it to `postgresql+psycopg://...`.

## 4. Add Redis (optional, for Celery)

1. **+ New** → **Database** → **Redis**.
2. Reference `REDIS_URL` in your API service variables.

## 5. Set environment variables

In the API service **Variables** tab, add:

| Variable | Value |
|----------|--------|
| `JWT_SECRET_KEY` | long random secret |
| `FOOTBALL_API_BASE_URL` | `https://v3.football.api-sports.io` |
| `FOOTBALL_API_KEY` | your API-Football key |
| `SETTINGS_ENCRYPTION_KEY` | 32+ character secret |
| `CORS_ORIGINS` | `https://your-admin.vercel.app` |
| `ADMIN_COOKIE_SECURE` | `true` |
| `ENVIRONMENT` | `production` |

`DATABASE_URL` and `REDIS_URL` come from Railway plugins via variable references.

## 6. Deploy

Railway builds using `backend/Dockerfile` and runs `scripts/start.sh`, which:

1. Runs `alembic upgrade head`
2. Starts `uvicorn` on Railway's `$PORT`

## 7. Public URL

1. Open the API service → **Settings** → **Networking**.
2. Click **Generate Domain** (e.g. `https://football-api-production.up.railway.app`).
3. Test: `https://YOUR-DOMAIN/health`
4. API docs: `https://YOUR-DOMAIN/docs`

## 8. Celery worker (separate service)

Create another service from the same repo:

- **Root Directory:** `backend`
- **Start Command:**
  ```bash
  celery -A app.workers.celery_app.celery worker --loglevel=info
  ```
- Share the same `DATABASE_URL`, `REDIS_URL`, and secrets.

Optional beat scheduler:

```bash
celery -A app.workers.celery_app.celery beat --loglevel=info
```

## 9. Connect admin panel

In `admin-web/.env` (or Vercel env vars):

```env
NEXT_PUBLIC_API_URL=https://YOUR-RAILWAY-API-DOMAIN
```

## Troubleshooting

- **Build fails:** Check Railway build logs; ensure Root Directory is `backend`.
- **DB connection error:** Confirm `DATABASE_URL` is referenced from Postgres.
- **502 on start:** Check deploy logs; migrations may fail if DB is not ready—redeploy once Postgres is up.
