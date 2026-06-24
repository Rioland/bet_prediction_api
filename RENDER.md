# Deploy Backend to Render

Render is a great choice for this FastAPI stack: API, PostgreSQL, Redis, and Celery workers.

## Option A: Blueprint (recommended)

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New** → **Blueprint**.
3. Connect your GitHub repo.
4. Render reads `render.yaml` at the repo root and creates:
   - **football-api** (Web Service)
   - **football-db** (PostgreSQL)
   - **football-redis** (Redis)
   - **football-celery-worker** (Background Worker)
5. When prompted, set:
   - `FOOTBALL_API_KEY` — your API-Football key
   - `CORS_ORIGINS` — e.g. `https://your-admin.vercel.app`
6. Deploy and wait for the build to finish.
7. Open the API URL → test `/health` and `/docs`.

## Option B: Manual Web Service

1. **New** → **Web Service** → connect GitHub repo.
2. Settings:
   - **Root Directory:** `backend`
   - **Runtime:** Docker
   - **Dockerfile Path:** `Dockerfile`
3. **New** → **PostgreSQL** → copy **Internal Database URL**.
4. Add environment variables:

| Variable | Value |
|----------|--------|
| `DATABASE_URL` | Render Postgres internal URL |
| `REDIS_URL` | Render Redis URL (if using Celery) |
| `JWT_SECRET_KEY` | random secret |
| `FOOTBALL_API_BASE_URL` | `https://v3.football.api-sports.io` |
| `FOOTBALL_API_KEY` | your key |
| `SETTINGS_ENCRYPTION_KEY` | 32+ chars |
| `CORS_ORIGINS` | your admin URL |
| `ADMIN_COOKIE_SECURE` | `true` |
| `ENVIRONMENT` | `production` |

5. **Create Web Service**.

Render sets `PORT` automatically; `scripts/start.sh` uses it.

## Celery worker (manual)

1. **New** → **Background Worker**
2. Same repo, **Root Directory:** `backend`
3. **Docker Command:**
   ```bash
   celery -A app.workers.celery_app.celery worker --loglevel=info
   ```
4. Use the same `DATABASE_URL`, `REDIS_URL`, and API secrets.

## Connect admin panel

```env
NEXT_PUBLIC_API_URL=https://football-api.onrender.com
```

(Set your actual Render URL.)

## Render vs Railway

| Feature | Render | Railway |
|---------|--------|---------|
| FastAPI API | Yes | Yes |
| PostgreSQL | Yes | Yes |
| Redis | Yes | Yes |
| Celery worker | Yes | Yes |
| Free tier | Yes (with sleep) | Limited credits |
| Blueprint file | `render.yaml` | `railway.toml` |

Both work well. Render’s free web services **spin down after inactivity** (~50s cold start on wake).

## Troubleshooting: `failed to resolve host 'postgres'`

This means `DATABASE_URL` still points at the **Docker Compose** hostname (`postgres`), not Render Postgres.

**Fix:**

1. In Render, open your **PostgreSQL** service (create one if missing: **New → PostgreSQL**).
2. Copy **Internal Database URL** (starts with `postgresql://` and a Render hostname like `dpg-xxxxx-a`).
3. Open your **football-api** web service → **Environment**.
4. Set `DATABASE_URL` to that Internal URL (delete any value containing `@postgres:`).
5. Set `ENVIRONMENT` = `production`.
6. **Save** and **Manual Deploy**.

If you used the Blueprint (`render.yaml`), confirm `football-db` exists and `DATABASE_URL` is linked under the web service env vars — not typed manually from `.env.example`.

## Notes

- `DATABASE_URL` from Render uses `postgresql://`; the app converts it automatically.
- Do not commit `.env` — use Render **Environment** tab. `.env` is excluded from Docker builds.
- Health check path: `/health`
