"""api-football client.

Only the endpoints the pipeline needs. All calls are rate-limit aware: the free
tier allows ~30 requests/minute, so callers should use ``season_fixtures`` (one
request per league-season) rather than looping over dates.
"""

import asyncio
from datetime import date
from typing import Any

import httpx

from app.core.config import settings

# Conservative pacing for the free api-football tier (~30 req/min).
_MIN_INTERVAL_SECONDS = 2.2


class FootballApiClient:
    def __init__(self, min_interval: float = _MIN_INTERVAL_SECONDS) -> None:
        self.base_url = settings.football_api_base_url.rstrip("/")
        self.headers = {"x-apisports-key": settings.football_api_key}
        self._min_interval = min_interval
        self._last_call = 0.0

    async def _get(self, path: str, params: dict[str, Any]) -> dict:
        # Space out calls so a long backfill does not trip the provider limit.
        elapsed = asyncio.get_event_loop().time() - self._last_call
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)

        url = f"{self.base_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            payload = response.json()

        self._last_call = asyncio.get_event_loop().time()
        errors = payload.get("errors")
        # api-football returns HTTP 200 with an errors object on quota/key problems.
        if errors and not isinstance(errors, list):
            raise RuntimeError(f"api-football error on /{path}: {errors}")
        return payload

    async def get_fixtures(self, on_date: date) -> dict:
        return await self._get("fixtures", {"date": str(on_date)})

    async def get_live(self) -> dict:
        return await self._get("fixtures", {"live": "all"})

    async def season_fixtures(self, league_id: int, season: int) -> list[dict]:
        """Every fixture for one league-season - the unit of historical backfill."""
        payload = await self._get("fixtures", {"league": league_id, "season": season})
        return payload.get("response", [])

    async def fixture_statistics(self, fixture_id: int) -> list[dict]:
        payload = await self._get("fixtures/statistics", {"fixture": fixture_id})
        return payload.get("response", [])

    async def fixture_odds(self, fixture_id: int, bookmaker: int | None = None) -> list[dict]:
        params: dict[str, Any] = {"fixture": fixture_id}
        if bookmaker is not None:
            params["bookmaker"] = bookmaker
        payload = await self._get("odds", params)
        return payload.get("response", [])

    async def leagues(self, country: str | None = None) -> list[dict]:
        params = {"country": country} if country else {}
        payload = await self._get("leagues", params)
        return payload.get("response", [])
