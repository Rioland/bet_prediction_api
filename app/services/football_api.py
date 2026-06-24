from datetime import date

import httpx

from app.core.config import settings


class FootballApiClient:
    def __init__(self) -> None:
        self.base_url = settings.football_api_base_url
        self.headers = {"x-apisports-key": settings.football_api_key}

    async def get_fixtures(self, on_date: date) -> dict:
        url = f"{self.base_url}/fixtures"
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(url, headers=self.headers, params={"date": str(on_date)})
            response.raise_for_status()
            return response.json()

    async def get_live(self) -> dict:
        url = f"{self.base_url}/fixtures"
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(url, headers=self.headers, params={"live": "all"})
            response.raise_for_status()
            return response.json()
