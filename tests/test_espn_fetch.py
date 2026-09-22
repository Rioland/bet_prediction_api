"""ESPN must be asked for one day at a time.

Its scoreboard answers a date range ("dates=20260901-20260914") with 400, and
the fetchers swallow errors, so a range request fails silently: every refresh
and backfill came back empty while looking healthy.
"""

import asyncio
from datetime import date

import httpx
import pytest

import app.football_api as football_api
from app.football_api import (
    ESPN_COMPETITIONS,
    _dates_in_window,
    _fetch_espn_fixtures,
    _fetch_espn_fixtures_sync,
)

TODAY = date(2026, 9, 20)


def _event(slug: str, day: str) -> dict:
    return {
        "id": f"{abs(hash((slug, day))) % 1_000_000}",
        "date": f"{day[:4]}-{day[4:6]}-{day[6:]}T15:00Z",
        "status": {"type": {"state": "post", "completed": True}},
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "score": "2", "team": {"id": "1", "displayName": f"{slug} Home"}},
                {"homeAway": "away", "score": "1", "team": {"id": "2", "displayName": f"{slug} Away"}},
            ],
        }],
    }


def _espn(requested: list[str]):
    """Behaves like the real scoreboard: a range is rejected, a day is served."""

    def handler(request: httpx.Request) -> httpx.Response:
        dates = request.url.params["dates"]
        requested.append(dates)
        if "-" in dates:
            return httpx.Response(400, json={"code": 400, "message": "Invalid dates"})
        slug = request.url.path.split("/")[-2]
        return httpx.Response(200, json={"events": [_event(slug, dates)]})

    return httpx.MockTransport(handler)


@pytest.fixture
def espn(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    requested: list[str] = []
    transport = _espn(requested)
    real_async, real_sync = httpx.AsyncClient, httpx.Client
    monkeypatch.setattr(
        football_api.httpx, "AsyncClient",
        lambda **kwargs: real_async(transport=transport, **kwargs),
    )
    monkeypatch.setattr(
        football_api.httpx, "Client",
        lambda **kwargs: real_sync(transport=transport, **kwargs),
    )
    return requested


def test_window_includes_both_ends() -> None:
    window = _dates_in_window(TODAY, days=2, lookback=3)
    assert window[0] == date(2026, 9, 17)
    assert window[-1] == date(2026, 9, 22)
    assert len(window) == 6


def test_window_of_one_day() -> None:
    assert _dates_in_window(TODAY, days=0) == [TODAY]


def test_async_fetch_asks_for_single_days(espn: list[str]) -> None:
    fixtures = asyncio.run(_fetch_espn_fixtures(TODAY, days=1, lookback=1))

    assert espn, "no request was made"
    assert all("-" not in dates for dates in espn)
    assert sorted(set(espn)) == ["20260919", "20260920", "20260921"]
    assert len(fixtures) == len(ESPN_COMPETITIONS) * 3


def test_sync_fetch_asks_for_single_days(espn: list[str]) -> None:
    fixtures = _fetch_espn_fixtures_sync(TODAY, days=2)

    assert all("-" not in dates for dates in espn)
    assert sorted(set(espn)) == ["20260920", "20260921", "20260922"]
    assert len(fixtures) == len(ESPN_COMPETITIONS) * 3


def test_completed_results_are_parsed(espn: list[str]) -> None:
    fixtures = asyncio.run(_fetch_espn_fixtures(TODAY, days=0))

    assert fixtures
    assert all(f["status"] == "FT" for f in fixtures)
    assert all((f["home_score"], f["away_score"]) == (2, 1) for f in fixtures)
