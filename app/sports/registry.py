"""Sport lookup.

Routes take a sport as a path parameter, so an unknown value must fail as a
404 rather than a KeyError.
"""

from __future__ import annotations

from fastapi import HTTPException

from app.sports import SportAdapter
from app.sports.basketball import BasketballAdapter
from app.sports.football import FootballAdapter

_ADAPTERS: dict[str, SportAdapter] = {
    FootballAdapter.name: FootballAdapter(),
    BasketballAdapter.name: BasketballAdapter(),
}

DEFAULT_SPORT = FootballAdapter.name


def available_sports() -> list[str]:
    return sorted(_ADAPTERS)


def get_adapter(sport: str) -> SportAdapter:
    adapter = _ADAPTERS.get(sport.lower().strip())
    if adapter is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown sport '{sport}'. Available: {', '.join(available_sports())}.",
        )
    return adapter
