"""Pluggable match-data sources.

A provider is anything that can answer three questions for a sport: what is
scheduled, what finished and what the market priced it at. Keeping that behind
a protocol means a provider swap - or a second provider for basketball - does
not reach into ingestion, features or training.

Nothing here talks to a specific vendor. The ESPN and football-data clients in
``app.football_api`` are adapted onto this interface in ``sources_espn``.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol, runtime_checkable

# One normalised fixture. Providers translate into this shape; everything
# downstream reads only these keys.
#
#   external_id   provider's stable id for the fixture
#   sport         "football" | "basketball"
#   league_id     provider's competition id
#   league_name   display name
#   home_team     display name (teams are keyed by name; see feed_sync)
#   away_team     display name
#   kickoff       ISO 8601 string, UTC
#   status        "NS" | "LIVE" | "FT" | ...
#   home_score    int | None - None until played
#   away_score    int | None
#   odds_home     float | None - decimal odds
#   odds_draw     float | None - None for sports without a draw
#   odds_away     float | None
Fixture = dict[str, Any]


@runtime_checkable
class DataSource(Protocol):
    """What the pipeline needs from any provider."""

    #: Sports this source can serve, e.g. {"football"}.
    sports: frozenset[str]

    #: Short identifier used in logs and provenance.
    name: str

    async def fetch_fixtures(
        self, sport: str, start: date, end: date
    ) -> list[Fixture]:
        """Scheduled and completed fixtures in a date range, inclusive.

        Completed fixtures must carry scores; this is how history accumulates.
        """
        ...

    async def fetch_odds(self, sport: str, fixtures: list[Fixture]) -> list[Fixture]:
        """Attach market odds where available.

        Returns the fixtures with odds fields populated. A source with no odds
        coverage returns them unchanged rather than raising - odds are optional
        everywhere except value detection.
        """
        ...


class NullOdds:
    """Mixin for sources that carry fixtures but no prices."""

    async def fetch_odds(self, sport: str, fixtures: list[Fixture]) -> list[Fixture]:
        return fixtures


def supports(source: DataSource, sport: str) -> bool:
    return sport in source.sports
