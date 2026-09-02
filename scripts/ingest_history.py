#!/usr/bin/env python3
"""Backfill historical fixtures from api-football.

    python scripts/ingest_history.py --leagues 39,140,135 --seasons 2021,2022,2023

League ids are api-football's: 39 Premier League, 140 La Liga, 135 Serie A,
78 Bundesliga, 61 Ligue 1. Each league-season is one request; statistics and
odds cost one request per fixture, so they are opt-in.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import SessionLocal  # noqa: E402
from app.services.football_api import FootballApiClient  # noqa: E402
from app.services.ingest import (  # noqa: E402
    FINISHED_STATUSES,
    apply_odds,
    apply_statistics,
    ingest_fixtures,
)
from app.models.entities import Match  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", required=True, help="comma-separated api-football league ids")
    parser.add_argument("--seasons", required=True, help="comma-separated seasons, e.g. 2022,2023")
    parser.add_argument("--with-stats", action="store_true", help="also fetch per-fixture statistics (slow)")
    parser.add_argument("--with-odds", action="store_true", help="also fetch closing odds (slow)")
    args = parser.parse_args()

    leagues = [int(x) for x in args.leagues.split(",") if x.strip()]
    seasons = [int(x) for x in args.seasons.split(",") if x.strip()]

    client = FootballApiClient()
    db = SessionLocal()
    total = 0
    try:
        for league_id in leagues:
            for season in seasons:
                fixtures = await client.season_fixtures(league_id, season)
                count = ingest_fixtures(db, fixtures)
                total += count
                print(f"league={league_id} season={season}: {count} fixtures")

        if args.with_stats or args.with_odds:
            pending = [
                m for m in db.query(Match).filter(Match.status.in_(FINISHED_STATUSES)).all()
                if m.external_id and (m.home_shots_on_target is None or m.odds_home is None)
            ]
            print(f"enriching {len(pending)} fixtures (this is one request each)...")
            for i, match in enumerate(pending, 1):
                if args.with_stats and match.home_shots_on_target is None:
                    apply_statistics(db, match, await client.fixture_statistics(match.external_id))
                if args.with_odds and match.odds_home is None:
                    apply_odds(db, match, await client.fixture_odds(match.external_id))
                if i % 25 == 0:
                    db.commit()
                    print(f"  {i}/{len(pending)}")
            db.commit()

        print(f"\ningested {total} fixtures total")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
