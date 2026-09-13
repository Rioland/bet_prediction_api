#!/usr/bin/env python3
"""Backfill completed fixtures so the model has something to learn from.

The refresh loop only looks a few days back. On a new deployment that means
weeks of waiting before any team reaches the minimum history and fixtures stop
reading "not enough history to analyse". This walks further back and fills the
gap in one run.

    python scripts/backfill_history.py --days 240

Uses ESPN's public scoreboard, which needs no API key. Requests are chunked
because the scoreboard caps how much one range returns.
"""

import argparse
import asyncio
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import Base, SessionLocal, engine  # noqa: E402
from app.football_api import _fetch_espn_fixtures  # noqa: E402
from app.models import Match, Team  # noqa: E402
from app.services.feed_sync import FINISHED_STATUSES, sync_fixtures  # noqa: E402

CHUNK_DAYS = 14


async def backfill(days: int, chunk: int = CHUNK_DAYS) -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    total = finished = 0
    try:
        today = date.today()
        for offset in range(days, 0, -chunk):
            window_end = today - timedelta(days=max(offset - chunk, 0))
            fixtures = await _fetch_espn_fixtures(window_end, days=0, lookback=chunk)
            if not fixtures:
                print(f"  {window_end}: nothing returned")
                continue

            done = [f for f in fixtures if f.get("status") in FINISHED_STATUSES]
            synced = sync_fixtures(db, fixtures)
            total += synced
            finished += len(done)
            print(f"  up to {window_end}: {synced} synced, {len(done)} completed")

        played = Counter()
        for match in db.query(Match).filter(Match.home_goals.isnot(None)).all():
            played[match.home_team_id] += 1
            played[match.away_team_id] += 1

        ready = sum(1 for count in played.values() if count >= 5)
        print(
            f"\n{total} fixtures synced, {finished} with results.\n"
            f"{ready} of {db.query(Team).count()} teams now have the 5 completed "
            "matches needed to be analysable."
        )
        if ready == 0:
            print(
                "\nNo team reached the threshold. Either the window is too short, "
                "or the source returned no completed fixtures for it — try a larger "
                "--days value."
            )
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180, help="how far back to reach")
    parser.add_argument("--chunk", type=int, default=CHUNK_DAYS, help="days per request")
    args = parser.parse_args()

    print(f"Backfilling {args.days} days in {args.chunk}-day chunks...")
    asyncio.run(backfill(args.days, args.chunk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
