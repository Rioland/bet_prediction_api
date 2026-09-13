#!/usr/bin/env python3
"""Verify the configured database is reachable and correctly shaped.

    python scripts/check_database.py

Reports the server, the tables present and the row counts, without printing
any credential. Run it after pointing PREDICTOR_DATABASE_URL at a new database
- Supabase, Render, or anything else - to confirm the link before deploying.
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import inspect, text  # noqa: E402

from app.config import DATABASE_URL  # noqa: E402
from app.database import engine  # noqa: E402

EXPECTED = [
    "users", "leagues", "teams", "matches",
    "published_tips", "articles", "betting_slips",
]


def _safe_target(url: str) -> str:
    """Host and database only - never the password."""
    parsed = urlparse(url)
    host = parsed.hostname or "?"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{host}{port}{parsed.path}"


def main() -> int:
    print(f"target: {_safe_target(DATABASE_URL)}")
    print(f"driver: {DATABASE_URL.split('://', 1)[0]}")

    # version() is Postgres-specific; SQLite needs its own probe.
    probe = "SELECT sqlite_version()" if engine.dialect.name == "sqlite" else "SELECT version()"
    try:
        with engine.connect() as connection:
            backend = connection.execute(text(probe)).scalar()
    except Exception as exc:
        print(f"\nCONNECTION FAILED: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        print(
            "\nFor Supabase, check you used the connection string from "
            "Settings > Database, not the REST URL, and that the password is filled in."
        )
        return 1

    print(f"server: {engine.dialect.name} {str(backend).split(' on ')[0]}")

    inspector = inspect(engine)
    present = set(inspector.get_table_names())
    missing = [t for t in EXPECTED if t not in present]

    print("\ntables:")
    with engine.connect() as connection:
        for table in EXPECTED:
            if table not in present:
                print(f"  {table:16} MISSING")
                continue
            count = connection.execute(text(f"SELECT count(*) FROM {table}")).scalar()
            print(f"  {table:16} {count:>8} rows")

    if missing:
        print(f"\n{len(missing)} table(s) missing. Run: python scripts/migrate_schema.py")
        return 1

    print("\nSchema is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
