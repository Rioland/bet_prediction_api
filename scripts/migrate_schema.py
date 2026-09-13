#!/usr/bin/env python3
"""Bring the database schema up to date.

The API does this itself at startup; this is for applying it by hand, e.g.
against a production database before a deploy.

    python scripts/migrate_schema.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import engine  # noqa: E402
from app.migrations import run_migrations  # noqa: E402


def main() -> int:
    applied = run_migrations(engine)
    for change in applied:
        print(f"  {change}")
    print(f"\n{len(applied)} change(s) applied." if applied else "\nSchema already current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
