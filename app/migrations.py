"""Additive schema changes that create_all cannot make.

SQLAlchemy's create_all builds missing tables but never alters existing ones,
so a deployed database keeps its old shape after a model change. This adds any
missing columns and indexes in place, is safe to run repeatedly, and runs at
startup so a deploy never serves code newer than its schema.

Only additive changes belong here. Anything that drops or rewrites data should
be written and reviewed deliberately, not applied by a startup helper.
"""

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.database import Base
from app.models import DEFAULT_SPORT

# table -> column -> DDL fragment. Must be valid on both PostgreSQL and SQLite.
ADDITIONS: dict[str, dict[str, str]] = {
    "leagues": {"sport": f"VARCHAR(20) NOT NULL DEFAULT '{DEFAULT_SPORT}'"},
    "matches": {"sport": f"VARCHAR(20) NOT NULL DEFAULT '{DEFAULT_SPORT}'"},
    "published_tips": {"sport": f"VARCHAR(20) NOT NULL DEFAULT '{DEFAULT_SPORT}'"},
    "betting_slips": {"odds_are_estimates": "BOOLEAN NOT NULL DEFAULT FALSE"},
}

INDEXES = [
    ("ix_leagues_sport", "leagues", "sport"),
    ("ix_matches_sport", "matches", "sport"),
    ("ix_published_tips_sport", "published_tips", "sport"),
]


def run_migrations(engine: Engine) -> list[str]:
    """Create missing tables, then add missing columns and indexes.

    Returns a description of each change applied; empty when already current.
    """
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    applied: list[str] = []

    with engine.begin() as connection:
        for table, columns in ADDITIONS.items():
            if table not in tables:
                continue
            existing = {c["name"] for c in inspector.get_columns(table)}
            for column, ddl in columns.items():
                if column in existing:
                    continue
                connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
                applied.append(f"{table}.{column}: added")

        for name, table, column in INDEXES:
            if table not in tables:
                continue
            if any(ix["name"] == name for ix in inspector.get_indexes(table)):
                continue
            connection.execute(text(f"CREATE INDEX {name} ON {table} ({column})"))
            applied.append(f"{name}: created")

    return applied
