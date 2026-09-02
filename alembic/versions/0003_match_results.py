"""Add result, stat and odds columns to matches.

Revision ID: 0003_match_results
"""

import sqlalchemy as sa
from alembic import op

revision: str = "0003_match_results"
down_revision: str | None = "0002_admin_rbac"
branch_labels = None
depends_on = None

_COLUMNS = [
    ("external_id", sa.Integer()),
    ("season", sa.Integer()),
    ("home_goals", sa.Integer()),
    ("away_goals", sa.Integer()),
    ("home_shots_on_target", sa.Integer()),
    ("away_shots_on_target", sa.Integer()),
    ("home_possession", sa.Float()),
    ("away_possession", sa.Float()),
    ("home_corners", sa.Integer()),
    ("away_corners", sa.Integer()),
    ("odds_home", sa.Float()),
    ("odds_draw", sa.Float()),
    ("odds_away", sa.Float()),
]


def upgrade() -> None:
    for name, type_ in _COLUMNS:
        op.add_column("matches", sa.Column(name, type_, nullable=True))
    op.create_index("ix_matches_external_id", "matches", ["external_id"], unique=True)
    op.create_index("ix_matches_season", "matches", ["season"])


def downgrade() -> None:
    op.drop_index("ix_matches_season", table_name="matches")
    op.drop_index("ix_matches_external_id", table_name="matches")
    for name, _ in reversed(_COLUMNS):
        op.drop_column("matches", name)
