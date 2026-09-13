"""Published tips and articles.

Revision ID: 0004_tips_and_articles
"""

import sqlalchemy as sa
from alembic import op

revision: str = "0004_tips_and_articles"
down_revision: str | None = "0003_match_results"
branch_labels = None
depends_on = None

TIP_RESULT = sa.Enum("PENDING", "WON", "LOST", "VOID", name="tipresult")


def upgrade() -> None:
    op.create_table(
        "published_tips",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("market", sa.String(40), nullable=False),
        sa.Column("selection", sa.String(40), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("odds", sa.Float(), nullable=False),
        sa.Column("is_vip", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("rationale", sa.String(400)),
        sa.Column("published_at", sa.DateTime(), nullable=False),
        sa.Column("kickoff_time", sa.DateTime(), nullable=False),
        sa.Column("result", TIP_RESULT, nullable=False, server_default="PENDING"),
        sa.Column("settled_at", sa.DateTime()),
        sa.UniqueConstraint("match_id", "market", "selection", name="uq_tip_selection"),
    )
    op.create_index("ix_published_tips_match_id", "published_tips", ["match_id"])
    op.create_index("ix_published_tips_result", "published_tips", ["result"])
    op.create_index("ix_published_tips_published_at", "published_tips", ["published_at"])
    op.create_index("ix_published_tips_kickoff_time", "published_tips", ["kickoff_time"])
    op.create_index("ix_published_tips_is_vip", "published_tips", ["is_vip"])
    op.create_index("ix_published_tips_market", "published_tips", ["market"])

    op.create_table(
        "articles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("slug", sa.String(200), nullable=False, unique=True),
        sa.Column("title", sa.String(250), nullable=False),
        sa.Column("excerpt", sa.String(500)),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("cover_image", sa.String(512)),
        sa.Column("author", sa.String(120)),
        sa.Column("published", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("published_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_articles_slug", "articles", ["slug"], unique=True)
    op.create_index("ix_articles_published", "articles", ["published"])
    op.create_index("ix_articles_published_at", "articles", ["published_at"])


def downgrade() -> None:
    op.drop_table("articles")
    op.drop_table("published_tips")
    TIP_RESULT.drop(op.get_bind(), checkfirst=True)
