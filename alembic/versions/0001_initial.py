"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-17
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE TYPE subscriptiontype AS ENUM ('FREE', 'PREMIUM')")
    op.execute("CREATE TYPE userrole AS ENUM ('USER', 'ADMIN')")
    op.execute(
        """
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(120) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            avatar VARCHAR(512),
            subscription_type subscriptiontype NOT NULL DEFAULT 'FREE',
            role userrole NOT NULL DEFAULT 'USER',
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        CREATE INDEX ix_users_email ON users(email);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS users")
    op.execute("DROP TYPE IF EXISTS userrole")
    op.execute("DROP TYPE IF EXISTS subscriptiontype")
