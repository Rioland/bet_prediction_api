"""admin and rbac tables

Revision ID: 0002_admin_rbac
Revises: 0001_initial
Create Date: 2026-06-17
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_admin_rbac"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TYPE userrole ADD VALUE IF NOT EXISTS 'PREMIUM_USER'")
    op.execute("ALTER TYPE userrole ADD VALUE IF NOT EXISTS 'MODERATOR'")
    op.execute("ALTER TYPE userrole ADD VALUE IF NOT EXISTS 'SUPER_ADMIN'")
    op.execute("CREATE TYPE userstatus AS ENUM ('ACTIVE', 'SUSPENDED', 'BANNED')")
    op.execute(
        """
        ALTER TABLE users ADD COLUMN IF NOT EXISTS status userstatus NOT NULL DEFAULT 'ACTIVE';
        ALTER TABLE users ADD COLUMN IF NOT EXISTS two_factor_enabled BOOLEAN NOT NULL DEFAULT FALSE;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS two_factor_secret VARCHAR(255);
        ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP;
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS roles (
            id SERIAL PRIMARY KEY,
            name VARCHAR(64) UNIQUE NOT NULL,
            description VARCHAR(255)
        );
        CREATE TABLE IF NOT EXISTS permissions (
            id SERIAL PRIMARY KEY,
            name VARCHAR(64) UNIQUE NOT NULL,
            description VARCHAR(255)
        );
        CREATE TABLE IF NOT EXISTS user_roles (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            role_id INTEGER NOT NULL REFERENCES roles(id),
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS admin_logs (
            id SERIAL PRIMARY KEY,
            admin_id INTEGER NOT NULL REFERENCES users(id),
            action VARCHAR(128) NOT NULL,
            target_user_id INTEGER REFERENCES users(id),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            ip_address VARCHAR(64),
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS login_history (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            ip_address VARCHAR(64),
            device VARCHAR(255),
            location VARCHAR(255),
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS devices (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            device_id VARCHAR(128) UNIQUE NOT NULL,
            platform VARCHAR(50) NOT NULL,
            push_token VARCHAR(255),
            last_active TIMESTAMP NOT NULL DEFAULT NOW()
        );
        CREATE TYPE reportstatus AS ENUM ('OPEN', 'RESOLVED');
        CREATE TABLE IF NOT EXISTS reports (
            id SERIAL PRIMARY KEY,
            reporter_user_id INTEGER NOT NULL REFERENCES users(id),
            target_user_id INTEGER REFERENCES users(id),
            category VARCHAR(64) NOT NULL,
            message TEXT NOT NULL,
            status reportstatus NOT NULL DEFAULT 'OPEN',
            moderation_notes TEXT,
            resolved_by INTEGER REFERENCES users(id),
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            resolved_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS system_settings (
            id SERIAL PRIMARY KEY,
            key VARCHAR(128) UNIQUE NOT NULL,
            encrypted_value TEXT NOT NULL,
            updated_by INTEGER NOT NULL REFERENCES users(id),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS system_settings;
        DROP TABLE IF EXISTS reports;
        DROP TYPE IF EXISTS reportstatus;
        DROP TABLE IF EXISTS devices;
        DROP TABLE IF EXISTS login_history;
        DROP TABLE IF EXISTS admin_logs;
        DROP TABLE IF EXISTS user_roles;
        DROP TABLE IF EXISTS permissions;
        DROP TABLE IF EXISTS roles;
        """
    )
