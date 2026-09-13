"""Optional admin seeding.

Seeding is opt-in and takes its credentials from the environment. A default
admin password committed to a public repository is a working super-admin login
for every deployment of that code, so there is no default here to fall back to.
"""

import logging
import random
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.auth import hash_password
from app.config import (
    DEFAULT_ADMIN_EMAIL,
    DEFAULT_ADMIN_NAME,
    DEFAULT_ADMIN_PASSWORD,
    IS_PRODUCTION,
    SEED_DEMO_USERS,
)
from app.models import User

logger = logging.getLogger(__name__)

MIN_PASSWORD_LENGTH = 12

DEMO_USERS = [
    ("Alice Johnson", "alice@example.com", "user", "free"),
    ("Bob Smith", "bob@example.com", "user", "premium"),
    ("Charlie Brown", "charlie@example.com", "moderator", "premium"),
    ("Diana Prince", "diana@example.com", "user", "free"),
    ("Eve Williams", "eve@example.com", "user", "premium"),
    ("Frank Castle", "frank@example.com", "user", "free"),
    ("Grace Hopper", "grace@example.com", "admin", "premium"),
    ("Henry Ford", "henry@example.com", "user", "free"),
    ("Isabel Diaz", "isabel@example.com", "user", "premium"),
    ("James Bond", "james@example.com", "user", "free"),
]


def seed_admin(db: Session) -> None:
    """Create the super admin from DEFAULT_ADMIN_EMAIL / DEFAULT_ADMIN_PASSWORD.

    Does nothing unless both are set, so a deployment that forgets them ends up
    with no admin rather than a publicly known one.
    """
    if not DEFAULT_ADMIN_EMAIL or not DEFAULT_ADMIN_PASSWORD:
        logger.info(
            "Admin seeding skipped: set DEFAULT_ADMIN_EMAIL and DEFAULT_ADMIN_PASSWORD to enable it."
        )
        return

    if len(DEFAULT_ADMIN_PASSWORD) < MIN_PASSWORD_LENGTH:
        raise RuntimeError(
            f"DEFAULT_ADMIN_PASSWORD must be at least {MIN_PASSWORD_LENGTH} characters."
        )

    if db.query(User).filter(User.email == DEFAULT_ADMIN_EMAIL).first():
        return

    db.add(
        User(
            name=DEFAULT_ADMIN_NAME,
            email=DEFAULT_ADMIN_EMAIL.lower(),
            password_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
            role="super_admin",
            status="active",
            subscription_type="premium",
            created_at=datetime.utcnow(),
        )
    )
    db.commit()
    logger.info("Seeded super admin %s", DEFAULT_ADMIN_EMAIL)


def seed_demo_users(db: Session) -> None:
    """Populate demo accounts for local dashboard work only.

    These share one password and include admin and moderator roles, so this
    refuses to run in production regardless of how the flag is set.
    """
    if not SEED_DEMO_USERS:
        return
    if IS_PRODUCTION:
        raise RuntimeError("SEED_DEMO_USERS must not be enabled in production.")

    rng = random.Random(42)
    for name, email, role, subscription in DEMO_USERS:
        if db.query(User).filter(User.email == email).first():
            continue
        db.add(
            User(
                name=name,
                email=email,
                password_hash=hash_password("Demo1234!"),
                role=role,
                status="active",
                subscription_type=subscription,
                created_at=datetime.utcnow() - timedelta(days=rng.randint(1, 180)),
            )
        )
    db.commit()
    logger.warning("Seeded %d demo users with a shared password.", len(DEMO_USERS))
