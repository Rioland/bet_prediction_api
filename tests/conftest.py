import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("FOOTBALL_API_BASE_URL", "https://example.com")
os.environ.setdefault("FOOTBALL_API_KEY", "test-key")
os.environ.setdefault("SETTINGS_ENCRYPTION_KEY", "test-encryption-key-32-chars-min")

import pyotp
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.db.session as db_module
from app.core.security import hash_password
from app.db.session import Base, get_db
from app.main import app
from app.models.entities import User, UserRole

TEST_PASSWORD = "password123"

TEST_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(bind=TEST_ENGINE, autoflush=False, autocommit=False)

db_module.engine = TEST_ENGINE
db_module.SessionLocal = TestSessionLocal


@pytest.fixture()
def db_session() -> Session:
    Base.metadata.create_all(bind=TEST_ENGINE)
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture()
def client(db_session: Session) -> TestClient:
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def admin_user(db_session: Session) -> User:
    user = User(
        name="Test Admin",
        email="admin@test.com",
        password_hash=hash_password(TEST_PASSWORD),
        role=UserRole.ADMIN,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture()
def admin_user_2fa(db_session: Session) -> tuple[User, str]:
    secret = pyotp.random_base32()
    user = User(
        name="2FA Admin",
        email="2fa-admin@test.com",
        password_hash=hash_password(TEST_PASSWORD),
        role=UserRole.ADMIN,
        two_factor_enabled=True,
        two_factor_secret=secret,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user, secret


def admin_login(
    client: TestClient,
    email: str,
    password: str,
    otp_code: str | None = None,
) -> TestClient:
    payload: dict[str, str] = {"email": email, "password": TEST_PASSWORD}
    if otp_code:
        payload["otp_code"] = otp_code
    response = client.post("/admin/auth/login", json=payload)
    assert response.status_code == 200, response.text
    return client
