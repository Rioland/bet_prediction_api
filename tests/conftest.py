import os

os.environ.setdefault("PREDICTOR_DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("JWT_SECRET", "test-secret-for-pytest-only")
# Otherwise the app lifespan fetches live fixtures and writes them into the
# test database, making every count non-deterministic.
os.environ["FIXTURE_REFRESH_ENABLED"] = "0"
os.environ.setdefault("OPAY_MERCHANT_ID", "256612345678901")
os.environ.setdefault("OPAY_PUBLIC_KEY", "OPAYPUB-test")
os.environ.setdefault("OPAY_SECRET_KEY", "OPAYPRV-route-tests")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.database as db_module
from app.database import Base
from app.rate_limit import limiter

# Rate limits are asserted in test_account.py; leaving them on everywhere would
# make unrelated tests fail depending on the order they run in.
limiter.enabled = False

# SQLite by default, for speed. Production runs PostgreSQL, and the two disagree
# on things tests must catch - a BOOLEAN column with DEFAULT 0 is valid SQLite
# and a hard error in PostgreSQL. Point TEST_DATABASE_URL at a disposable
# Postgres database to run the suite there:
#
#   TEST_DATABASE_URL=postgresql://localhost/predictor_test python -m pytest
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "").strip()

if TEST_DATABASE_URL:
    from app.config import _normalise_database_url

    TEST_ENGINE = create_engine(_normalise_database_url(TEST_DATABASE_URL), pool_pre_ping=True)
else:
    TEST_ENGINE = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
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
def trained_models(tmp_path, monkeypatch):
    """Train real models into a temp dir so prediction paths can be exercised."""
    import app.ml.train as train_module
    import app.services.prediction_service as prediction_service
    from app.ml.features import build_dataset
    from tests.test_training import _simulate_league

    monkeypatch.setattr(train_module, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(prediction_service, "MODEL_DIR", str(tmp_path))
    prediction_service.clear_model_cache()
    train_module.train_models(
        build_dataset(_simulate_league()), targets=["match_winner", "btts", "over_under_2_5"]
    )
    yield
    prediction_service.clear_model_cache()
