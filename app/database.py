from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.config import DATABASE_URL

if "sqlite" in DATABASE_URL:
    connect_args: dict = {"check_same_thread": False}
    engine_kwargs: dict = {}
else:
    # Without a connect timeout a bad or unreachable database makes startup
    # hang rather than fail, which looks from outside like a server that
    # accepts connections and never answers.
    connect_args = {"connect_timeout": 10}
    # Hosted databases drop idle connections; without this the first query
    # after an idle period fails instead of reconnecting.
    engine_kwargs = {"pool_pre_ping": True, "pool_recycle": 300}

engine = create_engine(DATABASE_URL, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
