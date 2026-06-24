import os

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_file() -> str | None:
    if os.getenv("ENVIRONMENT", "development") == "production":
        return None
    return ".env"


class Settings(BaseSettings):
    app_name: str = "Football AI Predictor API"
    environment: str = "development"
    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_minutes: int = 60 * 24 * 7
    football_provider: str = "api-football"
    football_api_base_url: str
    football_api_key: str
    model_dir: str = "app/ml/models"
    settings_encryption_key: str = "change-me-to-32-bytes-minimum"
    admin_cookie_secure: bool = False
    admin_cookie_samesite: str = "lax"
    cors_origins: str = "http://localhost:3000"
    port: int = 8000

    model_config = SettingsConfigDict(env_file=_env_file(), extra="ignore")

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        if value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql+psycopg://", 1)
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+psycopg://", 1)
        return value

    @model_validator(mode="after")
    def validate_production_urls(self) -> "Settings":
        if self.environment != "production":
            return self

        docker_hosts = ("@postgres:", "@postgres/", "@redis:", "@redis/")
        if any(host in self.database_url for host in docker_hosts[:2]):
            raise ValueError(
                "DATABASE_URL uses Docker hostname 'postgres'. On Render/Railway, set "
                "DATABASE_URL to your managed Postgres Internal Database URL."
            )
        if any(host in self.redis_url for host in docker_hosts[2:]):
            raise ValueError(
                "REDIS_URL uses Docker hostname 'redis'. On Render, link the Redis "
                "service or set REDIS_URL to your managed Redis connection string."
            )
        return self


settings = Settings()  # type: ignore[call-arg]
