from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    cors_origins: str = "http://localhost:3000"
    port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        if value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql+psycopg://", 1)
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+psycopg://", 1)
        return value


settings = Settings()  # type: ignore[call-arg]
