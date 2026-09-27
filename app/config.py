from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Secure Enterprise BI Copilot"
    frontend_origin: str = "http://localhost:3000"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    max_query_rows: int = 500
    query_timeout_seconds: int = 20
    database_url: str = "postgresql+psycopg://secure_bi:secure_bi@postgres:5432/secure_bi_demo"
    redis_url: str | None = "redis://redis:6379/0"
    rate_limit_per_minute: int = 10
    security_lab_enabled: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
