from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Secure Enterprise BI Copilot"
    frontend_origin: str = "http://localhost:3000"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5-mini"
    max_query_rows: int = 500
    query_timeout_seconds: int = 20
    database_url: str = "postgresql+psycopg://secure_bi:secure_bi@postgres:5432/secure_bi_demo"
    redis_url: str | None = "redis://redis:6379/0"
    # HTTP protection and LLM spend protection are intentionally independent.
    api_rate_limit_per_minute: int = 60
    llm_rate_limit_per_minute: int = 10
    # This is a per-worker traffic shaper. Redis remains the authoritative
    # cross-worker quota, scoped by tenant, user, and model.
    llm_worker_requests_per_second: float = 3.0
    llm_worker_max_burst: int = 3
    # Keep SDK retries bounded so a single DAG node cannot create an
    # unbounded number of provider calls after it has passed the Redis gate.
    openai_max_retries: int = 1
    security_lab_enabled: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
