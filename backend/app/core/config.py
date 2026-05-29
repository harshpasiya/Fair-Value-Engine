from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the Fair Value Engine.
    All values are read from environment variables (or .env file).
    Never hardcode secrets — always use this class.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------
    APP_NAME: str = "Fair Value Engine"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    API_PREFIX: str = "/api"
    ALLOWED_ORIGINS: list[str] = Field(
        default=["http://localhost:3000"],
        description="CORS allowed origins. Add production frontend URL here.",
    )

    # ------------------------------------------------------------------
    # Security / Auth
    # ------------------------------------------------------------------
    SECRET_KEY: str = Field(
        ...,
        description=(
            "JWT signing secret. "
            "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
        ),
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    ALGORITHM: str = "HS256"

    # ------------------------------------------------------------------
    # Database (PostgreSQL)
    # ------------------------------------------------------------------
    DATABASE_URL: PostgresDsn = Field(
        ...,
        description="Async PostgreSQL URL. Format: postgresql+asyncpg://user:pass@host:5432/dbname",
    )
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_ECHO: bool = False  # Set True to log all SQL queries (dev only)

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    REDIS_URL: RedisDsn = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for caching, Celery broker, and rate limiting.",
    )
    CACHE_TTL_SECONDS: int = 60 * 60 * 24  # 24 hours default cache TTL

    # ------------------------------------------------------------------
    # Celery
    # ------------------------------------------------------------------
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_TASK_TIMEOUT: int = 120  # seconds before a task is killed

    # ------------------------------------------------------------------
    # Claude / Anthropic
    # ------------------------------------------------------------------
    ANTHROPIC_API_KEY: str = Field(
        ...,
        description="Get from https://console.anthropic.com/",
    )
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
    CLAUDE_MAX_TOKENS: int = 1500
    CLAUDE_CACHE_TTL: int = 60 * 60 * 24  # 24 hrs — critical to control API costs

    # ------------------------------------------------------------------
    # External data APIs
    # ------------------------------------------------------------------
    NEWS_API_KEY: str = Field(
        default="",
        description="From https://newsapi.org — free tier: 100 requests/day",
    )
    YFINANCE_RATE_LIMIT_PER_HOUR: int = 2000

    # ------------------------------------------------------------------
    # Rate limiting (requests per user per day)
    # ------------------------------------------------------------------
    FREE_TIER_DAILY_ANALYSES: int = 10
    REGISTERED_TIER_DAILY_ANALYSES: int = 999999  # effectively unlimited

    # ------------------------------------------------------------------
    # ChromaDB (vector store for RAG)
    # ------------------------------------------------------------------
    CHROMA_PERSIST_DIR: str = "./data/chromadb"
    CHROMA_COLLECTION_NAME: str = "annual_reports"

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------
    MONTE_CARLO_SIMULATIONS: int = 10_000
    MONTE_CARLO_WACC_STD: float = 0.015   # ±1.5% std dev on WACC
    MONTE_CARLO_GROWTH_STD: float = 0.02  # ±2% std dev on growth rate

    # ------------------------------------------------------------------
    # Sentry (error monitoring)
    # ------------------------------------------------------------------
    SENTRY_DSN: str = Field(
        default="",
        description="Leave empty to disable Sentry. Get from https://sentry.io",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}, got '{v}'")
        return v

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError(
                "SECRET_KEY must be at least 32 characters. "
                "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return v

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def database_url_str(self) -> str:
        """Returns DATABASE_URL as a plain string (SQLAlchemy needs this)."""
        return str(self.DATABASE_URL)


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Use as a FastAPI dependency: settings = Depends(get_settings)
    Or import directly:          from app.core.config import settings
    """
    return Settings()


# Module-level singleton — import this everywhere in the app
settings = get_settings()


# ---------------------------------------------------------------------------
# Embedded tests — run with: python -m pytest backend/app/core/config.py -v
# ---------------------------------------------------------------------------

def test_settings_loads_successfully():
    """Settings should load without errors given valid env vars."""
    assert settings is not None
    assert settings.APP_NAME == "Fair Value Engine"


def test_default_environment_is_development():
    """Default ENVIRONMENT should be 'development'."""
    assert settings.ENVIRONMENT == "development"


def test_default_algorithm_is_hs256():
    """Default JWT algorithm should be HS256."""
    assert settings.ALGORITHM == "HS256"


def test_default_claude_model():
    """Default Claude model should be set."""
    assert "claude" in settings.CLAUDE_MODEL.lower()


def test_monte_carlo_defaults():
    """Monte Carlo defaults should be sensible values."""
    assert settings.MONTE_CARLO_SIMULATIONS == 10_000
    assert 0 < settings.MONTE_CARLO_WACC_STD < 0.1
    assert 0 < settings.MONTE_CARLO_GROWTH_STD < 0.1


def test_free_tier_daily_limit():
    """Free tier daily analyses should be a positive integer."""
    assert settings.FREE_TIER_DAILY_ANALYSES > 0


def test_secret_key_too_short_raises():
    """SECRET_KEY shorter than 32 chars should raise ValidationError."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="SECRET_KEY must be at least 32"):
        Settings(
            SECRET_KEY="short",
            DATABASE_URL="postgresql+asyncpg://u:p@localhost:5432/db",
            ANTHROPIC_API_KEY="sk-ant-test",
        )


def test_invalid_environment_raises():
    """An invalid ENVIRONMENT value should raise ValidationError."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Settings(
            SECRET_KEY="a" * 32,
            DATABASE_URL="postgresql+asyncpg://u:p@localhost:5432/db",
            ANTHROPIC_API_KEY="sk-ant-test",
            ENVIRONMENT="invalid_env",  # type: ignore[arg-type]
        )


def test_valid_environments_accepted():
    """All three valid ENVIRONMENT values should be accepted."""
    for env in ("development", "staging", "production"):
        s = Settings(
            SECRET_KEY="a" * 32,
            DATABASE_URL="postgresql+asyncpg://u:p@localhost:5432/db",
            ANTHROPIC_API_KEY="sk-ant-test",
            ENVIRONMENT=env,  # type: ignore[arg-type]
        )
        assert s.ENVIRONMENT == env


def test_is_production_false_in_development():
    """is_production should be False when ENVIRONMENT is development."""
    assert settings.is_production is False


def test_is_development_true():
    """is_development should be True when ENVIRONMENT is development."""
    assert settings.is_development is True


def test_database_url_str_is_string():
    """database_url_str should return a plain string starting with postgresql."""
    assert isinstance(settings.database_url_str, str)
    assert settings.database_url_str.startswith("postgresql")


def test_is_production_true_when_set():
    """is_production should be True when ENVIRONMENT is production."""
    s = Settings(
        SECRET_KEY="a" * 32,
        DATABASE_URL="postgresql+asyncpg://u:p@localhost:5432/db",
        ANTHROPIC_API_KEY="sk-ant-test",
        ENVIRONMENT="production",  # type: ignore[arg-type]
    )
    assert s.is_production is True
    assert s.is_development is False


def test_get_settings_returns_same_instance():
    """get_settings() should always return the same cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2