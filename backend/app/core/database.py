from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Declarative base — all ORM models inherit from this
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.
    Import this in every models/*.py file:
        from app.core.database import Base
    """
    pass


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _build_engine() -> AsyncEngine:
    """
    Creates the async SQLAlchemy engine with connection pooling.
    Called once at module load — reused for the lifetime of the process.
    """
    engine = create_async_engine(
        settings.database_url_str,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_pre_ping=True,      # reconnect if a stale connection is detected
        pool_recycle=3600,       # recycle connections every 1 hour
        echo=settings.DB_ECHO,   # set DB_ECHO=True in .env to log all SQL
        future=True,
    )

    if settings.is_development:
        @event.listens_for(engine.sync_engine, "connect")
        def on_connect(dbapi_conn, connection_record):  # noqa: ARG001
            logger.debug("New DB connection established")

    return engine


engine: AsyncEngine = _build_engine()


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # keep ORM objects accessible after commit
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependency — use this in every route that needs DB access
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields an async database session and guarantees cleanup.

    Usage in a FastAPI route:
        @router.get("/stocks/{ticker}")
        async def get_stock(ticker: str, db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except SQLAlchemyError as exc:
            await session.rollback()
            logger.error("Database error — transaction rolled back: %s", exc)
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Startup / shutdown helpers — called from main.py lifespan
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """
    Creates all tables on startup (dev/test only).
    In production use Alembic migrations instead.
    """
    async with engine.begin() as conn:
        from app.models import ai_models, stock, valuation  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialised")


async def close_db() -> None:
    """Disposes the engine connection pool on app shutdown."""
    await engine.dispose()
    logger.info("Database connection pool closed")


# ---------------------------------------------------------------------------
# Health check — used by the /health endpoint
# ---------------------------------------------------------------------------

async def check_db_health() -> dict[str, str]:
    """
    Pings the database. Returns {"status": "ok"} or {"status": "error", "detail": "..."}.
    Never raises — safe to call from a health endpoint.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        logger.error("Database health check failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Utility: managed connection for seed scripts
# ---------------------------------------------------------------------------

async def with_connection(func, *args, **kwargs) -> None:
    """
    Runs an async function inside a managed DB connection.

    Usage:
        async def seed(conn):
            await conn.execute(...)
        asyncio.run(with_connection(seed))
    """
    async with engine.begin() as conn:
        await func(conn, *args, **kwargs)


# ---------------------------------------------------------------------------
# Embedded tests — run with: python -m pytest backend/app/core/database.py -v
# ---------------------------------------------------------------------------

if __name__ == "__main__" or True:
    # Guard so pytest can collect these even when the module is imported normally
    import pytest
    from unittest.mock import AsyncMock, MagicMock, patch

    # ---- Engine tests ------------------------------------------------------

    def test_engine_is_created():
        """Engine should be created at module import time."""
        assert engine is not None

    def test_engine_has_correct_pool_size():
        """Engine pool size should match settings.DB_POOL_SIZE."""
        assert engine.pool.size() == settings.DB_POOL_SIZE

    def test_async_session_local_bound_to_engine():
        """AsyncSessionLocal factory should be bound to the engine."""
        assert AsyncSessionLocal.kw["bind"] is engine
        assert AsyncSessionLocal.kw["expire_on_commit"] is False

    def test_base_is_declarative_base():
        """Base should be a valid SQLAlchemy DeclarativeBase subclass."""
        from sqlalchemy.orm import DeclarativeBase
        assert issubclass(Base, DeclarativeBase)

    def test_base_metadata_exists():
        """Base.metadata must exist for create_all / drop_all."""
        assert Base.metadata is not None

    # ---- get_db() tests ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_db_yields_session():
        """get_db() should yield an AsyncSession."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.database.AsyncSessionLocal", return_value=mock_cm):
            gen = get_db()
            session = await gen.__anext__()
            assert isinstance(session, AsyncMock)

    @pytest.mark.asyncio
    async def test_get_db_commits_on_success():
        """get_db() should commit after a successful request."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.database.AsyncSessionLocal", return_value=mock_cm):
            async for _ in get_db():
                pass

        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_db_rolls_back_on_error():
        """get_db() should rollback and re-raise on SQLAlchemyError."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.commit = AsyncMock(side_effect=SQLAlchemyError("DB error"))
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.database.AsyncSessionLocal", return_value=mock_cm):
            with pytest.raises(SQLAlchemyError, match="DB error"):
                async for _ in get_db():
                    pass

        mock_session.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_db_always_closes_session():
        """get_db() must close the session even when commit fails."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.commit = AsyncMock(side_effect=SQLAlchemyError("fail"))
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.database.AsyncSessionLocal", return_value=mock_cm):
            with pytest.raises(SQLAlchemyError):
                async for _ in get_db():
                    pass

        mock_session.close.assert_awaited_once()

    # ---- check_db_health() tests -------------------------------------------

    @pytest.mark.asyncio
    async def test_health_check_ok():
        """check_db_health() should return {"status": "ok"} when DB is up."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.database.engine") as mock_engine:
            mock_engine.connect = MagicMock(return_value=mock_cm)
            result = await check_db_health()

        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_health_check_error():
        """check_db_health() should return error dict when DB is down."""
        with patch("app.core.database.engine") as mock_engine:
            mock_engine.connect.side_effect = Exception("Connection refused")
            result = await check_db_health()

        assert result["status"] == "error"
        assert "Connection refused" in result["detail"]

    @pytest.mark.asyncio
    async def test_health_check_never_raises():
        """check_db_health() must never propagate exceptions."""
        with patch("app.core.database.engine") as mock_engine:
            mock_engine.connect.side_effect = RuntimeError("Unexpected")
            result = await check_db_health()  # must not raise

        assert result["status"] == "error"

    # ---- init_db() / close_db() tests --------------------------------------

    @pytest.mark.asyncio
    async def test_init_db_calls_create_all():
        """init_db() should call Base.metadata.create_all via run_sync."""
        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.database.engine") as mock_engine:
            mock_engine.begin = MagicMock(return_value=mock_cm)
            await init_db()

        mock_conn.run_sync.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_db_disposes_engine():
        """close_db() should dispose the engine connection pool."""
        with patch("app.core.database.engine") as mock_engine:
            mock_engine.dispose = AsyncMock()
            await close_db()

        mock_engine.dispose.assert_awaited_once()

    # ---- with_connection() tests -------------------------------------------

    @pytest.mark.asyncio
    async def test_with_connection_calls_func_with_conn():
        """with_connection() should call the function with a connection."""
        received = []

        async def capture(conn):
            received.append(conn)

        mock_conn = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.database.engine") as mock_engine:
            mock_engine.begin = MagicMock(return_value=mock_cm)
            await with_connection(capture)

        assert received[0] is mock_conn

    @pytest.mark.asyncio
    async def test_with_connection_passes_extra_args():
        """with_connection() should forward extra args and kwargs."""
        received = {}

        async def capture(conn, ticker, limit=10):
            received["ticker"] = ticker
            received["limit"] = limit

        mock_conn = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.database.engine") as mock_engine:
            mock_engine.begin = MagicMock(return_value=mock_cm)
            await with_connection(capture, "RELIANCE.NS", limit=50)

        assert received["ticker"] == "RELIANCE.NS"
        assert received["limit"] == 50