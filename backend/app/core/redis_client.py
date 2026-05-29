from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection pool — single pool shared across the entire process
# ---------------------------------------------------------------------------

_pool: aioredis.ConnectionPool = aioredis.ConnectionPool.from_url(
    str(settings.REDIS_URL),
    max_connections=20,
    decode_responses=True,   # all responses come back as str, not bytes
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True,
)


def get_redis_client() -> Redis:
    """
    Returns a Redis client backed by the shared connection pool.
    Does NOT open a new connection — reuses pool connections.

    Usage (direct):
        redis = get_redis_client()
        await redis.set("key", "value")

    Usage (FastAPI dependency):
        async def my_route(redis: Redis = Depends(get_redis_client)):
            ...
    """
    return aioredis.Redis(connection_pool=_pool)


async def close_redis() -> None:
    """
    Disconnects all pool connections on app shutdown.
    Call from the FastAPI lifespan shutdown hook.
    """
    await _pool.disconnect()
    logger.info("Redis connection pool closed")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def check_redis_health() -> dict[str, str]:
    """
    Pings Redis and returns a status dict.
    Returns {"status": "ok", "latency_ms": "..."} or {"status": "error", "detail": "..."}
    Never raises — safe to call from a /health endpoint.
    """
    import time
    try:
        redis = get_redis_client()
        start = time.monotonic()
        await redis.ping()
        latency_ms = round((time.monotonic() - start) * 1000, 2)
        return {"status": "ok", "latency_ms": str(latency_ms)}
    except Exception as exc:  # noqa: BLE001
        logger.error("Redis health check failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Generic cache helpers
# ---------------------------------------------------------------------------

async def cache_set(
    key: str,
    value: Any,
    ttl: int = settings.CACHE_TTL_SECONDS,
    redis: Redis | None = None,
) -> bool:
    """
    Serialises `value` to JSON and stores it in Redis with a TTL.

    Args:
        key:   Cache key (e.g. "stock:RELIANCE.NS:overview")
        value: Any JSON-serialisable Python object
        ttl:   Time-to-live in seconds (default: 24 hours from settings)
        redis: Optional Redis client — creates one if not provided

    Returns:
        True on success, False on failure (never raises)
    """
    _redis = redis or get_redis_client()
    try:
        serialised = json.dumps(value, default=str)
        await _redis.setex(key, ttl, serialised)
        logger.debug("Cache SET  key=%s  ttl=%ss", key, ttl)
        return True
    except (RedisError, TypeError) as exc:
        logger.error("Cache SET failed  key=%s  error=%s", key, exc)
        return False


async def cache_get(
    key: str,
    redis: Redis | None = None,
) -> Any | None:
    """
    Retrieves and deserialises a cached value from Redis.

    Returns:
        The deserialised Python object, or None if the key doesn't exist / expired.
    """
    _redis = redis or get_redis_client()
    try:
        raw = await _redis.get(key)
        if raw is None:
            logger.debug("Cache MISS key=%s", key)
            return None
        logger.debug("Cache HIT  key=%s", key)
        return json.loads(raw)
    except (RedisError, json.JSONDecodeError) as exc:
        logger.error("Cache GET failed  key=%s  error=%s", key, exc)
        return None


async def cache_delete(
    key: str,
    redis: Redis | None = None,
) -> bool:
    """
    Deletes a key from the cache.

    Returns:
        True if the key existed and was deleted, False otherwise.
    """
    _redis = redis or get_redis_client()
    try:
        deleted = await _redis.delete(key)
        logger.debug("Cache DEL  key=%s  existed=%s", key, bool(deleted))
        return bool(deleted)
    except RedisError as exc:
        logger.error("Cache DEL failed  key=%s  error=%s", key, exc)
        return False


async def cache_exists(
    key: str,
    redis: Redis | None = None,
) -> bool:
    """Returns True if a key exists in Redis (not expired)."""
    _redis = redis or get_redis_client()
    try:
        return bool(await _redis.exists(key))
    except RedisError:
        return False


# ---------------------------------------------------------------------------
# Cache key builders — central place for all key naming conventions
# ---------------------------------------------------------------------------

class CacheKeys:
    """
    Namespaced cache key builders.
    Keeps key naming consistent across the entire codebase.

    Usage:
        key = CacheKeys.stock_overview("RELIANCE.NS")
        # → "fve:stock:RELIANCE.NS:overview"
    """
    _PREFIX = "fve"  # fair-value-engine namespace

    @staticmethod
    def stock_overview(ticker: str) -> str:
        return f"fve:stock:{ticker}:overview"

    @staticmethod
    def stock_financials(ticker: str) -> str:
        return f"fve:stock:{ticker}:financials"

    @staticmethod
    def valuation_result(ticker: str) -> str:
        return f"fve:valuation:{ticker}:result"

    @staticmethod
    def valuation_job(job_id: str) -> str:
        return f"fve:valuation:job:{job_id}"

    @staticmethod
    def analyst_brief(ticker: str) -> str:
        return f"fve:ai:{ticker}:brief"

    @staticmethod
    def sentiment(ticker: str) -> str:
        return f"fve:ai:{ticker}:sentiment"

    @staticmethod
    def peer_comps(ticker: str) -> str:
        return f"fve:peers:{ticker}:comps"

    @staticmethod
    def rate_limit(user_id: str) -> str:
        return f"fve:ratelimit:{user_id}:daily"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

async def check_rate_limit(
    user_id: str,
    max_requests: int,
    redis: Redis | None = None,
) -> dict[str, int | bool]:
    """
    Sliding-window daily rate limiter using Redis INCR + EXPIRE.

    Args:
        user_id:      Unique user identifier (user UUID or IP address)
        max_requests: Maximum allowed requests per day
        redis:        Optional Redis client

    Returns:
        {
            "allowed": bool,       # True if request is within limit
            "count":   int,        # Requests made today (including this one)
            "limit":   int,        # Max allowed
            "remaining": int,      # Requests remaining today
        }
    """
    _redis = redis or get_redis_client()
    key = CacheKeys.rate_limit(user_id)

    try:
        pipe = _redis.pipeline()
        await pipe.incr(key)
        await pipe.ttl(key)
        count, ttl = await pipe.execute()

        # Set 24-hour expiry only on the first request of the day
        if ttl == -1:
            await _redis.expire(key, 60 * 60 * 24)

        allowed = count <= max_requests
        return {
            "allowed": allowed,
            "count": count,
            "limit": max_requests,
            "remaining": max(0, max_requests - count),
        }
    except RedisError as exc:
        # Fail open — allow the request if Redis is down
        logger.error("Rate limit check failed for user=%s  error=%s", user_id, exc)
        return {"allowed": True, "count": 0, "limit": max_requests, "remaining": max_requests}


# ---------------------------------------------------------------------------
# FastAPI dependency — yields a Redis client per request
# ---------------------------------------------------------------------------

async def get_redis() -> AsyncGenerator[Redis, None]:
    """
    FastAPI dependency that yields a Redis client.

    Usage:
        @router.get("/health")
        async def health(redis: Redis = Depends(get_redis)):
            return await check_redis_health()
    """
    redis = get_redis_client()
    try:
        yield redis
    finally:
        # Connection returns to pool automatically — no explicit close needed
        pass


# ---------------------------------------------------------------------------
# Embedded tests — run with: python -m pytest backend/app/core/redis_client.py -v
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402


def test_cache_keys_stock_overview():
    """CacheKeys.stock_overview should return correct namespaced key."""
    assert CacheKeys.stock_overview("RELIANCE.NS") == "fve:stock:RELIANCE.NS:overview"


def test_cache_keys_stock_financials():
    assert CacheKeys.stock_financials("TCS.NS") == "fve:stock:TCS.NS:financials"


def test_cache_keys_valuation_result():
    assert CacheKeys.valuation_result("HDFCBANK.NS") == "fve:valuation:HDFCBANK.NS:result"


def test_cache_keys_valuation_job():
    assert CacheKeys.valuation_job("job-123") == "fve:valuation:job:job-123"


def test_cache_keys_analyst_brief():
    assert CacheKeys.analyst_brief("INFY.NS") == "fve:ai:INFY.NS:brief"


def test_cache_keys_sentiment():
    assert CacheKeys.sentiment("ZOMATO.NS") == "fve:ai:ZOMATO.NS:sentiment"


def test_cache_keys_peer_comps():
    assert CacheKeys.peer_comps("RELIANCE.NS") == "fve:peers:RELIANCE.NS:comps"


def test_cache_keys_rate_limit():
    assert CacheKeys.rate_limit("user-abc") == "fve:ratelimit:user-abc:daily"


def test_cache_keys_are_unique():
    """All CacheKey builders should return distinct keys for the same ticker."""
    ticker = "RELIANCE.NS"
    keys = [
        CacheKeys.stock_overview(ticker),
        CacheKeys.stock_financials(ticker),
        CacheKeys.valuation_result(ticker),
        CacheKeys.analyst_brief(ticker),
        CacheKeys.sentiment(ticker),
        CacheKeys.peer_comps(ticker),
    ]
    assert len(keys) == len(set(keys)), "Cache keys must all be unique"


import pytest  # noqa: E402


@pytest.mark.asyncio
async def test_cache_set_success():
    """cache_set() should serialise value and call redis.setex."""
    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock(return_value=True)

    result = await cache_set("test:key", {"ticker": "RELIANCE"}, ttl=300, redis=mock_redis)

    assert result is True
    mock_redis.setex.assert_awaited_once()
    call_args = mock_redis.setex.call_args
    assert call_args[0][0] == "test:key"
    assert call_args[0][1] == 300
    assert '"ticker"' in call_args[0][2]


@pytest.mark.asyncio
async def test_cache_set_returns_false_on_redis_error():
    """cache_set() should return False (not raise) on RedisError."""
    from redis.exceptions import RedisError
    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock(side_effect=RedisError("connection lost"))

    result = await cache_set("test:key", {"data": 1}, redis=mock_redis)

    assert result is False


@pytest.mark.asyncio
async def test_cache_get_hit():
    """cache_get() should deserialise and return the cached value on a hit."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value='{"ticker": "RELIANCE", "price": 1400}')

    result = await cache_get("test:key", redis=mock_redis)

    assert result == {"ticker": "RELIANCE", "price": 1400}


@pytest.mark.asyncio
async def test_cache_get_miss():
    """cache_get() should return None on a cache miss."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)

    result = await cache_get("test:key", redis=mock_redis)

    assert result is None


@pytest.mark.asyncio
async def test_cache_get_returns_none_on_redis_error():
    """cache_get() should return None (not raise) on RedisError."""
    from redis.exceptions import RedisError
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(side_effect=RedisError("timeout"))

    result = await cache_get("test:key", redis=mock_redis)

    assert result is None


@pytest.mark.asyncio
async def test_cache_delete_existing_key():
    """cache_delete() should return True when the key existed."""
    mock_redis = AsyncMock()
    mock_redis.delete = AsyncMock(return_value=1)

    result = await cache_delete("test:key", redis=mock_redis)

    assert result is True


@pytest.mark.asyncio
async def test_cache_delete_missing_key():
    """cache_delete() should return False when the key did not exist."""
    mock_redis = AsyncMock()
    mock_redis.delete = AsyncMock(return_value=0)

    result = await cache_delete("test:key", redis=mock_redis)

    assert result is False


@pytest.mark.asyncio
async def test_cache_exists_true():
    """cache_exists() should return True when the key is present."""
    mock_redis = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=1)

    result = await cache_exists("test:key", redis=mock_redis)

    assert result is True


@pytest.mark.asyncio
async def test_cache_exists_false():
    """cache_exists() should return False when the key is absent."""
    mock_redis = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=0)

    result = await cache_exists("test:key", redis=mock_redis)

    assert result is False


@pytest.mark.asyncio
async def test_rate_limit_allows_first_request():
    """Rate limiter should allow the first request (count=1 < limit=10)."""
    mock_redis = AsyncMock()
    mock_pipe = AsyncMock()
    mock_pipe.incr = AsyncMock()
    mock_pipe.ttl = AsyncMock()
    mock_pipe.execute = AsyncMock(return_value=[1, -1])  # count=1, ttl=-1 (new key)
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)
    mock_redis.expire = AsyncMock()

    result = await check_rate_limit("user-abc", max_requests=10, redis=mock_redis)

    assert result["allowed"] is True
    assert result["count"] == 1
    assert result["remaining"] == 9
    mock_redis.expire.assert_awaited_once()  # TTL set on first request


@pytest.mark.asyncio
async def test_rate_limit_blocks_when_exceeded():
    """Rate limiter should block when count exceeds max_requests."""
    mock_redis = AsyncMock()
    mock_pipe = AsyncMock()
    mock_pipe.incr = AsyncMock()
    mock_pipe.ttl = AsyncMock()
    mock_pipe.execute = AsyncMock(return_value=[11, 3600])  # count=11, ttl already set
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)

    result = await check_rate_limit("user-abc", max_requests=10, redis=mock_redis)

    assert result["allowed"] is False
    assert result["count"] == 11
    assert result["remaining"] == 0


@pytest.mark.asyncio
async def test_rate_limit_fails_open_on_redis_error():
    """Rate limiter should allow requests (fail open) when Redis is unavailable."""
    from redis.exceptions import RedisError
    mock_redis = AsyncMock()
    mock_pipe = AsyncMock()
    mock_pipe.execute = AsyncMock(side_effect=RedisError("Redis down"))
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)

    result = await check_rate_limit("user-abc", max_requests=10, redis=mock_redis)

    # Must fail open — never block users because Redis is down
    assert result["allowed"] is True


@pytest.mark.asyncio
async def test_check_redis_health_ok():
    """check_redis_health() should return status ok when Redis responds."""
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)

    with patch("app.core.redis_client.get_redis_client", return_value=mock_redis):
        result = await check_redis_health()

    assert result["status"] == "ok"
    assert "latency_ms" in result


@pytest.mark.asyncio
async def test_check_redis_health_error():
    """check_redis_health() should return error dict when Redis is down."""
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(side_effect=Exception("Connection refused"))

    with patch("app.core.redis_client.get_redis_client", return_value=mock_redis):
        result = await check_redis_health()

    assert result["status"] == "error"
    assert "Connection refused" in result["detail"]


@pytest.mark.asyncio
async def test_check_redis_health_never_raises():
    """check_redis_health() must never propagate exceptions."""
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(side_effect=RuntimeError("Unexpected"))

    with patch("app.core.redis_client.get_redis_client", return_value=mock_redis):
        result = await check_redis_health()  # must not raise

    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_cache_set_and_get_roundtrip():
    """cache_set then cache_get should return the original value."""
    stored: dict[str, str] = {}

    async def fake_setex(key, ttl, value):
        stored[key] = value

    async def fake_get(key):
        return stored.get(key)

    mock_redis = AsyncMock()
    mock_redis.setex = fake_setex
    mock_redis.get = fake_get

    original = {"ticker": "TCS.NS", "fair_value": 3800, "scenario": "base"}
    await cache_set("roundtrip:key", original, redis=mock_redis)
    result = await cache_get("roundtrip:key", redis=mock_redis)

    assert result == original