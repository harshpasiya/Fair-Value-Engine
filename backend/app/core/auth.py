from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.core.config import settings
from app.core.redis_client import CacheKeys, check_rate_limit, get_redis_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# bcrypt silently truncates at 72 bytes on some versions and raises on others.
# We truncate explicitly so behaviour is consistent across all bcrypt versions.
_BCRYPT_MAX_BYTES = 72


def _truncate(plain: str) -> str:
    """Truncates a password to 72 bytes (bcrypt hard limit)."""
    encoded = plain.encode("utf-8")
    return encoded[:_BCRYPT_MAX_BYTES].decode("utf-8", errors="ignore")


def hash_password(plain: str) -> str:
    """
    Returns a bcrypt hash of the given plain-text password.
    Automatically truncates to 72 bytes to satisfy bcrypt's hard limit.
    """
    return pwd_context.hash(_truncate(plain))


def verify_password(plain: str, hashed: str) -> bool:
    """
    Returns True if plain matches the bcrypt hash.
    Truncates plain to 72 bytes before comparison — must mirror hash_password().
    """
    return pwd_context.verify(_truncate(plain), hashed)


# ---------------------------------------------------------------------------
# Token schemas
# ---------------------------------------------------------------------------

class TokenPayload(BaseModel):
    """Decoded JWT payload."""
    sub: str                  # user ID (UUID string)
    email: str
    tier: str = "free"        # "free" | "registered"
    exp: datetime | None = None


class TokenResponse(BaseModel):
    """Returned to the client after successful login."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int           # seconds until expiry


# ---------------------------------------------------------------------------
# JWT creation
# ---------------------------------------------------------------------------

def create_access_token(
    user_id: str,
    email: str,
    tier: str = "free",
    expires_delta: timedelta | None = None,
) -> TokenResponse:
    """
    Creates a signed JWT access token.

    Args:
        user_id:       UUID string of the authenticated user
        email:         User's email address (included in payload)
        tier:          "free" or "registered" — controls rate limits
        expires_delta: Override default expiry (defaults to settings value)

    Returns:
        TokenResponse with the signed token and expiry info
    """
    delta = expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(tz=timezone.utc) + delta

    payload: dict[str, Any] = {
        "sub": user_id,
        "email": email,
        "tier": tier,
        "exp": expire,
        "iat": datetime.now(tz=timezone.utc),
    }

    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    logger.debug("Access token created for user=%s tier=%s", user_id, tier)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=int(delta.total_seconds()),
    )


# ---------------------------------------------------------------------------
# JWT verification
# ---------------------------------------------------------------------------

def decode_token(token: str) -> TokenPayload:
    """
    Decodes and validates a JWT token.

    Raises:
        HTTPException 401 if the token is invalid, expired, or malformed.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        user_id: str = payload.get("sub", "")
        email: str = payload.get("email", "")
        tier: str = payload.get("tier", "free")

        if not user_id or not email:
            logger.warning("Token missing sub or email fields")
            raise credentials_exception

        return TokenPayload(sub=user_id, email=email, tier=tier)

    except JWTError as exc:
        logger.warning("JWT decode failed: %s", exc)
        raise credentials_exception from exc


# ---------------------------------------------------------------------------
# FastAPI security scheme
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> TokenPayload:
    """
    FastAPI dependency — extracts and validates the Bearer token.

    Usage:
        @router.get("/protected")
        async def protected(user: TokenPayload = Depends(get_current_user)):
            return {"user_id": user.sub}

    Raises:
        HTTPException 401 if no token or invalid token.
    """
    return decode_token(credentials.credentials)


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> TokenPayload | None:
    """
    FastAPI dependency — returns TokenPayload if a valid token is present,
    or None for unauthenticated requests (public routes with optional auth).

    Usage:
        @router.get("/public")
        async def public(user: TokenPayload | None = Depends(get_current_user_optional)):
            if user:
                # personalise response
            ...
    """
    if credentials is None:
        return None
    try:
        return decode_token(credentials.credentials)
    except HTTPException:
        return None


# ---------------------------------------------------------------------------
# Rate-limited user dependency
# ---------------------------------------------------------------------------

async def get_rate_limited_user(
    user: TokenPayload = Depends(get_current_user),
) -> TokenPayload:
    """
    FastAPI dependency — validates token AND enforces daily analysis rate limit.

    Free tier:       settings.FREE_TIER_DAILY_ANALYSES  (default: 10/day)
    Registered tier: settings.REGISTERED_TIER_DAILY_ANALYSES (default: unlimited)

    Raises:
        HTTPException 401 if unauthenticated.
        HTTPException 429 if the user has exceeded their daily limit.

    Usage:
        @router.post("/valuation/run")
        async def run_valuation(user: TokenPayload = Depends(get_rate_limited_user)):
            ...
    """
    max_requests = (
        settings.FREE_TIER_DAILY_ANALYSES
        if user.tier == "free"
        else settings.REGISTERED_TIER_DAILY_ANALYSES
    )

    redis = get_redis_client()
    result = await check_rate_limit(
        user_id=user.sub,
        max_requests=max_requests,
        redis=redis,
    )

    if not result["allowed"]:
        logger.warning(
            "Rate limit exceeded user=%s tier=%s count=%s limit=%s",
            user.sub, user.tier, result["count"], result["limit"],
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "message": "Daily analysis limit reached. Upgrade to registered tier for unlimited access.",
                "limit": result["limit"],
                "count": result["count"],
                "tier": user.tier,
            },
        )

    logger.debug(
        "Rate limit OK user=%s remaining=%s", user.sub, result["remaining"]
    )
    return user


# ---------------------------------------------------------------------------
# Utility: generate a secure secret key (helper for devs)
# ---------------------------------------------------------------------------

def generate_secret_key() -> str:
    """
    Generates a cryptographically secure 64-character hex secret key.
    Run once to generate your SECRET_KEY for .env:
        python -c "from app.core.auth import generate_secret_key; print(generate_secret_key())"
    """
    import secrets
    return secrets.token_hex(32)


# ---------------------------------------------------------------------------
# Embedded tests — run with: python -m pytest backend/app/core/auth.py -v
# ---------------------------------------------------------------------------

import pytest  # noqa: E402
from datetime import timedelta  # noqa: E402 (already imported above, explicit for clarity)
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402


# ---- Password hashing -------------------------------------------------------

def test_hash_password_returns_hash():
    """hash_password() should return a non-empty bcrypt hash."""
    hashed = hash_password("mysecretpassword")
    assert hashed
    assert hashed != "mysecretpassword"
    assert hashed.startswith("$2b$")


def test_verify_password_correct():
    """verify_password() should return True for matching plain/hash pair."""
    plain = "correcthorse"
    hashed = hash_password(plain)
    assert verify_password(plain, hashed) is True


def test_verify_password_wrong():
    """verify_password() should return False for non-matching password."""
    hashed = hash_password("original")
    assert verify_password("wrongpassword", hashed) is False


def test_hash_is_unique_per_call():
    """Two hashes of the same password should differ (bcrypt salting)."""
    h1 = hash_password("samepassword")
    h2 = hash_password("samepassword")
    assert h1 != h2


def test_hash_password_truncates_at_72_bytes():
    """Passwords longer than 72 bytes should be truncated, not raise."""
    long_password = "a" * 100
    hashed = hash_password(long_password)
    assert hashed.startswith("$2b$")
    assert verify_password(long_password, hashed) is True


def test_verify_password_truncation_consistency():
    """Characters beyond byte 72 must not affect hash or verify result."""
    base = "x" * 72
    longer = base + "EXTRA_CHARS_THAT_EXCEED_LIMIT"
    hashed = hash_password(base)
    assert verify_password(longer, hashed) is True


# ---- Token creation ---------------------------------------------------------

def test_create_access_token_returns_token_response():
    """create_access_token() should return a valid TokenResponse."""
    response = create_access_token(
        user_id="user-123",
        email="test@example.com",
        tier="free",
    )
    assert response.access_token
    assert response.token_type == "bearer"
    assert response.expires_in > 0


def test_create_access_token_with_custom_expiry():
    """create_access_token() should respect custom expires_delta."""
    response = create_access_token(
        user_id="user-123",
        email="test@example.com",
        expires_delta=timedelta(minutes=30),
    )
    assert response.expires_in == 30 * 60


def test_create_access_token_registered_tier():
    """Token payload should encode tier correctly."""
    response = create_access_token(
        user_id="user-456",
        email="pro@example.com",
        tier="registered",
    )
    # Decode and verify
    payload = decode_token(response.access_token)
    assert payload.tier == "registered"


# ---- Token decoding ---------------------------------------------------------

def test_decode_token_valid():
    """decode_token() should return correct TokenPayload for a valid token."""
    response = create_access_token(
        user_id="user-789",
        email="decode@example.com",
        tier="free",
    )
    payload = decode_token(response.access_token)
    assert payload.sub == "user-789"
    assert payload.email == "decode@example.com"
    assert payload.tier == "free"


def test_decode_token_invalid_raises_401():
    """decode_token() should raise HTTP 401 for a garbage token."""
    with pytest.raises(HTTPException) as exc_info:
        decode_token("not.a.valid.token")
    assert exc_info.value.status_code == 401


def test_decode_token_tampered_raises_401():
    """decode_token() should raise HTTP 401 for a tampered token."""
    response = create_access_token("user-1", "test@x.com")
    tampered = response.access_token[:-5] + "XXXXX"
    with pytest.raises(HTTPException) as exc_info:
        decode_token(tampered)
    assert exc_info.value.status_code == 401


def test_decode_token_expired_raises_401():
    """decode_token() should raise HTTP 401 for an expired token."""
    response = create_access_token(
        user_id="user-1",
        email="exp@example.com",
        expires_delta=timedelta(seconds=-1),  # already expired
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_token(response.access_token)
    assert exc_info.value.status_code == 401


def test_decode_token_wrong_secret_raises_401():
    """decode_token() should reject tokens signed with a different secret."""
    wrong_token = jwt.encode(
        {"sub": "user-1", "email": "x@x.com", "tier": "free"},
        "completely_different_secret_key_here",
        algorithm=settings.ALGORITHM,
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_token(wrong_token)
    assert exc_info.value.status_code == 401


def test_decode_token_missing_sub_raises_401():
    """decode_token() should raise 401 if 'sub' field is missing."""
    bad_token = jwt.encode(
        {"email": "no_sub@example.com"},  # no 'sub'
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_token(bad_token)
    assert exc_info.value.status_code == 401


# ---- TokenPayload model -----------------------------------------------------

def test_token_payload_defaults():
    """TokenPayload should default tier to 'free'."""
    payload = TokenPayload(sub="user-1", email="a@b.com")
    assert payload.tier == "free"
    assert payload.exp is None


def test_token_response_defaults():
    """TokenResponse should default token_type to 'bearer'."""
    resp = TokenResponse(access_token="abc", expires_in=3600)
    assert resp.token_type == "bearer"


# ---- Rate limited user dependency -------------------------------------------

@pytest.mark.asyncio
async def test_get_rate_limited_user_allows_within_limit():
    """get_rate_limited_user() should return user when within limit."""
    user = TokenPayload(sub="user-1", email="a@b.com", tier="free")

    with patch("app.core.auth.check_rate_limit", new=AsyncMock(return_value={
        "allowed": True, "count": 3, "limit": 10, "remaining": 7,
    })):
        with patch("app.core.auth.get_redis_client", return_value=AsyncMock()):
            result = await get_rate_limited_user(user=user)

    assert result.sub == "user-1"


@pytest.mark.asyncio
async def test_get_rate_limited_user_blocks_when_exceeded():
    """get_rate_limited_user() should raise HTTP 429 when limit exceeded."""
    user = TokenPayload(sub="user-1", email="a@b.com", tier="free")

    with patch("app.core.auth.check_rate_limit", new=AsyncMock(return_value={
        "allowed": False, "count": 11, "limit": 10, "remaining": 0,
    })):
        with patch("app.core.auth.get_redis_client", return_value=AsyncMock()):
            with pytest.raises(HTTPException) as exc_info:
                await get_rate_limited_user(user=user)

    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_registered_tier_uses_higher_limit():
    """Registered tier users should have higher rate limit applied."""
    user = TokenPayload(sub="user-2", email="pro@b.com", tier="registered")
    captured = {}

    async def capture_limit(user_id, max_requests, redis):
        captured["max_requests"] = max_requests
        return {"allowed": True, "count": 1, "limit": max_requests, "remaining": max_requests - 1}

    with patch("app.core.auth.check_rate_limit", new=capture_limit):
        with patch("app.core.auth.get_redis_client", return_value=AsyncMock()):
            await get_rate_limited_user(user=user)

    assert captured["max_requests"] == settings.REGISTERED_TIER_DAILY_ANALYSES


# ---- Utility ----------------------------------------------------------------

def test_generate_secret_key_length():
    """generate_secret_key() should return a 64-char hex string."""
    key = generate_secret_key()
    assert len(key) == 64


def test_generate_secret_key_is_unique():
    """generate_secret_key() should return different values each call."""
    assert generate_secret_key() != generate_secret_key()


def test_full_auth_roundtrip():
    """Full roundtrip: hash password → create token → decode token."""
    # Hash password
    plain = "SecurePass123"
    hashed = hash_password(plain)
    assert verify_password(plain, hashed)

    # Create token
    token_resp = create_access_token(
        user_id="user-roundtrip",
        email="roundtrip@test.com",
        tier="registered",
    )

    # Decode token
    payload = decode_token(token_resp.access_token)
    assert payload.sub == "user-roundtrip"
    assert payload.email == "roundtrip@test.com"
    assert payload.tier == "registered"