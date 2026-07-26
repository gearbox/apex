"""Redis connection management."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)

_pool: aioredis.ConnectionPool | None = None


def _redacted_url(redis_url: str) -> str:
    """Strip userinfo (password) from a redis:// URL for safe logging."""
    parts = urlsplit(redis_url)
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def init_redis_pool(
    redis_url: str,
    *,
    socket_connect_timeout: float = 0.25,
    socket_timeout: float = 0.25,
) -> aioredis.ConnectionPool:
    """Create and store a global connection pool.

    F4 (issue #142): explicit socket timeouts are required because
    TokenRevocationService.is_revoked runs on every authenticated request —
    without a bound, a network fault stalls each request until the OS-level
    TCP timeout, turning the documented fail-open posture into fail-slow.
    `retry_on_timeout=False` so a timeout fails straight into the fail-open
    branch instead of silently retrying and doubling the stall.
    """
    global _pool
    _pool = aioredis.ConnectionPool.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=socket_connect_timeout,
        socket_timeout=socket_timeout,
        retry_on_timeout=False,
    )
    logger.info(
        "redis.pool_initialized",
        url=_redacted_url(redis_url),
        socket_connect_timeout=socket_connect_timeout,
        socket_timeout=socket_timeout,
    )
    return _pool


def get_redis_pool() -> aioredis.ConnectionPool:
    """Get the initialized pool. Raises RuntimeError if not initialized."""
    if _pool is None:
        raise RuntimeError("Redis pool not initialized. Call init_redis_pool() first.")
    return _pool


async def close_redis_pool() -> None:
    """Close the global pool."""
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
        logger.info("redis.pool_closed")


def get_redis_client() -> aioredis.Redis:
    """Get a Redis client from the pool."""
    return aioredis.Redis(connection_pool=get_redis_pool())
