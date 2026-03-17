"""Redis connection management."""

from __future__ import annotations

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)

_pool: aioredis.ConnectionPool | None = None


def init_redis_pool(redis_url: str) -> aioredis.ConnectionPool:
    """Create and store a global connection pool."""
    global _pool
    _pool = aioredis.ConnectionPool.from_url(redis_url, decode_responses=True)
    logger.info("redis.pool_initialized", url=redis_url)
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
