"""Redis connection management.

Each environment (development, staging, production) runs its own Redis
container. This is a deployment invariant, not a coincidence: keys and
pub/sub channels are deliberately unnamespaced, so pointing two environments
at one Redis would cross-deliver SSE broadcasts and ops events and let one
environment's workers claim the other's leases. Do not consolidate.

Two pools, deliberately. The default pool serves short-lived operations -
token revocation on every authenticated request, worker lease renewals, SSE
ticket lookups - and is tuned for fail-fast (50ms socket timeout). The SSE
pool serves EventBus.subscribe, which checks out one connection per connected
client and holds it for the life of the stream.

They are separate because redis-py raises ConnectionError on pool exhaustion
rather than blocking, and ConnectionError is a RedisError: a shared pool
exhausted by SSE clients would push TokenRevocationService into its fail-open
branch, silently accepting revoked JWTs, and push LeaderLease into its
fail-closed branch, stalling every background worker. Long-lived subscribers
must never be able to starve the auth path.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)

_pool: aioredis.ConnectionPool | None = None
_pool_url: str | None = None
_sse_pool: aioredis.ConnectionPool | None = None
_sse_pool_url: str | None = None


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
    socket_timeout: float = 0.05,
    health_check_interval: float = 30.0,
    max_connections: int = 50,
) -> aioredis.ConnectionPool:
    """Create and store the global short-lived-operation connection pool.

    F4/G3b (issue #142): explicit socket timeouts are required because
    TokenRevocationService.is_revoked/get_current_epoch run on every
    authenticated request/refresh — without a bound, a network fault
    stalls each request until the OS-level TCP timeout, turning the
    documented fail-open posture into fail-slow. `retry_on_timeout=False`
    so a timeout fails straight into the fail-open branch instead of
    silently retrying and doubling the stall.

    These defaults exist only as a fallback for callers that don't pass
    explicit values — production call sites should always pass
    `Settings.redis_socket_connect_timeout_seconds`/
    `redis_socket_timeout_seconds`/`redis_health_check_interval_seconds`/
    `redis_max_connections` explicitly, so there is exactly one source of
    truth for these numbers. `health_check_interval` makes redis-py
    proactively PING idle connections so a connection gone stale while
    unused (rather than mid-request) is caught and replaced before it can
    produce a spurious is_revoked failure.

    This pool must never carry a long-lived-per-client consumer (SSE) — see
    `init_sse_redis_pool` and the module docstring for why.

    Calling this again with the same URL is a no-op that returns the
    existing pool. Calling it with a *different* URL raises — reinitializing
    would silently orphan the previous pool's connections rather than
    closing them.

    Raises:
        RuntimeError: If a pool already exists for a different URL. Call
            `close_redis_pool()` first to replace it intentionally.
    """
    global _pool, _pool_url
    if _pool is not None:
        if _pool_url == redis_url:
            logger.debug("redis.pool_already_initialized", url=_redacted_url(redis_url))
            return _pool
        raise RuntimeError(
            "Redis pool already initialized with a different URL; call close_redis_pool() first."
        )

    _pool = aioredis.ConnectionPool.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=socket_connect_timeout,
        socket_timeout=socket_timeout,
        retry_on_timeout=False,
        health_check_interval=health_check_interval,
        max_connections=max_connections,
    )
    _pool_url = redis_url
    logger.info(
        "redis.pool_initialized",
        url=_redacted_url(redis_url),
        socket_connect_timeout=socket_connect_timeout,
        socket_timeout=socket_timeout,
        health_check_interval=health_check_interval,
        max_connections=max_connections,
    )
    return _pool


def get_redis_pool() -> aioredis.ConnectionPool:
    """Get the initialized pool. Raises RuntimeError if not initialized."""
    if _pool is None:
        raise RuntimeError("Redis pool not initialized. Call init_redis_pool() first.")
    return _pool


def get_redis_client() -> aioredis.Redis:
    """Get a Redis client from the short-lived-operation pool.

    Never hold this client across an `await` boundary that can span more
    than one request — long-lived consumers (SSE) must use
    `get_sse_redis_client()` instead, or they can exhaust this pool and push
    TokenRevocationService/LeaderLease into their failure postures. See the
    module docstring.
    """
    return aioredis.Redis(connection_pool=get_redis_pool())


def init_sse_redis_pool(
    redis_url: str,
    *,
    socket_connect_timeout: float = 0.25,
    socket_timeout: float = 0.05,
    health_check_interval: float = 30.0,
    max_connections: int = 500,
) -> aioredis.ConnectionPool:
    """Create and store the global SSE connection pool.

    Dedicated pool for `EventBus.subscribe`, which checks out one connection
    per connected client and holds it for the life of the stream (minutes to
    hours). Isolated from `init_redis_pool`'s short-lived-operation pool so
    SSE concurrency can never exhaust the auth hot path — see the module
    docstring for the specific failure cascade this prevents.

    Same idempotence contract as `init_redis_pool`: a repeat call with the
    same URL is a no-op; a different URL raises.

    Raises:
        RuntimeError: If an SSE pool already exists for a different URL.
    """
    global _sse_pool, _sse_pool_url
    if _sse_pool is not None:
        if _sse_pool_url == redis_url:
            logger.debug("redis.sse_pool_already_initialized", url=_redacted_url(redis_url))
            return _sse_pool
        raise RuntimeError(
            "SSE Redis pool already initialized with a different URL; "
            "call close_redis_pool() first."
        )

    _sse_pool = aioredis.ConnectionPool.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=socket_connect_timeout,
        socket_timeout=socket_timeout,
        retry_on_timeout=False,
        health_check_interval=health_check_interval,
        max_connections=max_connections,
    )
    _sse_pool_url = redis_url
    logger.info(
        "redis.sse_pool_initialized",
        url=_redacted_url(redis_url),
        socket_connect_timeout=socket_connect_timeout,
        socket_timeout=socket_timeout,
        health_check_interval=health_check_interval,
        max_connections=max_connections,
    )
    return _sse_pool


def get_sse_redis_pool() -> aioredis.ConnectionPool:
    """Get the initialized SSE pool. Raises RuntimeError if not initialized."""
    if _sse_pool is None:
        raise RuntimeError("SSE Redis pool not initialized. Call init_sse_redis_pool() first.")
    return _sse_pool


def get_sse_redis_client() -> aioredis.Redis:
    """Get a Redis client from the SSE (long-lived-subscriber) pool.

    Only `EventBus.subscribe` should call this. Every other Redis consumer —
    including `EventBus.publish`/`publish_system` — is short-lived and
    belongs on `get_redis_client()`.
    """
    return aioredis.Redis(connection_pool=get_sse_redis_pool())


async def close_redis_pool() -> None:
    """Close both the shared and SSE global pools."""
    global _pool, _pool_url, _sse_pool, _sse_pool_url
    if _pool is not None:
        await _pool.aclose()
        _pool = None
        _pool_url = None
        logger.info("redis.pool_closed")
    if _sse_pool is not None:
        await _sse_pool.aclose()
        _sse_pool = None
        _sse_pool_url = None
        logger.info("redis.sse_pool_closed")
