"""Rate limiting middleware.

Uses the `limits` library for sliding-window counters backed by Redis
(or in-memory when Redis is not configured). Uses the async storage/strategy
variants (`limits.aio`) — under Redis, `hit()`/`get_window_stats()` are
network round-trips and must not block the event loop.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog
from limits import parse
from limits.aio.storage import MemoryStorage, RedisStorage, Storage
from limits.aio.strategies import MovingWindowRateLimiter
from litestar import Request, Response
from litestar.enums import ScopeType
from litestar.middleware import MiddlewareProtocol
from litestar.status_codes import HTTP_429_TOO_MANY_REQUESTS

from src.api.schemas.errors import ErrorEnvelope
from src.core.config import Settings, get_settings

if TYPE_CHECKING:
    from litestar.types import ASGIApp, Message, Receive, Scope, Send

logger = structlog.get_logger(__name__)

# Global storage instance
_limiter_storage: Storage | None = None


def create_storage(redis_url: str | None) -> Storage:
    """Create limits storage backend.

    Args:
        redis_url: Redis connection string, or None for in-memory.

    Returns:
        Storage instance (Redis or Memory).
    """
    if not redis_url:
        return MemoryStorage()
    # limits.aio's default async Redis implementation is coredis, which isn't
    # in this project's dependency tree. redispy (the plain `redis` package,
    # already a dependency and used elsewhere via redis.asyncio — see
    # src/core/redis.py) has async support and works as a drop-in.
    return RedisStorage(redis_url, implementation="redispy")


def init_rate_limiter(settings: Settings) -> None:
    """Initialize the global rate limiter storage.

    Args:
        settings: Application settings.
    """
    global _limiter_storage
    _limiter_storage = create_storage(settings.redis_url)
    logger.info("rate_limiter.initialized", backend="redis" if settings.redis_url else "memory")

    if settings.trusted_ip_header == "none" and not settings.debug:
        logger.warning(
            "rate_limiter.trusted_ip_header_unset",
            hint=(
                "trusted_ip_header is 'none' outside debug mode — rate limiting keys "
                "on the raw ASGI connection address, which in production is almost "
                "always a single load-balancer/proxy IP shared by every client. Set "
                "TRUSTED_IP_HEADER=cf-connecting-ip (behind Cloudflare) or "
                "x-forwarded-for (behind a generic reverse proxy, plus "
                "TRUSTED_PROXY_HOPS) so per-client limits are actually per-client."
            ),
        )


def get_rate_limiter_storage() -> Storage:
    """Get the initialized rate limiter storage.

    Returns:
        Storage instance.

    Raises:
        RuntimeError: If storage is not initialized.
    """
    if _limiter_storage is None:
        raise RuntimeError("Rate limiter storage not initialized")
    return _limiter_storage


def get_real_ip(request: Request[Any, Any, Any], settings: Settings) -> str:
    """Extract the real client IP address from the configured trusted source.

    Trusts only the header configured via ``settings.trusted_ip_header`` —
    never the leftmost X-Forwarded-For entry, which is fully client-controlled
    and lets any caller mint a fresh rate-limit bucket per request. With no
    trusted header configured ("none", the default), falls back to the raw
    ASGI connection address only.

    Args:
        request: The incoming request.
        settings: Application settings (trusted_ip_header / trusted_proxy_hops).

    Returns:
        The client IP address string.
    """
    if settings.trusted_ip_header == "cf-connecting-ip":
        if ip := request.headers.get("CF-Connecting-IP"):
            return ip.strip()
    elif (
        settings.trusted_ip_header == "x-forwarded-for"
        and (forwarded_for := request.headers.get("X-Forwarded-For"))
        and (parts := [p.strip() for p in forwarded_for.split(",") if p.strip()])
    ):
        # Rightmost-minus-hops: the entry appended by the trusted proxy
        # closest to us, never the client-supplied leftmost entry.
        idx = max(0, len(parts) - 1 - settings.trusted_proxy_hops)
        return parts[idx]

    return request.client.host if request.client else "127.0.0.1"


def build_rate_limit_config(settings: Settings) -> dict[str, str]:
    """Build the route configuration mapping for the middleware.

    Args:
        settings: Application settings containing limit strings.

    Returns:
        Dictionary mapping '{METHOD} {path}' to limit strings.
    """
    return {
        "POST /v1/auth/register": settings.rate_limit_register,
        "POST /v1/auth/login": settings.rate_limit_login,
        "POST /v1/auth/forgot-password": settings.rate_limit_forgot_password,
        "POST /v1/auth/resend-verification": settings.rate_limit_resend_verification,
        "POST /v1/events/sse-ticket": settings.rate_limit_sse_ticket,
    }


class RateLimitMiddleware(MiddlewareProtocol):
    """Middleware for rate limiting specific routes.

    Checks requests against configured limits using the limits library.
    Returns 429 Too Many Requests with Retry-After header on limit breach.
    """

    def __init__(self, app: ASGIApp, config: dict[str, str]) -> None:
        """Initialize middleware.

        Args:
            app: The next ASGI app in the chain.
            config: Mapping of 'METHOD /path' strings to limits strings (e.g. '5/hour').
        """
        self.app = app

        # Parse limits once during initialization
        self.route_limits = {}
        for route_key, limit_str in config.items():
            self.route_limits[route_key] = parse(limit_str)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle incoming ASGI request."""
        if scope["type"] != ScopeType.HTTP:
            return await self.app(scope, receive, send)

        request: Request[Any, Any, Any] = Request(scope)
        method = request.method
        path = request.url.path
        route_key = f"{method} {path}"

        limit_item = self.route_limits.get(route_key)

        if not limit_item:
            # Route not rate limited, pass through
            return await self.app(scope, receive, send)

        storage = get_rate_limiter_storage()
        limiter = MovingWindowRateLimiter(storage)
        # Resolved fresh per request (not captured at middleware construction,
        # which happens once at app-creation time) so trust-mode config is
        # never stale relative to the currently active Settings.
        ip = get_real_ip(request, get_settings())

        # We form a unique key using the route and the IP
        key = f"rate_limit:{route_key}:{ip}"

        # hit() atomically increments and returns False when limit is exceeded
        allowed = await limiter.hit(limit_item, key)
        stats = await limiter.get_window_stats(limit_item, key)

        limit = limit_item.amount
        remaining = max(0, stats.remaining)
        reset_time = int(stats.reset_time)

        if not allowed:
            retry_after = max(0, reset_time - int(time.time()))

            logger.warning("rate_limit.exceeded", ip=ip, route=route_key)

            response = Response(
                content=ErrorEnvelope(
                    error="rate_limited",
                    message=f"Too many requests. Try again in {retry_after} seconds.",
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    detail={"retry_after": retry_after},
                ),
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                },
            )
            asgi_response = response.to_asgi_response(app=None, request=request)
            return await asgi_response(scope, receive, send)

        # Inject rate limit headers into every non-429 response
        rate_limit_headers = [
            (b"x-ratelimit-limit", str(limit).encode()),
            (b"x-ratelimit-remaining", str(remaining).encode()),
            (b"x-ratelimit-reset", str(reset_time).encode()),
        ]

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = {
                    **message,
                    "headers": list(message.get("headers", [])) + rate_limit_headers,
                }
            await send(message)

        return await self.app(scope, receive, send_with_headers)
