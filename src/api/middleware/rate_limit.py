"""Rate limiting middleware.

Uses the `limits` library for sliding-window counters backed by Redis
(or in-memory when Redis is not configured).
"""

from __future__ import annotations

import structlog
from limits import parse
from limits.storage import MemoryStorage, RedisStorage, Storage
from limits.strategies import MovingWindowRateLimiter
from litestar import Request, Response
from litestar.middleware import MiddlewareProtocol
from litestar.status_codes import HTTP_429_TOO_MANY_REQUESTS
from litestar.types import ASGIApp, Receive, Scope, Send

from src.core.config import Settings

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
    return RedisStorage(redis_url) if redis_url else MemoryStorage()


def init_rate_limiter(settings: Settings) -> None:
    """Initialize the global rate limiter storage.

    Args:
        settings: Application settings.
    """
    global _limiter_storage
    _limiter_storage = create_storage(settings.redis_url)
    logger.info("rate_limiter.initialized", backend="redis" if settings.redis_url else "memory")


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


def get_real_ip(request: Request) -> str:
    """Extract real client IP address.

    Respects X-Forwarded-For if present (assuming litestar runs behind a proxy),
    otherwise falls back to the connection client host.

    Args:
        request: The incoming request.

    Returns:
        The client IP address string.
    """
    if forwarded_for := request.headers.get("X-Forwarded-For"):
        # First IP in the list is the original client
        return forwarded_for.split(",")[0].strip()

    return request.client.host if request.client else "127.0.0.1"


def build_rate_limit_config(settings: Settings) -> dict[str, str]:
    """Build the route configuration mapping for the middleware.

    Args:
        settings: Application settings containing limit strings.

    Returns:
        Dictionary mapping '{METHOD} {path}' to limit strings.
    """
    return {
        "POST /api/v1/auth/register": settings.rate_limit_register,
        "POST /api/v1/auth/login": settings.rate_limit_login,
        "POST /api/v1/auth/forgot-password": settings.rate_limit_forgot_password,
        "POST /api/v1/auth/resend-verification": settings.rate_limit_resend_verification,
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
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope)
        method = request.method
        path = request.url.path
        route_key = f"{method} {path}"

        limit_item = self.route_limits.get(route_key)

        if not limit_item:
            # Route not rate limited, pass through
            return await self.app(scope, receive, send)

        storage = get_rate_limiter_storage()
        limiter = MovingWindowRateLimiter(storage)
        ip = get_real_ip(request)

        # We form a unique key using the route and the IP
        key = f"rate_limit:{route_key}:{ip}"

        if not limiter.test(limit_item, key):
            # Limit exceeded
            # Find time until reset window
            retry_after = str(int(limit_item.get_expiry()))

            logger.warning("rate_limit.exceeded", ip=ip, route=route_key)

            response = Response(
                content={"detail": "Too Many Requests"},
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": retry_after},
            )
            # Middleware must return ASGI application response
            asgi_response = response.to_asgi_response(app=None, request=request)
            return await asgi_response(scope, receive, send)

        # Hit the storage to record the request
        limiter.hit(limit_item, key)
        return await self.app(scope, receive, send)
