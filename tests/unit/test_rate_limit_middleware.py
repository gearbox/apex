"""Unit tests for RateLimitMiddleware's Redis failure posture (R1/R2).

Fail OPEN on storage errors, matching TokenRevocationService.is_revoked's
documented posture — a Redis blip must not turn into a 500 on login/register/
content-cookie. See src/api/middleware/rate_limit.py and CLAUDE.md's
"Background Workers" section for the rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from limits.aio.storage import MemoryStorage
from litestar.status_codes import (
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from litestar.testing import AsyncTestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from src.api.app import app as application
from src.api.middleware.rate_limit import (
    MovingWindowRateLimiter,
    get_rate_limiter_storage,
    init_rate_limiter,
)
from src.core.config import Settings

if TYPE_CHECKING:
    from collections.abc import Generator

_PRODUCT_HEADERS = {"X-Product-Id": "vex"}

# redis_url=None -> MemoryStorage, so counters are process-local and never
# actually raise RedisError on their own; the errors below are injected by
# patching MovingWindowRateLimiter directly, which sits in front of whatever
# storage backend is configured.
_TEST_SETTINGS = Settings(
    redis_url=None,
    rate_limit_register="5/hour",
    rate_limit_login="10/minute",
    rate_limit_forgot_password="3/hour",
    rate_limit_resend_verification="3/hour",
    rate_limit_sse_ticket="10/minute",
    trusted_ip_header="x-forwarded-for",
    aisha_poller_enabled=False,
)


@pytest.fixture
def override_limiter() -> Generator[None]:
    with (
        patch("src.api.app.get_settings", return_value=_TEST_SETTINGS),
        patch("src.api.dependencies.common.get_settings", return_value=_TEST_SETTINGS),
        patch("src.api.middleware.rate_limit.get_settings", return_value=_TEST_SETTINGS),
    ):
        init_rate_limiter(_TEST_SETTINGS)
        storage = get_rate_limiter_storage()
        if isinstance(storage, MemoryStorage):
            storage.storage.clear()
        yield


@pytest.mark.asyncio
class TestRateLimiterFailsOpenOnHitError:
    async def test_request_proceeds_instead_of_500(self, override_limiter: None) -> None:  # noqa: ARG002
        with patch.object(MovingWindowRateLimiter, "hit", side_effect=RedisConnectionError("down")):
            async with AsyncTestClient(app=application) as client:
                response = await client.post("/v1/auth/register", json={}, headers=_PRODUCT_HEADERS)
        assert response.status_code != HTTP_429_TOO_MANY_REQUESTS
        assert response.status_code != HTTP_500_INTERNAL_SERVER_ERROR

    async def test_no_rate_limit_headers_on_fail_open_response(
        self,
        override_limiter: None,  # noqa: ARG002
    ) -> None:
        """A fail-open response must never carry X-RateLimit-* headers — a
        client reading X-RateLimit-Remaining: 0 would back off incorrectly."""
        with patch.object(MovingWindowRateLimiter, "hit", side_effect=RedisConnectionError("down")):
            async with AsyncTestClient(app=application) as client:
                response = await client.post("/v1/auth/register", json={}, headers=_PRODUCT_HEADERS)
        assert "x-ratelimit-limit" not in response.headers
        assert "x-ratelimit-remaining" not in response.headers
        assert "x-ratelimit-reset" not in response.headers

    async def test_logs_storage_error_with_route_and_error_type(
        self,
        override_limiter: None,  # noqa: ARG002
    ) -> None:
        with (
            patch.object(MovingWindowRateLimiter, "hit", side_effect=RedisConnectionError("down")),
            patch("src.api.middleware.rate_limit.logger") as mock_logger,
        ):
            async with AsyncTestClient(app=application) as client:
                await client.post("/v1/auth/register", json={}, headers=_PRODUCT_HEADERS)

        mock_logger.warning.assert_any_call(
            "rate_limit.storage_error",
            route="POST /v1/auth/register",
            error="down",
            error_type="ConnectionError",
        )


@pytest.mark.asyncio
class TestRateLimiterFailsOpenOnWindowStatsError:
    async def test_request_proceeds_instead_of_500(self, override_limiter: None) -> None:  # noqa: ARG002
        with patch.object(
            MovingWindowRateLimiter, "get_window_stats", side_effect=RedisConnectionError("down")
        ):
            async with AsyncTestClient(app=application) as client:
                response = await client.post("/v1/auth/register", json={}, headers=_PRODUCT_HEADERS)
        assert response.status_code != HTTP_429_TOO_MANY_REQUESTS
        assert response.status_code != HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
class TestRateLimiterNarrowCatch:
    async def test_non_redis_error_is_not_swallowed(self, override_limiter: None) -> None:  # noqa: ARG002
        """A ValueError (e.g. a `limits` programming error or bad limit
        string) must surface as a request error, not be treated as a
        storage-degradation fail-open — the catch is (RedisError, OSError)
        only, never a blanket Exception."""
        with patch.object(MovingWindowRateLimiter, "hit", side_effect=ValueError("bug")):
            async with AsyncTestClient(app=application) as client:
                response = await client.post("/v1/auth/register", json={}, headers=_PRODUCT_HEADERS)
        assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
class TestRateLimiterHealthyStorageUnaffected:
    async def test_allow_deny_behavior_unchanged(self, override_limiter: None) -> None:  # noqa: ARG002
        """Sanity check: the fail-open try/except introduces no regression
        when storage is healthy — limits are still enforced normally."""
        async with AsyncTestClient(app=application) as client:
            for _ in range(5):
                response = await client.post("/v1/auth/register", json={}, headers=_PRODUCT_HEADERS)
                assert response.status_code != HTTP_429_TOO_MANY_REQUESTS

            response = await client.post("/v1/auth/register", json={}, headers=_PRODUCT_HEADERS)
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS
