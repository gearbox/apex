from collections.abc import Generator
from unittest.mock import patch

import pytest
from limits.storage import MemoryStorage
from litestar.status_codes import HTTP_429_TOO_MANY_REQUESTS
from litestar.testing import AsyncTestClient

from src.api.app import app as application
from src.api.middleware.rate_limit import (
    get_rate_limiter_storage,
    init_rate_limiter,
)
from src.core.config import Settings

# Default headers to satisfy ProductMiddleware in unit tests
_PRODUCT_HEADERS = {"X-Product-Id": "vex"}

# Shared test settings — redis_url=None forces MemoryStorage so the rate limit
# counters are process-local and isolated across tests even when REDIS_URL is
# present in the environment (e.g. `make test-all` Docker environment).
_TEST_SETTINGS = Settings(
    redis_url=None,
    rate_limit_register="5/hour",
    rate_limit_login="10/minute",
    rate_limit_forgot_password="3/hour",
    rate_limit_resend_verification="3/hour",
    rate_limit_sse_ticket="10/minute",
)


@pytest.fixture
def override_limiter_defaults() -> Generator[None]:
    """Ensure MemoryStorage is used for every rate-limiting unit test.

    Patches get_settings() in both the app lifespan and the dependency
    container so the lifespan never creates a RedisStorage that would
    persist counters across tests.
    """
    with (
        patch("src.api.app.get_settings", return_value=_TEST_SETTINGS),
        patch("src.api.dependencies.common.get_settings", return_value=_TEST_SETTINGS),
    ):
        init_rate_limiter(_TEST_SETTINGS)
        storage = get_rate_limiter_storage()
        if isinstance(storage, MemoryStorage):
            storage.storage.clear()
        yield


@pytest.mark.asyncio
async def test_register_rate_limit_allows_under_threshold(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        # Make requests under the limit (5 per hour)
        for i in range(5):
            # Using incorrect input on purpose, we only care about if the rate limit middleware allows it through
            response = await client.post(
                "/v1/auth/register",
                json={"email": f"test{i}@example.com"},
                headers=_PRODUCT_HEADERS,
            )
            # The body validation might fail, returning 422 Unprocessable Entity
            # The core point is that it DOES NOT return 429
            assert response.status_code != HTTP_429_TOO_MANY_REQUESTS


@pytest.mark.asyncio
async def test_register_rate_limit_blocks_over_threshold(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        # 5 requests allowed
        for _ in range(5):
            await client.post("/v1/auth/register", json={}, headers=_PRODUCT_HEADERS)

        # 6th request should hit 429 Too Many Requests
        response = await client.post("/v1/auth/register", json={}, headers=_PRODUCT_HEADERS)
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS
        assert "retry-after" in response.headers


@pytest.mark.asyncio
async def test_login_rate_limit_separate_window(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        # Exhaust register rate limit (5)
        for _ in range(5):
            await client.post("/v1/auth/register", json={}, headers=_PRODUCT_HEADERS)

        response_register = await client.post(
            "/v1/auth/register", json={}, headers=_PRODUCT_HEADERS
        )
        assert response_register.status_code == HTTP_429_TOO_MANY_REQUESTS

        # Login rate limit is independent (10/min) and should succeed
        for _ in range(10):
            response_login = await client.post("/v1/auth/login", json={}, headers=_PRODUCT_HEADERS)
            assert response_login.status_code != HTTP_429_TOO_MANY_REQUESTS

        # 11th should fail
        response_login_11 = await client.post("/v1/auth/login", json={}, headers=_PRODUCT_HEADERS)
        assert response_login_11.status_code == HTTP_429_TOO_MANY_REQUESTS


@pytest.mark.asyncio
async def test_forgot_password_rate_limit(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        # 3 requests allowed
        for _ in range(3):
            response = await client.post(
                "/v1/auth/forgot-password", json={}, headers=_PRODUCT_HEADERS
            )
            assert response.status_code != HTTP_429_TOO_MANY_REQUESTS

        # 4th should fail
        response = await client.post("/v1/auth/forgot-password", json={}, headers=_PRODUCT_HEADERS)
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS


@pytest.mark.asyncio
async def test_resend_verification_rate_limit(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        # 3 requests allowed
        for _ in range(3):
            response = await client.post(
                "/v1/auth/resend-verification", json={}, headers=_PRODUCT_HEADERS
            )
            assert response.status_code != HTTP_429_TOO_MANY_REQUESTS

        # 4th should fail
        response = await client.post(
            "/v1/auth/resend-verification", json={}, headers=_PRODUCT_HEADERS
        )
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS


@pytest.mark.asyncio
async def test_rate_limit_headers_on_allowed_requests(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        response = await client.post("/v1/auth/register", json={}, headers=_PRODUCT_HEADERS)
        assert response.status_code != HTTP_429_TOO_MANY_REQUESTS
        assert "x-ratelimit-limit" in response.headers
        assert "x-ratelimit-remaining" in response.headers
        assert "x-ratelimit-reset" in response.headers
        assert int(response.headers["x-ratelimit-limit"]) == 5
        assert int(response.headers["x-ratelimit-remaining"]) == 4


@pytest.mark.asyncio
async def test_rate_limit_headers_on_429(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        for _ in range(5):
            await client.post("/v1/auth/register", json={}, headers=_PRODUCT_HEADERS)

        response = await client.post("/v1/auth/register", json={}, headers=_PRODUCT_HEADERS)
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS

        # All four headers must be present
        assert "retry-after" in response.headers
        assert "x-ratelimit-limit" in response.headers
        assert "x-ratelimit-remaining" in response.headers
        assert "x-ratelimit-reset" in response.headers
        assert response.headers["x-ratelimit-remaining"] == "0"

        # Body must follow the unified error envelope
        body = response.json()
        assert body["error"] == "rate_limited"
        assert body["status_code"] == HTTP_429_TOO_MANY_REQUESTS
        assert "retry_after" in body["detail"]


@pytest.mark.asyncio
async def test_different_ips_have_separate_counters(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        # Exhaust limit for IP 1
        headers_ip1 = {"X-Forwarded-For": "192.168.1.1", **_PRODUCT_HEADERS}
        for _ in range(5):
            await client.post("/v1/auth/register", json={}, headers=headers_ip1)

        response_ip1 = await client.post("/v1/auth/register", json={}, headers=headers_ip1)
        assert response_ip1.status_code == HTTP_429_TOO_MANY_REQUESTS

        # IP 2 should have its own fresh counter
        headers_ip2 = {"X-Forwarded-For": "192.168.1.2", **_PRODUCT_HEADERS}
        response_ip2 = await client.post("/v1/auth/register", json={}, headers=headers_ip2)
        assert response_ip2.status_code != HTTP_429_TOO_MANY_REQUESTS
