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


@pytest.fixture
def override_limiter_defaults() -> None:
    """Fixture to ensure MemoryStorage is used."""
    settings = Settings(
        redis_url=None,
        rate_limit_register="5/hour",
        rate_limit_login="10/minute",
        rate_limit_forgot_password="3/hour",
        rate_limit_resend_verification="3/hour",
    )
    init_rate_limiter(settings)

    # We also need to clear history if it was already used
    storage = get_rate_limiter_storage()
    if isinstance(storage, MemoryStorage):
        storage.storage.clear()


@pytest.mark.asyncio
async def test_register_rate_limit_allows_under_threshold(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        # Make requests under the limit (5 per hour)
        for i in range(5):
            # Using incorrect input on purpose, we only care about if the rate limit middleware allows it through
            response = await client.post(
                "/v1/auth/register", json={"email": f"test{i}@example.com"}
            )
            # The body validation might fail, returning 422 Unprocessable Entity
            # The core point is that it DOES NOT return 429
            assert response.status_code != HTTP_429_TOO_MANY_REQUESTS


@pytest.mark.asyncio
async def test_register_rate_limit_blocks_over_threshold(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        # 5 requests allowed
        for _ in range(5):
            await client.post("/v1/auth/register", json={})

        # 6th request should hit 429 Too Many Requests
        response = await client.post("/v1/auth/register", json={})
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS
        assert "retry-after" in response.headers


@pytest.mark.asyncio
async def test_login_rate_limit_separate_window(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        # Exhaust register rate limit (5)
        for _ in range(5):
            await client.post("/v1/auth/register", json={})

        response_register = await client.post("/v1/auth/register", json={})
        assert response_register.status_code == HTTP_429_TOO_MANY_REQUESTS

        # Login rate limit is independent (10/min) and should succeed
        for _ in range(10):
            response_login = await client.post("/v1/auth/login", json={})
            assert response_login.status_code != HTTP_429_TOO_MANY_REQUESTS

        # 11th should fail
        response_login_11 = await client.post("/v1/auth/login", json={})
        assert response_login_11.status_code == HTTP_429_TOO_MANY_REQUESTS


@pytest.mark.asyncio
async def test_forgot_password_rate_limit(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        # 3 requests allowed
        for _ in range(3):
            response = await client.post("/v1/auth/forgot-password", json={})
            assert response.status_code != HTTP_429_TOO_MANY_REQUESTS

        # 4th should fail
        response = await client.post("/v1/auth/forgot-password", json={})
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS


@pytest.mark.asyncio
async def test_resend_verification_rate_limit(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        # 3 requests allowed
        for _ in range(3):
            response = await client.post("/v1/auth/resend-verification", json={})
            assert response.status_code != HTTP_429_TOO_MANY_REQUESTS

        # 4th should fail
        response = await client.post("/v1/auth/resend-verification", json={})
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS


@pytest.mark.asyncio
async def test_different_ips_have_separate_counters(override_limiter_defaults: None) -> None:  # noqa: ARG001
    async with AsyncTestClient(app=application) as client:
        # Exhaust limit for IP 1
        headers_ip1 = {"X-Forwarded-For": "192.168.1.1"}
        for _ in range(5):
            await client.post("/v1/auth/register", json={}, headers=headers_ip1)

        response_ip1 = await client.post("/v1/auth/register", json={}, headers=headers_ip1)
        assert response_ip1.status_code == HTTP_429_TOO_MANY_REQUESTS

        # IP 2 should have its own fresh counter
        headers_ip2 = {"X-Forwarded-For": "192.168.1.2"}
        response_ip2 = await client.post("/v1/auth/register", json={}, headers=headers_ip2)
        assert response_ip2.status_code != HTTP_429_TOO_MANY_REQUESTS
