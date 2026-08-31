"""Unit tests for Redis connection pool management."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.core.redis as redis_module
from src.core.redis import (
    _redacted_url,
    close_redis_pool,
    get_operational_redis_client,
    get_operational_redis_pool,
    get_redis_client,
    get_redis_pool,
    get_sse_redis_client,
    get_sse_redis_pool,
    init_operational_redis_pool,
    init_redis_pool,
    init_sse_redis_pool,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_pool() -> object:
    """Reset the global Redis pools after each test."""
    original_pool = redis_module._pool
    original_pool_url = redis_module._pool_url
    original_sse_pool = redis_module._sse_pool
    original_sse_pool_url = redis_module._sse_pool_url
    original_operational_pool = redis_module._operational_pool
    original_operational_pool_url = redis_module._operational_pool_url
    yield
    redis_module._pool = original_pool
    redis_module._pool_url = original_pool_url
    redis_module._sse_pool = original_sse_pool
    redis_module._sse_pool_url = original_sse_pool_url
    redis_module._operational_pool = original_operational_pool
    redis_module._operational_pool_url = original_operational_pool_url


class TestInitRedisPool:
    def test_creates_pool_and_stores_globally(self) -> None:
        with patch("src.core.redis.aioredis.ConnectionPool.from_url") as mock_from_url:
            mock_pool = MagicMock()
            mock_from_url.return_value = mock_pool

            result = init_redis_pool("redis://localhost:6379")

        assert result is mock_pool
        assert redis_module._pool is mock_pool
        assert redis_module._pool_url == "redis://localhost:6379"
        mock_from_url.assert_called_once_with(
            "redis://localhost:6379",
            decode_responses=True,
            socket_connect_timeout=0.25,
            socket_timeout=0.05,
            retry_on_timeout=False,
            health_check_interval=30.0,
            max_connections=50,
        )

    def test_passes_through_custom_timeouts(self) -> None:
        with patch("src.core.redis.aioredis.ConnectionPool.from_url") as mock_from_url:
            init_redis_pool(
                "redis://localhost:6379",
                socket_connect_timeout=1.5,
                socket_timeout=2.5,
                health_check_interval=60.0,
                max_connections=123,
            )

        mock_from_url.assert_called_once_with(
            "redis://localhost:6379",
            decode_responses=True,
            socket_connect_timeout=1.5,
            socket_timeout=2.5,
            retry_on_timeout=False,
            health_check_interval=60.0,
            max_connections=123,
        )

    def test_repeat_call_same_url_is_noop_and_returns_existing_pool(self) -> None:
        with patch("src.core.redis.aioredis.ConnectionPool.from_url") as mock_from_url:
            mock_pool = MagicMock()
            mock_from_url.return_value = mock_pool

            first = init_redis_pool("redis://localhost:6379")
            second = init_redis_pool("redis://localhost:6379")

        assert first is mock_pool
        assert second is mock_pool
        mock_from_url.assert_called_once()

    def test_repeat_call_different_url_raises(self) -> None:
        with patch("src.core.redis.aioredis.ConnectionPool.from_url") as mock_from_url:
            mock_from_url.return_value = MagicMock()
            init_redis_pool("redis://localhost:6379")

            with pytest.raises(RuntimeError, match="different URL"):
                init_redis_pool("redis://otherhost:6379")


class TestInitOperationalRedisPool:
    def test_creates_pool_and_stores_globally(self) -> None:
        with patch("src.core.redis.aioredis.ConnectionPool.from_url") as mock_from_url:
            mock_pool = MagicMock()
            mock_from_url.return_value = mock_pool

            result = init_operational_redis_pool("redis://localhost:6379")

        assert result is mock_pool
        assert redis_module._operational_pool is mock_pool
        assert redis_module._operational_pool_url == "redis://localhost:6379"
        mock_from_url.assert_called_once_with(
            "redis://localhost:6379",
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.75,
            retry_on_timeout=True,
            health_check_interval=30.0,
            max_connections=20,
        )

    def test_passes_through_custom_timeouts(self) -> None:
        with patch("src.core.redis.aioredis.ConnectionPool.from_url") as mock_from_url:
            init_operational_redis_pool(
                "redis://localhost:6379",
                socket_connect_timeout=1.5,
                socket_timeout=2.5,
                health_check_interval=60.0,
                max_connections=123,
            )

        mock_from_url.assert_called_once_with(
            "redis://localhost:6379",
            decode_responses=True,
            socket_connect_timeout=1.5,
            socket_timeout=2.5,
            retry_on_timeout=True,
            health_check_interval=60.0,
            max_connections=123,
        )

    def test_repeat_call_same_url_is_noop_and_returns_existing_pool(self) -> None:
        with patch("src.core.redis.aioredis.ConnectionPool.from_url") as mock_from_url:
            mock_pool = MagicMock()
            mock_from_url.return_value = mock_pool

            first = init_operational_redis_pool("redis://localhost:6379")
            second = init_operational_redis_pool("redis://localhost:6379")

        assert first is mock_pool
        assert second is mock_pool
        mock_from_url.assert_called_once()

    def test_repeat_call_different_url_raises(self) -> None:
        with patch("src.core.redis.aioredis.ConnectionPool.from_url") as mock_from_url:
            mock_from_url.return_value = MagicMock()
            init_operational_redis_pool("redis://localhost:6379")

            with pytest.raises(RuntimeError, match="different URL"):
                init_operational_redis_pool("redis://otherhost:6379")


class TestGetOperationalRedisPool:
    def test_returns_pool_when_initialized(self) -> None:
        mock_pool = MagicMock()
        redis_module._operational_pool = mock_pool

        assert get_operational_redis_pool() is mock_pool

    def test_raises_when_not_initialized(self) -> None:
        redis_module._operational_pool = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_operational_redis_pool()


class TestGetOperationalRedisClient:
    def test_returns_client_from_operational_pool(self) -> None:
        mock_pool = MagicMock()
        redis_module._operational_pool = mock_pool

        with patch("src.core.redis.aioredis.Redis") as mock_redis_cls:
            mock_client = MagicMock()
            mock_redis_cls.return_value = mock_client

            client = get_operational_redis_client()

        assert client is mock_client
        mock_redis_cls.assert_called_once_with(connection_pool=mock_pool)


class TestRedactedUrl:
    def test_strips_password(self) -> None:
        assert _redacted_url("redis://:supersecret@redis:6379/0") == "redis://redis:6379/0"

    def test_strips_username_and_password(self) -> None:
        assert _redacted_url("redis://user:supersecret@redis:6379/0") == "redis://redis:6379/0"

    def test_no_credentials_unchanged(self) -> None:
        assert _redacted_url("redis://localhost:6379") == "redis://localhost:6379"

    def test_pool_initialized_log_never_contains_password(self) -> None:
        # Mocks the module logger directly rather than structlog.testing.capture_logs():
        # capture_logs() patches the *global* processors list, which is fragile across
        # this test session (cache_logger_on_first_use permanently binds module-level
        # loggers to whatever list is live on their first real call — see
        # tests/unit/api/conftest.py for the full story). Asserting on the mock's call
        # args is deterministic regardless of what else in the suite touches structlog.
        with (
            patch("src.core.redis.aioredis.ConnectionPool.from_url"),
            patch("src.core.redis.logger") as mock_logger,
        ):
            init_redis_pool("redis://:supersecret@redis:6379/0")

        mock_logger.info.assert_called_once_with(
            "redis.pool_initialized",
            url="redis://redis:6379/0",
            socket_connect_timeout=0.25,
            socket_timeout=0.05,
            health_check_interval=30.0,
            max_connections=50,
        )


class TestGetRedisPool:
    def test_returns_pool_when_initialized(self) -> None:
        mock_pool = MagicMock()
        redis_module._pool = mock_pool

        assert get_redis_pool() is mock_pool

    def test_raises_when_not_initialized(self) -> None:
        redis_module._pool = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_redis_pool()


class TestInitSseRedisPool:
    def test_creates_pool_and_stores_globally(self) -> None:
        with patch("src.core.redis.aioredis.ConnectionPool.from_url") as mock_from_url:
            mock_pool = MagicMock()
            mock_from_url.return_value = mock_pool

            result = init_sse_redis_pool("redis://localhost:6379")

        assert result is mock_pool
        assert redis_module._sse_pool is mock_pool
        assert redis_module._sse_pool_url == "redis://localhost:6379"
        mock_from_url.assert_called_once_with(
            "redis://localhost:6379",
            decode_responses=True,
            socket_connect_timeout=0.25,
            socket_timeout=0.05,
            retry_on_timeout=False,
            health_check_interval=30.0,
            max_connections=500,
        )

    def test_repeat_call_same_url_is_noop(self) -> None:
        with patch("src.core.redis.aioredis.ConnectionPool.from_url") as mock_from_url:
            mock_pool = MagicMock()
            mock_from_url.return_value = mock_pool

            first = init_sse_redis_pool("redis://localhost:6379")
            second = init_sse_redis_pool("redis://localhost:6379")

        assert first is mock_pool
        assert second is mock_pool
        mock_from_url.assert_called_once()

    def test_repeat_call_different_url_raises(self) -> None:
        with patch("src.core.redis.aioredis.ConnectionPool.from_url") as mock_from_url:
            mock_from_url.return_value = MagicMock()
            init_sse_redis_pool("redis://localhost:6379")

            with pytest.raises(RuntimeError, match="different URL"):
                init_sse_redis_pool("redis://otherhost:6379")

    def test_default_and_sse_pools_are_distinct_objects(self) -> None:
        with patch("src.core.redis.aioredis.ConnectionPool.from_url") as mock_from_url:
            default_pool = MagicMock(name="default_pool")
            sse_pool = MagicMock(name="sse_pool")
            mock_from_url.side_effect = [default_pool, sse_pool]

            init_redis_pool("redis://localhost:6379")
            init_sse_redis_pool("redis://localhost:6379")

        assert get_redis_pool() is default_pool
        assert get_sse_redis_pool() is sse_pool
        assert get_redis_pool() is not get_sse_redis_pool()


class TestGetSseRedisPool:
    def test_returns_pool_when_initialized(self) -> None:
        mock_pool = MagicMock()
        redis_module._sse_pool = mock_pool

        assert get_sse_redis_pool() is mock_pool

    def test_raises_when_not_initialized(self) -> None:
        redis_module._sse_pool = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_sse_redis_pool()

    def test_max_connections_reflects_configured_setting(self) -> None:
        """Pins the public `max_connections` attribute that
        event_bus.subscribe_failed / health.sse.subscribe_failed logging
        reads — must stay the real, undoctored redis-py attribute, not a
        private one that can silently change shape across releases. Uses the
        real (lazily-connecting) ConnectionPool.from_url rather than a mock so
        the attribute is genuinely sourced from the constructor arg."""
        init_sse_redis_pool("redis://localhost:6379", max_connections=321)

        assert get_sse_redis_pool().max_connections == 321


class TestGetSseRedisClient:
    def test_returns_client_from_sse_pool(self) -> None:
        mock_pool = MagicMock()
        redis_module._sse_pool = mock_pool

        with patch("src.core.redis.aioredis.Redis") as mock_redis_cls:
            mock_client = MagicMock()
            mock_redis_cls.return_value = mock_client

            client = get_sse_redis_client()

        assert client is mock_client
        mock_redis_cls.assert_called_once_with(connection_pool=mock_pool)


class TestCloseRedisPool:
    async def test_closes_and_clears_all_pools(self) -> None:
        mock_pool = AsyncMock()
        mock_sse_pool = AsyncMock()
        mock_operational_pool = AsyncMock()
        redis_module._pool = mock_pool
        redis_module._pool_url = "redis://localhost:6379"
        redis_module._sse_pool = mock_sse_pool
        redis_module._sse_pool_url = "redis://localhost:6379"
        redis_module._operational_pool = mock_operational_pool
        redis_module._operational_pool_url = "redis://localhost:6379"

        await close_redis_pool()

        mock_pool.aclose.assert_awaited_once()
        mock_sse_pool.aclose.assert_awaited_once()
        mock_operational_pool.aclose.assert_awaited_once()
        assert redis_module._pool is None
        assert redis_module._pool_url is None
        assert redis_module._sse_pool is None
        assert redis_module._sse_pool_url is None
        assert redis_module._operational_pool is None
        assert redis_module._operational_pool_url is None

    async def test_noop_when_pools_are_none(self) -> None:
        redis_module._pool = None
        redis_module._sse_pool = None
        redis_module._operational_pool = None
        # Should not raise
        await close_redis_pool()

    async def test_closing_default_pool_does_not_require_sse_pool(self) -> None:
        """Only the default pool was ever initialized (e.g. a non-SSE worker process)."""
        mock_pool = AsyncMock()
        redis_module._pool = mock_pool
        redis_module._sse_pool = None

        await close_redis_pool()

        mock_pool.aclose.assert_awaited_once()
        assert redis_module._pool is None


class TestGetRedisClient:
    def test_returns_client_from_pool(self) -> None:
        mock_pool = MagicMock()
        redis_module._pool = mock_pool

        with patch("src.core.redis.aioredis.Redis") as mock_redis_cls:
            mock_client = MagicMock()
            mock_redis_cls.return_value = mock_client

            client = get_redis_client()

        assert client is mock_client
        mock_redis_cls.assert_called_once_with(connection_pool=mock_pool)
