"""Unit tests for Redis connection pool management."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.core.redis as redis_module
from src.core.redis import (
    _redacted_url,
    close_redis_pool,
    get_redis_client,
    get_redis_pool,
    init_redis_pool,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_pool() -> object:
    """Reset the global Redis pool after each test."""
    original = redis_module._pool
    yield
    redis_module._pool = original


class TestInitRedisPool:
    def test_creates_pool_and_stores_globally(self) -> None:
        with patch("src.core.redis.aioredis.ConnectionPool.from_url") as mock_from_url:
            mock_pool = MagicMock()
            mock_from_url.return_value = mock_pool

            result = init_redis_pool("redis://localhost:6379")

        assert result is mock_pool
        assert redis_module._pool is mock_pool
        mock_from_url.assert_called_once_with("redis://localhost:6379", decode_responses=True)


class TestRedactedUrl:
    def test_strips_password(self) -> None:
        assert _redacted_url("redis://:supersecret@redis:6379/0") == "redis://redis:6379/0"

    def test_strips_username_and_password(self) -> None:
        assert _redacted_url("redis://user:supersecret@redis:6379/0") == "redis://redis:6379/0"

    def test_no_credentials_unchanged(self) -> None:
        assert _redacted_url("redis://localhost:6379") == "redis://localhost:6379"

    def test_pool_initialized_log_never_contains_password(self) -> None:
        import structlog

        with (
            patch("src.core.redis.aioredis.ConnectionPool.from_url"),
            structlog.testing.capture_logs() as captured,
        ):
            init_redis_pool("redis://:supersecret@redis:6379/0")

        logged = [entry for entry in captured if entry.get("event") == "redis.pool_initialized"]
        assert logged, "expected a redis.pool_initialized log entry"
        assert "supersecret" not in str(logged[0])


class TestGetRedisPool:
    def test_returns_pool_when_initialized(self) -> None:
        mock_pool = MagicMock()
        redis_module._pool = mock_pool

        assert get_redis_pool() is mock_pool

    def test_raises_when_not_initialized(self) -> None:
        redis_module._pool = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_redis_pool()


class TestCloseRedisPool:
    async def test_closes_and_clears_pool(self) -> None:
        mock_pool = AsyncMock()
        redis_module._pool = mock_pool

        await close_redis_pool()

        mock_pool.aclose.assert_awaited_once()
        assert redis_module._pool is None

    async def test_noop_when_pool_is_none(self) -> None:
        redis_module._pool = None
        # Should not raise
        await close_redis_pool()


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
