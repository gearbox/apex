"""Tests for infrastructure health checkers."""

from unittest.mock import AsyncMock, MagicMock

from src.api.services.health.checkers.infrastructure import (
    PostgresChecker,
    R2Checker,
    RedisChecker,
)
from src.core.enums import ComponentStatus


class TestPostgresChecker:
    async def test_healthy(self) -> None:
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_session)

        checker = PostgresChecker(session_factory=mock_factory)
        result = await checker.check()
        assert result.status == ComponentStatus.healthy

    async def test_unhealthy_on_exception(self) -> None:
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(side_effect=ConnectionError("refused"))

        mock_factory = MagicMock(return_value=mock_session)

        checker = PostgresChecker(session_factory=mock_factory)
        result = await checker.check()
        assert result.status == ComponentStatus.unhealthy
        assert "refused" in result.message


class TestRedisChecker:
    async def test_healthy_bool(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        checker = RedisChecker(redis=mock_redis)
        result = await checker.check()
        assert result.status == ComponentStatus.healthy

    async def test_healthy_pong(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=b"PONG")
        checker = RedisChecker(redis=mock_redis)
        result = await checker.check()
        assert result.status == ComponentStatus.healthy

    async def test_unhealthy_on_exception(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("no redis"))
        checker = RedisChecker(redis=mock_redis)
        result = await checker.check()
        assert result.status == ComponentStatus.unhealthy


class TestR2Checker:
    async def test_healthy(self) -> None:
        mock_r2 = AsyncMock()
        mock_r2.health_check = AsyncMock(return_value=True)
        checker = R2Checker(r2_storage=mock_r2)
        result = await checker.check()
        assert result.status == ComponentStatus.healthy

    async def test_unhealthy_when_health_check_false(self) -> None:
        mock_r2 = AsyncMock()
        mock_r2.health_check = AsyncMock(return_value=False)
        checker = R2Checker(r2_storage=mock_r2)
        result = await checker.check()
        assert result.status == ComponentStatus.unhealthy

    async def test_unhealthy_on_exception(self) -> None:
        mock_r2 = AsyncMock()
        mock_r2.health_check = AsyncMock(side_effect=Exception("access denied"))
        checker = R2Checker(r2_storage=mock_r2)
        result = await checker.check()
        assert result.status == ComponentStatus.unhealthy
