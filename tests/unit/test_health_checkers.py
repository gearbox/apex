"""Tests for infrastructure health checkers."""

from unittest.mock import AsyncMock, MagicMock

from src.api.services.health.checkers.infrastructure import (
    PostgresChecker,
    R2Checker,
    RedisChecker,
)
from src.api.services.health.checkers.token_revocation import TokenRevocationChecker
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

    def test_does_not_gate_readiness(self) -> None:
        """issue #142 G2 — a slow HeadBucket must never fail GET /health/ready."""
        assert R2Checker.gates_readiness is False


class TestTokenRevocationChecker:
    """issue #142 G2/A2 — diagnostic checker, must never gate readiness."""

    def test_does_not_gate_readiness(self) -> None:
        assert TokenRevocationChecker.gates_readiness is False

    async def test_inactive_when_disabled(self) -> None:
        mock_service = MagicMock()
        mock_service.enabled = False
        mock_service.failed_write_count = 0
        checker = TokenRevocationChecker(mock_service)

        result = await checker.check()

        assert result.status == ComponentStatus.inactive

    async def test_unhealthy_when_circuit_open(self) -> None:
        mock_service = MagicMock()
        mock_service.enabled = True
        mock_service.circuit_open = True
        mock_service.failed_write_count = 2
        checker = TokenRevocationChecker(mock_service)

        result = await checker.check()

        assert result.status == ComponentStatus.unhealthy

    async def test_healthy_when_enabled_and_circuit_closed(self) -> None:
        mock_service = MagicMock()
        mock_service.enabled = True
        mock_service.circuit_open = False
        mock_service.failed_write_count = 0
        checker = TokenRevocationChecker(mock_service)

        result = await checker.check()

        assert result.status == ComponentStatus.healthy
