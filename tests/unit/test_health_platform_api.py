"""Tests for platform API health checkers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx

from src.api.services.health.checkers.platform_api import VastAIChecker
from src.core.enums import ComponentCategory, ComponentStatus


class TestVastAIChecker:
    def test_name_and_category(self) -> None:
        checker = VastAIChecker(http_client=AsyncMock(), api_key="key")
        assert checker.name == "vastai_api"
        assert checker.category == ComponentCategory.platform_api
        assert checker.product_id is None  # global

    async def test_inactive_when_no_api_key(self) -> None:
        checker = VastAIChecker(http_client=AsyncMock(), api_key=None)
        result = await checker.check()
        assert result.status == ComponentStatus.inactive
        assert result.message is not None
        assert "not configured" in result.message

    async def test_inactive_when_empty_api_key(self) -> None:
        checker = VastAIChecker(http_client=AsyncMock(), api_key="")
        result = await checker.check()
        assert result.status == ComponentStatus.inactive

    async def test_healthy_on_200(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))
        checker = VastAIChecker(http_client=mock_client, api_key="vast-key-123")
        result = await checker.check()
        assert result.status == ComponentStatus.healthy
        assert result.metadata["status_code"] == 200

    async def test_401_is_degraded(self) -> None:
        """Auth failure = API reachable but key is wrong."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=401))
        checker = VastAIChecker(http_client=mock_client, api_key="bad-key")
        result = await checker.check()
        assert result.status == ComponentStatus.degraded
        assert result.message is not None
        assert "authentication failed" in result.message

    async def test_403_is_degraded(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=403))
        checker = VastAIChecker(http_client=mock_client, api_key="key")
        result = await checker.check()
        assert result.status == ComponentStatus.degraded

    async def test_inactive_when_whitespace_api_key(self) -> None:
        """Whitespace-only key from env misconfiguration should be treated as unconfigured."""
        checker = VastAIChecker(http_client=AsyncMock(), api_key="   ")
        result = await checker.check()
        assert result.status == ComponentStatus.inactive

    async def test_unhealthy_on_5xx(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=502))
        checker = VastAIChecker(http_client=mock_client, api_key="key")
        result = await checker.check()
        assert result.status == ComponentStatus.unhealthy
        assert result.message is not None
        assert "502" in result.message

    async def test_unhealthy_on_connection_error(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("dns failed"))
        checker = VastAIChecker(http_client=mock_client, api_key="key")
        result = await checker.check()
        assert result.status == ComponentStatus.unhealthy
        assert result.message is not None
        assert "dns failed" in result.message

    async def test_unhealthy_on_timeout(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ReadTimeout("read timed out"))
        checker = VastAIChecker(http_client=mock_client, api_key="key")
        result = await checker.check()
        assert result.status == ComponentStatus.unhealthy

    async def test_sends_auth_header(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))
        checker = VastAIChecker(http_client=mock_client, api_key="my-vast-key")
        await checker.check()
        call_args = mock_client.get.call_args
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer my-vast-key"


class TestVastAICheckerProtocolCompliance:
    def test_is_health_checker(self) -> None:
        from src.api.services.health.base import HealthChecker

        checker = VastAIChecker(http_client=AsyncMock(), api_key="key")
        assert isinstance(checker, HealthChecker)
