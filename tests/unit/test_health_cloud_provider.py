"""Tests for cloud provider health checkers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx

from src.api.services.health.checkers.cloud_provider import (
    CloudProviderChecker,
    GrokChecker,
)
from src.core.enums import ComponentCategory, ComponentStatus

# ---------------------------------------------------------------------------
# CloudProviderChecker base behavior
# ---------------------------------------------------------------------------


class _StubChecker(CloudProviderChecker):
    """Concrete subclass for testing the abstract base."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        probes: list[tuple[str, dict[str, str]]],
        product_id: str = "test",
    ) -> None:
        super().__init__(http_client)
        self._probes = probes
        self._product_id = product_id

    @property
    def name(self) -> str:
        return "stub_provider"

    @property
    def product_id(self) -> str:
        return self._product_id

    def _get_probe_chain(self) -> list[tuple[str, dict[str, str]]]:
        return self._probes


class TestCloudProviderCheckerBase:
    """Tests for the abstract base probe chain logic."""

    async def test_first_probe_success_short_circuits(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))
        checker = _StubChecker(
            mock_client,
            probes=[
                ("https://api.example.com/status", {"Authorization": "Bearer key"}),
                ("https://api.example.com/models", {"Authorization": "Bearer key"}),
            ],
        )
        result = await checker.check()
        assert result.status == ComponentStatus.healthy
        assert result.category == ComponentCategory.cloud_provider
        # Only first probe should be called
        assert mock_client.get.call_count == 1

    async def test_first_probe_5xx_falls_through_to_second(self) -> None:
        responses = [
            MagicMock(status_code=500),  # first probe fails
            MagicMock(status_code=200),  # second succeeds
        ]
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=responses)

        checker = _StubChecker(
            mock_client,
            probes=[
                ("https://api.example.com/status", {}),
                ("https://api.example.com/models", {}),
            ],
        )
        result = await checker.check()
        assert result.status == ComponentStatus.healthy
        assert mock_client.get.call_count == 2

    async def test_2xx_is_healthy(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))
        checker = _StubChecker(mock_client, probes=[("https://api.example.com/x", {})])
        result = await checker.check()
        assert result.status == ComponentStatus.healthy

    async def test_429_is_healthy(self) -> None:
        """Rate-limited = API is alive, transient condition."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=429))
        checker = _StubChecker(mock_client, probes=[("https://api.example.com/x", {})])
        result = await checker.check()
        assert result.status == ComponentStatus.healthy

    async def test_401_is_degraded(self) -> None:
        """Auth failure = reachable but unusable."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=401))
        checker = _StubChecker(mock_client, probes=[("https://api.example.com/x", {})])
        result = await checker.check()
        assert result.status == ComponentStatus.degraded
        assert result.message is not None
        assert "authentication failed" in result.message

    async def test_403_is_degraded(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=403))
        checker = _StubChecker(mock_client, probes=[("https://api.example.com/x", {})])
        result = await checker.check()
        assert result.status == ComponentStatus.degraded

    async def test_401_does_not_fallthrough_to_next_probe(self) -> None:
        """Auth failure returns immediately — no point trying other probes."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=401))
        checker = _StubChecker(
            mock_client,
            probes=[
                ("https://api.example.com/status", {}),
                ("https://api.example.com/models", {}),
            ],
        )
        result = await checker.check()
        assert result.status == ComponentStatus.degraded
        assert mock_client.get.call_count == 1  # did NOT try second probe

    async def test_other_4xx_is_degraded(self) -> None:
        """Non-auth 4xx (e.g. 404, 422) = reachable but unexpected response."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=404))
        checker = _StubChecker(mock_client, probes=[("https://api.example.com/x", {})])
        result = await checker.check()
        assert result.status == ComponentStatus.degraded
        assert result.message is not None
        assert "unexpected client error" in result.message

    async def test_all_probes_fail_returns_unhealthy(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        checker = _StubChecker(
            mock_client,
            probes=[
                ("https://api.example.com/a", {}),
                ("https://api.example.com/b", {}),
            ],
        )
        result = await checker.check()
        assert result.status == ComponentStatus.unhealthy
        assert result.message is not None
        assert "all probes failed" in result.message

    async def test_all_probes_5xx_returns_unhealthy(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=503))

        checker = _StubChecker(
            mock_client,
            probes=[("https://a.com/x", {}), ("https://b.com/y", {})],
        )
        result = await checker.check()
        assert result.status == ComponentStatus.unhealthy

    async def test_product_id_propagated(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))

        checker = _StubChecker(mock_client, probes=[("https://a.com", {})], product_id="vex")
        result = await checker.check()
        assert result.product_id == "vex"

    async def test_metadata_includes_probe_url(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))

        checker = _StubChecker(
            mock_client,
            probes=[("https://api.x.ai/v1/models", {})],
        )
        result = await checker.check()
        assert result.metadata["probe_url"] == "https://api.x.ai/v1/models"
        assert result.metadata["status_code"] == 200


# ---------------------------------------------------------------------------
# GrokChecker specifics
# ---------------------------------------------------------------------------


class TestGrokChecker:
    def _make_checker(self, product_id: str = "vex") -> tuple[GrokChecker, AsyncMock]:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        checker = GrokChecker(
            http_client=mock_client,
            api_base_url="https://api.x.ai",
            api_key="xai-test-key",
            product_id=product_id,
        )
        return checker, mock_client

    def test_name(self) -> None:
        checker, _ = self._make_checker()
        assert checker.name == "grok"

    def test_product_id(self) -> None:
        checker, _ = self._make_checker("synthara")
        assert checker.product_id == "synthara"

    def test_probe_chain_urls(self) -> None:
        checker, _ = self._make_checker()
        probes = checker._get_probe_chain()
        assert len(probes) >= 1
        # First probe should target /v1/models
        assert "/v1/models" in probes[0][0]
        # All probes should have auth headers
        for _, headers in probes:
            assert "Authorization" in headers
            assert "Bearer xai-test-key" in headers["Authorization"]

    async def test_healthy_on_200(self) -> None:
        checker, mock_client = self._make_checker()
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))
        result = await checker.check()
        assert result.status == ComponentStatus.healthy
        assert result.product_id == "vex"

    async def test_unhealthy_on_network_error(self) -> None:
        checker, mock_client = self._make_checker()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("unreachable"))
        result = await checker.check()
        assert result.status == ComponentStatus.unhealthy

    def test_trailing_slash_stripped_from_base_url(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        checker = GrokChecker(
            http_client=mock_client,
            api_base_url="https://api.x.ai/",
            api_key="key",
            product_id="vex",
        )
        probes = checker._get_probe_chain()
        for url, _ in probes:
            assert "//" not in url.replace("https://", "")


class TestCloudProviderCheckerProtocolCompliance:
    """Verify CloudProviderChecker subclasses satisfy the HealthChecker protocol."""

    def test_grok_is_health_checker(self) -> None:
        from src.api.services.health.base import HealthChecker

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        checker = GrokChecker(
            http_client=mock_client,
            api_base_url="https://api.x.ai",
            api_key="key",
            product_id="vex",
        )
        assert isinstance(checker, HealthChecker)
