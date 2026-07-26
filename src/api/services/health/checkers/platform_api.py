"""Platform API health checkers — Vast.ai and similar infrastructure APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.api.services.health.base import ComponentHealth
from src.core.enums import ComponentCategory, ComponentStatus

if TYPE_CHECKING:
    import httpx

logger = structlog.get_logger()


class VastAIChecker:
    """Health checker for the Vast.ai API.

    Checks whether the Vast.ai API is reachable and our API key is valid.
    This is a prerequisite for deploying/managing GPU nodes (Aisha).

    Probe: GET https://console.vast.ai/api/v0/users/current
    - Returns user info if API key is valid
    - Proves API is up and we can interact with it

    If VASTAI_API_KEY is not configured, reports inactive (not unhealthy).
    """

    name = "vastai_api"
    category = ComponentCategory.platform_api
    product_id = None  # global — not product-scoped
    gates_readiness = True  # not in _READINESS_CATEGORIES anyway (issue #142 G2)

    # Vast.ai API base URL
    _VASTAI_API_BASE = "https://console.vast.ai/api/v0"

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        api_key: str | None,
    ) -> None:
        self._client = http_client
        self._api_key = api_key

    async def check(self) -> ComponentHealth:
        if not self._api_key or not self._api_key.strip():
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.inactive,
                latency_ms=0.0,
                message="vastai api key not configured",
            )

        try:
            resp = await self._client.get(
                f"{self._VASTAI_API_BASE}/users/current",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=10.0,
            )
            if resp.status_code < 400 or resp.status_code == 429:
                return ComponentHealth(
                    name=self.name,
                    category=self.category,
                    status=ComponentStatus.healthy,
                    latency_ms=0.0,
                    metadata={"status_code": resp.status_code},
                )
            if resp.status_code in (401, 403):
                logger.warning(
                    "health.vastai.auth_failed",
                    status_code=resp.status_code,
                )
                return ComponentHealth(
                    name=self.name,
                    category=self.category,
                    status=ComponentStatus.degraded,
                    latency_ms=0.0,
                    message=f"authentication failed (HTTP {resp.status_code}) — check API key",
                    metadata={"status_code": resp.status_code},
                )
            if resp.status_code < 500:
                logger.warning(
                    "health.vastai.client_error",
                    status_code=resp.status_code,
                )
                return ComponentHealth(
                    name=self.name,
                    category=self.category,
                    status=ComponentStatus.degraded,
                    latency_ms=0.0,
                    message=f"unexpected client error (HTTP {resp.status_code})",
                    metadata={"status_code": resp.status_code},
                )
            logger.warning(
                "health.vastai.server_error",
                status_code=resp.status_code,
            )
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unhealthy,
                latency_ms=0.0,
                message=f"vast.ai API returned HTTP {resp.status_code}",
                metadata={"status_code": resp.status_code},
            )
        except Exception as exc:
            logger.warning(
                "health.vastai.probe_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unhealthy,
                latency_ms=0.0,
                message=str(exc),
            )
