"""Cloud provider health checkers.

Strategy per provider:
1. Try the dedicated status/health endpoint (if provider exposes one)
2. Fall back to a lightweight authenticated endpoint (e.g. list-models)
3. Report unhealthy if all probes fail

Subclasses implement _get_probe_chain() to declare ordered (url, headers) pairs.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import structlog

from src.api.services.health.base import ComponentHealth
from src.core.enums import ComponentCategory, ComponentStatus

if TYPE_CHECKING:
    import httpx

logger = structlog.get_logger()


class CloudProviderChecker(abc.ABC):
    """Abstract base for HTTP-based cloud provider health checkers.

    Subclasses only need to implement:
    - name (property)
    - product_id (property)
    - _get_probe_chain() — ordered list of (url, headers) probe targets

    The base handles:
    - Iterating through the probe chain with short timeouts
    - Status classification (healthy: 2xx/3xx/429; degraded: 4xx auth/client errors; unhealthy: 5xx/connection failures)
    - Error logging and fallback
    """

    category = ComponentCategory.cloud_provider
    # Not in _READINESS_CATEGORIES anyway (category != infrastructure), but
    # every checker must declare this explicitly for HealthChecker protocol
    # conformance (issue #142 G2).
    gates_readiness = True

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._client = http_client

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def product_id(self) -> str: ...

    @abc.abstractmethod
    def _get_probe_chain(self) -> list[tuple[str, dict[str, str]]]:
        """Return ordered list of (url, headers) to probe.

        The first successful probe (status < 500) short-circuits.
        Each probe gets a 5-second timeout.

        Returns:
            List of (url, headers) tuples. First entry = preferred probe.
        """
        ...

    async def check(self) -> ComponentHealth:
        """Execute the probe chain. Return healthy if any probe succeeds."""
        probes = self._get_probe_chain()
        last_error = ""

        for url, headers in probes:
            try:
                resp = await self._client.get(url, headers=headers, timeout=5.0)
                if resp.status_code < 400 or resp.status_code == 429:
                    return ComponentHealth(
                        name=self.name,
                        category=self.category,
                        status=ComponentStatus.healthy,
                        latency_ms=0.0,
                        product_id=self.product_id,
                        metadata={"probe_url": url, "status_code": resp.status_code},
                    )
                if resp.status_code in (401, 403):
                    return ComponentHealth(
                        name=self.name,
                        category=self.category,
                        status=ComponentStatus.degraded,
                        latency_ms=0.0,
                        message=f"authentication failed (HTTP {resp.status_code}) — check API key",
                        product_id=self.product_id,
                        metadata={"probe_url": url, "status_code": resp.status_code},
                    )
                # 4xx other than 401/403/429 — API is reachable but returning client errors.
                # Treat as degraded rather than falling through to next probe.
                if resp.status_code < 500:
                    return ComponentHealth(
                        name=self.name,
                        category=self.category,
                        status=ComponentStatus.degraded,
                        latency_ms=0.0,
                        message=f"unexpected client error (HTTP {resp.status_code})",
                        product_id=self.product_id,
                        metadata={"probe_url": url, "status_code": resp.status_code},
                    )
                last_error = f"HTTP {resp.status_code} from {url}"
                logger.warning(
                    "health.cloud_provider.probe_failed",
                    name=self.name,
                    url=url,
                    status_code=resp.status_code,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "health.cloud_provider.probe_error",
                    name=self.name,
                    url=url,
                    error=str(exc),
                )

        return ComponentHealth(
            name=self.name,
            category=self.category,
            status=ComponentStatus.unhealthy,
            latency_ms=0.0,
            message=f"all probes failed: {last_error}",
            product_id=self.product_id,
        )


class GrokChecker(CloudProviderChecker):
    """Health checker for xAI Grok API.

    Probe chain:
    1. REST API models endpoint: GET {base_url}/v1/models
       - Lightweight, authenticated, always available on REST API
    2. API key info endpoint: GET {base_url}/v1/api-key
       - Fallback if /v1/models is unavailable

    Note: The GrokClient uses gRPC (xai-sdk), but the health check uses
    REST HTTP to avoid the heavier gRPC setup. The REST and gRPC APIs
    share the same backend — if REST is up, gRPC is up.
    """

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        api_base_url: str,
        api_key: str,
        product_id: str,
    ) -> None:
        super().__init__(http_client)
        self._api_base_url = api_base_url.rstrip("/")
        self._api_key = api_key
        self._product_id = product_id

    @property
    def name(self) -> str:
        return "grok"

    @property
    def product_id(self) -> str:
        return self._product_id

    def _get_probe_chain(self) -> list[tuple[str, dict[str, str]]]:
        auth_headers = {"Authorization": f"Bearer {self._api_key}"}
        return [
            # Primary: lightweight models list
            (f"{self._api_base_url}/v1/models", auth_headers),
            # Fallback: API root (may return 404 but proves reachability)
            (f"{self._api_base_url}/v1/api-key", auth_headers),
        ]
