"""Token-revocation backend health checker (issue #142 A2).

The fail-open posture of TokenRevocationService rests entirely on an
operator noticing that Redis is unavailable — before this checker, the only
signal was a log line (`authrev.backend_unavailable`/`authrev.circuit_opened`).
RedisChecker's plain PING doesn't cover this: a circuit breaker can be open
(actively short-circuiting every is_revoked call) even while Redis itself
has recovered but the half-open probe hasn't landed yet, and PING carries no
write-failure counter. This checker surfaces both directly from the service.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.services.health.base import ComponentHealth
from src.core.enums import ComponentCategory, ComponentStatus

if TYPE_CHECKING:
    from src.api.services.token_revocation import TokenRevocationService


class TokenRevocationChecker:
    """Surfaces TokenRevocationService's fail-open posture as health data."""

    name = "token_revocation"
    category = ComponentCategory.infrastructure
    product_id = None

    def __init__(self, service: TokenRevocationService) -> None:
        self._service = service

    async def check(self) -> ComponentHealth:
        metadata: dict[str, object] = {"failed_write_count": self._service.failed_write_count}

        if not self._service.enabled:
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.inactive,
                latency_ms=0.0,
                message="Redis not configured — revocation degraded to refresh-token-only",
                metadata=metadata,
            )

        if self._service.circuit_open:
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unhealthy,
                latency_ms=0.0,
                message="Circuit breaker open — is_revoked() is failing open on every request",
                metadata=metadata,
            )

        return ComponentHealth(
            name=self.name,
            category=self.category,
            status=ComponentStatus.healthy,
            latency_ms=0.0,
            metadata=metadata,
        )
