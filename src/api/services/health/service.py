"""HealthService — orchestrates health checks for the three endpoint tiers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from src.core.enums import ComponentCategory, ComponentStatus

from .base import ComponentHealth
from .registry import HealthCheckRegistry

logger = structlog.get_logger()

# Readiness checks only these categories (fast, no external deps)
_READINESS_CATEGORIES = {ComponentCategory.infrastructure}

# Components excluded from readiness even if infrastructure
# (R2 HeadBucket can be slow, not needed for accepting traffic)
_READINESS_EXCLUDED = {"r2"}


class HealthService:
    """Orchestrates health checks across the three endpoint tiers.

    Args:
        registry: The populated HealthCheckRegistry.
        timeout_seconds: Per-check timeout (from Settings).
    """

    def __init__(
        self,
        registry: HealthCheckRegistry,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._registry = registry
        self._timeout_seconds = timeout_seconds

    async def readiness(self) -> tuple[bool, dict[str, str]]:
        """Run infrastructure-only checks for the readiness probe.

        Returns:
            Tuple of (is_ready, {component_name: status_value}).
            Uses a tight timeout (3s) and excludes slow components (R2).
        """
        results = await self._registry.check_all(
            timeout_seconds=min(self._timeout_seconds, 3.0),
            categories=_READINESS_CATEGORIES,
        )
        # Exclude R2 from readiness determination
        relevant = [r for r in results if r.name not in _READINESS_EXCLUDED]
        checks = {r.name: r.status.value for r in results}
        is_ready = all(r.status == ComponentStatus.healthy for r in relevant)
        return is_ready, checks

    async def detailed(self) -> dict[str, Any]:
        """Run all checks and build the detailed admin response.

        Returns:
            Dict structure matching DetailedHealthResponse schema.
        """
        results = await self._registry.check_all(timeout_seconds=self._timeout_seconds)
        return self._build_detailed_response(results)

    def _build_detailed_response(self, results: list[ComponentHealth]) -> dict[str, Any]:
        """Build the structured response from raw check results.

        Groups results by category, splits cloud_providers by product_id.
        Computes aggregate status per category and overall.
        """
        infra: list[dict[str, Any]] = []
        platform: list[dict[str, Any]] = []
        cloud: dict[str, list[dict[str, Any]]] = {}
        gpu_sessions: dict[str, Any] | None = None

        for r in results:
            item = self._serialize_component(r)

            if r.category == ComponentCategory.infrastructure:
                infra.append(item)
            elif r.category == ComponentCategory.platform_api:
                platform.append(item)
            elif r.category == ComponentCategory.cloud_provider:
                pid = r.product_id or "_global"
                cloud.setdefault(pid, []).append(item)
            elif r.category == ComponentCategory.gpu_session:
                gpu_sessions = {
                    "status": r.status.value,
                    "total": r.metadata.get("total", 0),
                    "healthy": r.metadata.get("healthy", 0),
                    "stale": r.metadata.get("stale", 0),
                    "message": r.message,
                }

        # Default gpu_sessions if no checker registered yet (Phase 1)
        if gpu_sessions is None:
            gpu_sessions = {
                "status": ComponentStatus.inactive.value,
                "total": 0,
                "healthy": 0,
                "stale": 0,
                "message": "gpu session monitoring not enabled",
            }

        # Aggregate statuses
        all_statuses = [r.status for r in results]
        overall = self._aggregate_status(all_statuses)

        return {
            "status": overall.value,
            "checked_at": datetime.now(UTC).isoformat(),
            "infrastructure": {
                "status": self._aggregate_status(
                    [r.status for r in results if r.category == ComponentCategory.infrastructure]
                ).value,
                "components": infra,
            },
            "platform_apis": {
                "status": self._aggregate_status(
                    [r.status for r in results if r.category == ComponentCategory.platform_api]
                ).value
                if any(r.category == ComponentCategory.platform_api for r in results)
                else ComponentStatus.inactive.value,
                "components": platform,
            },
            "cloud_providers": {
                pid: {
                    "status": self._aggregate_status(
                        [
                            r.status
                            for r in results
                            if r.category == ComponentCategory.cloud_provider
                            and (r.product_id or "_global") == pid
                        ]
                    ).value,
                    "components": components,
                }
                for pid, components in cloud.items()
            },
            "gpu_sessions": gpu_sessions,
        }

    @staticmethod
    def _serialize_component(r: ComponentHealth) -> dict[str, Any]:
        """Serialize a single ComponentHealth to dict."""
        result: dict[str, Any] = {
            "name": r.name,
            "status": r.status.value,
            "latency_ms": r.latency_ms,
        }
        if r.message:
            result["message"] = r.message
        if r.metadata:
            result["metadata"] = r.metadata
        return result

    @staticmethod
    def _aggregate_status(statuses: list[ComponentStatus]) -> ComponentStatus:
        """Compute the worst-case aggregate status.

        Priority: unhealthy > unknown > degraded > inactive > healthy.
        Empty list → healthy (vacuous truth — nothing is broken).
        """
        if not statuses:
            return ComponentStatus.healthy

        priority = {
            ComponentStatus.unhealthy: 4,
            ComponentStatus.unknown: 3,
            ComponentStatus.degraded: 2,
            ComponentStatus.inactive: 1,
            ComponentStatus.healthy: 0,
        }
        worst = max(statuses, key=lambda s: priority.get(s, 0))
        return worst
