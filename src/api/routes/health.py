"""Health check endpoints — three-tier: liveness, readiness, admin-detailed."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from litestar import Controller, Response, get
from litestar.di import Provide

from src.api.dependencies.auth import get_current_admin_user
from src.api.schemas.health import (
    CategoryHealthResponse,
    ComponentHealthResponse,
    DetailedHealthResponse,
    GpuSessionHealthResponse,
    LivenessResponse,
    ReadinessResponse,
)
from src.api.security import auth_guard
from src.api.services.health.service import HealthService
from src.db.models import User


class HealthController(Controller):
    """Public health probes — no auth required.

    - /health/live  — liveness (Docker HEALTHCHECK, load balancers)
    - /health/ready — readiness (infra-only, controls traffic routing)
    """

    path = "/health"
    tags: Sequence[str] | None = ["health"]

    @get("/live")
    async def liveness(self) -> LivenessResponse:
        """Liveness probe. Always 200 if the process is serving HTTP."""
        return LivenessResponse(status="alive")

    @get("/ready")
    async def readiness(
        self,
        health_service: HealthService,
    ) -> Response[ReadinessResponse]:
        """Readiness probe. 200 if infra is healthy, 503 otherwise.

        Checks: PostgreSQL, Redis.
        Excludes: R2 (slow HeadBucket, not critical for accepting traffic).
        """
        is_ready, checks = await health_service.readiness()
        body = ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            checks=checks,
        )
        return Response(content=body, status_code=200 if is_ready else 503)


class AdminHealthController(Controller):
    """Admin-only detailed health — requires authentication + admin role."""

    path = "/v1/admin/health"
    tags: Sequence[str] | None = ["admin", "health"]
    guards = [auth_guard]
    dependencies = {
        "admin_user": Provide(get_current_admin_user),
    }

    @get("/")
    async def detailed(
        self,
        health_service: HealthService,
        admin_user: User,  # noqa: ARG002  — triggers admin authorization check
    ) -> DetailedHealthResponse:
        """Full system health — all categories, latencies, metadata."""
        data: dict[str, Any] = await health_service.detailed()
        return DetailedHealthResponse(
            status=str(data["status"]),
            checked_at=str(data["checked_at"]),
            infrastructure=_to_category(data["infrastructure"]),
            platform_apis=_to_category(data["platform_apis"]),
            cloud_providers={
                pid: _to_category(cat) for pid, cat in (data["cloud_providers"] or {}).items()
            },
            gpu_sessions=_to_gpu_sessions(data["gpu_sessions"]),
        )


def _to_category(cat: dict[str, Any]) -> CategoryHealthResponse:
    components = [ComponentHealthResponse(**c) for c in (cat.get("components") or [])]
    return CategoryHealthResponse(status=str(cat["status"]), components=components)


def _to_gpu_sessions(gpu: dict[str, Any]) -> GpuSessionHealthResponse:
    return GpuSessionHealthResponse(
        status=str(gpu["status"]),
        total=int(gpu["total"]),
        healthy=int(gpu["healthy"]),
        stale=int(gpu["stale"]),
        message=str(gpu.get("message", "")),
    )
