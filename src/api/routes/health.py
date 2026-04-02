"""Health check endpoints — three-tier: liveness, readiness, admin-detailed."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from litestar import Controller, Response, get
from litestar.di import Provide
from litestar.response import ServerSentEvent
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_admin_user
from src.api.schemas.health import (
    CategoryHealthResponse,
    ComponentHealthResponse,
    DetailedHealthResponse,
    GpuSessionHealthResponse,
    HealthSnapshotResponse,
    LivenessResponse,
    ReadinessResponse,
)
from src.api.security import auth_guard
from src.api.services.health.history import parse_history_params
from src.api.services.health.service import HealthService
from src.api.services.health.stream import health_sse_generator
from src.core.config import get_settings
from src.db.models import User
from src.db.repositories.health import HealthSnapshotRepository


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

    @get("/stream")
    async def stream(
        self,
        health_service: HealthService,
        admin_user: User,  # noqa: ARG002  — triggers admin authorization check
    ) -> ServerSentEvent:
        """SSE stream of real-time health snapshots for the admin panel."""
        settings = get_settings()
        return ServerSentEvent(
            content=health_sse_generator(
                health_service=health_service,
                settings=settings,
            )
        )

    @get("/history")
    async def history(
        self,
        session: AsyncSession,
        admin_user: User,  # noqa: ARG002  — triggers admin authorization check
        after: str | None = None,
        before: str | None = None,
        limit: int = 60,
    ) -> list[HealthSnapshotResponse]:
        """Paginated historical snapshots for dashboard charts.

        Args:
            session: Request-scoped DB session.
            admin_user: Admin user (authorization check).
            after: ISO 8601 datetime — only snapshots after this time.
            before: ISO 8601 datetime — only snapshots before this time.
            limit: Maximum results (default 60, max 1440).

        Returns:
            List of snapshots ordered by checked_at DESC.

        Raises:
            ValidationException: If after/before are not valid ISO 8601 datetimes (400).
        """
        query = parse_history_params(after=after, before=before, limit=limit)
        repo = HealthSnapshotRepository(session)
        snapshots = await repo.list_range(
            after=query.after,
            before=query.before,
            limit=query.limit,
        )
        return [
            HealthSnapshotResponse(
                checked_at=s.checked_at.isoformat(),
                overall_status=s.overall_status,
                snapshot_data=s.snapshot_data,
            )
            for s in snapshots
        ]


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
