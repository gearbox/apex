"""Health check endpoints — three-tier: liveness, readiness, admin-detailed."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from datetime import datetime
from typing import Any

import msgspec
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
from src.api.services.health.service import HealthService
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
        """SSE stream of real-time health snapshots.

        Admin-only. Streams health.snapshot events at the configured
        interval. The frontend admin panel subscribes here using fetch()
        with ReadableStream (supports Authorization headers, unlike EventSource).

        Unlike the user SSE endpoint, this uses direct Redis Pub/Sub
        subscription (not the ticket-based EventBus pattern) because:
        - Admin auth is already enforced by the controller guard
        - Health snapshots go to a dedicated channel, not user channels
        - Simpler pattern for a single-purpose admin stream

        Falls back to periodic polling if Redis is not configured.
        """
        settings = get_settings()

        async def event_generator() -> AsyncGenerator[dict[str, str]]:
            heartbeat_interval = 15

            if settings.redis_url is not None:
                from src.core.redis import get_redis_client

                client = get_redis_client()
                pubsub = client.pubsub()
                try:
                    await pubsub.subscribe("health:stream")

                    while True:
                        try:
                            message = await asyncio.wait_for(
                                pubsub.get_message(
                                    ignore_subscribe_messages=True,
                                    timeout=float(heartbeat_interval),
                                ),
                                timeout=float(heartbeat_interval) + 1,
                            )
                            if message is not None and message["type"] == "message":
                                data = message["data"]
                                yield {
                                    "event": "health.snapshot",
                                    "data": data if isinstance(data, str) else data.decode(),
                                }
                            else:
                                yield {"comment": "keepalive"}
                        except TimeoutError:
                            yield {"comment": "keepalive"}
                except asyncio.CancelledError:
                    pass
                finally:
                    await pubsub.unsubscribe("health:stream")
                    await pubsub.aclose()  # type: ignore[no-untyped-call]
            else:
                # Fallback: poll HealthService directly
                while True:
                    try:
                        data_dict = await health_service.check_all_and_build()
                        yield {
                            "event": "health.snapshot",
                            "data": msgspec.json.encode(data_dict).decode(),
                        }
                    except Exception:
                        yield {"comment": "error"}
                    await asyncio.sleep(settings.health_snapshot_interval_seconds)

        return ServerSentEvent(content=event_generator())

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
        """
        parsed_after = datetime.fromisoformat(after) if after else None
        parsed_before = datetime.fromisoformat(before) if before else None
        clamped_limit = min(max(limit, 1), 1440)

        repo = HealthSnapshotRepository(session)
        snapshots = await repo.list_range(
            after=parsed_after,
            before=parsed_before,
            limit=clamped_limit,
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
