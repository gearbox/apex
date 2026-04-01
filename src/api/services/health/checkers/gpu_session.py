"""GPU session reconciler — detects stale Aisha nodes.

Probes all active GPU sessions in the database. If a session's
ComfyUI endpoint is unreachable, it's flagged as stale.
This is detection only — recovery is a separate future worker.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from sqlalchemy import select, update

from src.core.enums import ComponentCategory, ComponentStatus, GpuSessionStatus
from src.db.models.gpu_session import GpuSession

from ..base import ComponentHealth

if TYPE_CHECKING:
    import httpx
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = structlog.get_logger()

# ComfyUI health endpoint — lightweight, returns node info
_COMFYUI_PROBE_PATH = "/object_info"
_PROBE_TIMEOUT_SECONDS = 10.0

# Max sessions to probe per reconciliation cycle.
# In practice active+stale session count should be low (tens, not thousands),
# but this prevents runaway memory/concurrency if something goes wrong.
_MAX_PROBE_SESSIONS = 200


class GpuSessionReconciler:
    """Reconcile DB-claimed active GPU sessions against actual node reachability.

    This is a HealthChecker that:
    1. Queries all gpu_sessions with status = 'active' or 'stale'
    2. Probes each node's ComfyUI endpoint concurrently
    3. Marks unreachable sessions as 'stale' in the DB
    4. Clears staleness for sessions that have recovered
    5. Returns an aggregate ComponentHealth result

    The reconciler has a DB write side-effect (stale marking). This is
    intentional — staleness detection must be persisted for the future
    recovery worker to act on.

    The checker satisfies the HealthChecker protocol:
    - name: "gpu_sessions"
    - category: ComponentCategory.gpu_session
    - product_id: None (global — reconciles across all products)
    """

    name = "gpu_sessions"
    category = ComponentCategory.gpu_session
    product_id = None

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        http_client: httpx.AsyncClient,
    ) -> None:
        self._session_factory = session_factory
        self._client = http_client

    async def check(self) -> ComponentHealth:
        """Run reconciliation. Returns aggregate status."""
        try:
            return await self._reconcile()
        except Exception as exc:
            logger.exception("health.gpu_sessions.reconcile_error")
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unknown,
                latency_ms=0.0,
                message=f"reconciliation failed: {exc}",
                metadata={"total": 0, "healthy": 0, "stale": 0},
            )

    async def _reconcile(self) -> ComponentHealth:
        """Core reconciliation logic."""
        active_sessions = await self._get_active_sessions()

        if not active_sessions:
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.inactive,
                latency_ms=0.0,
                message="no active gpu sessions",
                metadata={"total": 0, "healthy": 0, "stale": 0},
            )

        # Probe all sessions concurrently.
        # _probe_node catches all exceptions and returns bool — no need for
        # return_exceptions=True, so probe_results is cleanly list[bool].
        probe_tasks = [self._probe_node(s.node_host, s.node_port) for s in active_sessions]
        probe_results: list[bool] = await asyncio.gather(*probe_tasks)

        # Classify results — collect IDs for batch DB writes
        healthy_count = 0
        stale_count = 0
        newly_stale_ids: list[UUID] = []
        recovered_ids: list[UUID] = []

        for session, reachable in zip(active_sessions, probe_results, strict=True):
            if reachable:
                healthy_count += 1
                if session.stale_detected_at is not None:
                    recovered_ids.append(session.id)
            else:
                stale_count += 1
                if session.stale_detected_at is None:
                    newly_stale_ids.append(session.id)

        # Batch DB writes — one UPDATE each, not per-session
        if newly_stale_ids:
            await self._mark_stale(newly_stale_ids)
            logger.warning(
                "health.gpu_sessions.stale_detected",
                count=len(newly_stale_ids),
                session_ids=[str(sid) for sid in newly_stale_ids],
            )

        if recovered_ids:
            await self._clear_stale_batch(recovered_ids)
            logger.info(
                "health.gpu_sessions.recovered",
                count=len(recovered_ids),
                session_ids=[str(sid) for sid in recovered_ids],
            )

        # Aggregate status — explicit if/elif for clarity
        total = len(active_sessions)
        if stale_count == 0:
            status = ComponentStatus.healthy
        elif healthy_count > 0:
            status = ComponentStatus.degraded
        else:
            status = ComponentStatus.unhealthy

        return ComponentHealth(
            name=self.name,
            category=self.category,
            status=status,
            latency_ms=0.0,
            message=f"{healthy_count}/{total} nodes reachable, {stale_count} stale",
            metadata={
                "total": total,
                "healthy": healthy_count,
                "stale": stale_count,
            },
        )

    async def _get_active_sessions(self) -> list[GpuSession]:
        """Query sessions that should be probed for reachability.

        Includes both 'active' and 'stale' sessions (stale for self-healing).
        Capped at _MAX_PROBE_SESSIONS to bound probe concurrency and memory.
        Ordered by created_at DESC so newest sessions are probed first.
        """
        async with self._session_factory() as session:
            stmt = (
                select(GpuSession)
                .where(
                    GpuSession.status.in_(
                        [
                            GpuSessionStatus.active.value,
                            GpuSessionStatus.stale.value,
                        ]
                    )
                )
                .order_by(GpuSession.created_at.desc())
                .limit(_MAX_PROBE_SESSIONS)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def _probe_node(self, host: str | None, port: int | None) -> bool:
        """Probe ComfyUI /object_info on a GPU node.

        Returns True if reachable, False otherwise.
        Handles missing host/port gracefully (returns False).
        """
        if not host or not port:
            logger.warning(
                "health.gpu_session.missing_endpoint",
                host=host,
                port=port,
            )
            return False

        try:
            resp = await self._client.get(
                f"http://{host}:{port}{_COMFYUI_PROBE_PATH}",
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
            return resp.status_code == 200
        except Exception as exc:
            logger.debug(
                "health.gpu_session.probe_failed",
                host=host,
                port=port,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return False

    async def _mark_stale(self, session_ids: list[UUID]) -> None:
        """Mark sessions as stale in the DB.

        Sets stale_detected_at to now, but only for sessions where
        it's still NULL (idempotent — won't overwrite an earlier detection).
        Also transitions status from 'active' to 'stale'.
        """
        async with self._session_factory() as session:
            stmt = (
                update(GpuSession)
                .where(
                    GpuSession.id.in_(session_ids),
                    GpuSession.stale_detected_at.is_(None),
                )
                .values(
                    stale_detected_at=datetime.now(UTC),
                    status=GpuSessionStatus.stale.value,
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def _clear_stale_batch(self, session_ids: list[UUID]) -> None:
        """Batch-clear staleness for sessions that have recovered.

        Transitions status back to 'active' and resets stale tracking.
        Handles transient network blips — if nodes come back, they
        don't stay stuck in 'stale'.
        """
        if not session_ids:
            return
        async with self._session_factory() as session:
            stmt = (
                update(GpuSession)
                .where(GpuSession.id.in_(session_ids))
                .values(
                    stale_detected_at=None,
                    stale_notified=False,
                    status=GpuSessionStatus.active.value,
                )
            )
            await session.execute(stmt)
            await session.commit()
