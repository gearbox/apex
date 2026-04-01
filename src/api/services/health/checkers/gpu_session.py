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

        # Probe all active/stale sessions concurrently
        probe_tasks = [self._probe_node(s.node_host, s.node_port) for s in active_sessions]
        probe_results = await asyncio.gather(*probe_tasks, return_exceptions=True)

        healthy_count = 0
        stale_count = 0
        newly_stale_ids: list[UUID] = []

        for session, result in zip(active_sessions, probe_results, strict=True):
            if result is True:
                healthy_count += 1
                # If previously marked stale but now reachable, clear staleness
                if session.stale_detected_at is not None:
                    await self._clear_stale(session.id)
                    logger.info(
                        "health.gpu_session.recovered",
                        session_id=str(session.id),
                        node_host=session.node_host,
                    )
            else:
                stale_count += 1
                # Only mark if not already stale
                if session.stale_detected_at is None:
                    newly_stale_ids.append(session.id)

        # Batch-mark newly stale sessions
        if newly_stale_ids:
            await self._mark_stale(newly_stale_ids)
            logger.warning(
                "health.gpu_sessions.stale_detected",
                count=len(newly_stale_ids),
                session_ids=[str(sid) for sid in newly_stale_ids],
            )

        total = len(active_sessions)
        status = (
            ComponentStatus.healthy
            if stale_count == 0
            else ComponentStatus.degraded
            if healthy_count > 0
            else ComponentStatus.unhealthy
        )

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
        """Query all sessions that should be probed for reachability.

        Includes both 'active' and 'stale' sessions:
        - active: normal health check
        - stale: check if node has recovered (self-healing)
        """
        async with self._session_factory() as session:
            stmt = select(GpuSession).where(
                GpuSession.status.in_(
                    [
                        GpuSessionStatus.active.value,
                        GpuSessionStatus.stale.value,
                    ]
                )
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

    async def _clear_stale(self, session_id: UUID) -> None:
        """Clear staleness for a session that has recovered.

        Transitions status back to 'active' and resets stale_detected_at.
        This handles transient network blips — if the node comes back,
        we don't want it stuck in 'stale' forever.
        """
        async with self._session_factory() as session:
            stmt = (
                update(GpuSession)
                .where(GpuSession.id == session_id)
                .values(
                    stale_detected_at=None,
                    stale_notified=False,
                    status=GpuSessionStatus.active.value,
                )
            )
            await session.execute(stmt)
            await session.commit()
