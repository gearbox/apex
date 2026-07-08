"""GPU session reconciler — detects stale Aisha nodes.

Probes all active GPU sessions in the database. If a session's
ComfyUI tunnel endpoint is unreachable past the grace window, it's
flagged as stale. Stale sessions that become reachable again recover
automatically.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from sqlalchemy import select, update

from src.api.services.generation.tunnel_validation import (
    InvalidTunnelHostnameError,
    session_comfyui_base_url,
)
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
    2. Probes each node's ComfyUI tunnel endpoint concurrently
    3. Marks unreachable sessions as 'stale' after the grace window expires
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
        *,
        tunnel_domain: str,
        tunnel_prefix: str | None,
        grace_seconds: int,
    ) -> None:
        self._session_factory = session_factory
        self._client = http_client
        self._tunnel_domain = tunnel_domain
        self._tunnel_prefix = tunnel_prefix
        self._grace_seconds = grace_seconds

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
        # _probe_session catches all exceptions and returns bool — no need for
        # return_exceptions=True, so probe_results is cleanly list[bool].
        probe_tasks = [self._probe_session(s) for s in active_sessions]
        probe_results: list[bool] = await asyncio.gather(*probe_tasks)

        now = datetime.now(UTC)

        # Classify results — collect IDs for batch DB writes
        healthy_count = 0
        stale_count = 0
        reachable_ids: list[UUID] = []
        newly_stale_ids: list[UUID] = []
        recovered_ids: list[UUID] = []

        for session, reachable in zip(active_sessions, probe_results, strict=True):
            if reachable:
                healthy_count += 1
                reachable_ids.append(session.id)
                if session.stale_detected_at is not None:
                    recovered_ids.append(session.id)
            else:
                stale_count += 1
                if session.stale_detected_at is None:
                    # Defensive: created_at has a server_default so it should
                    # never be None in practice, but fall back to `now` (elapsed=0,
                    # within grace) rather than blow up if it somehow is.
                    last_seen = session.last_reachable_at or session.created_at or now
                    elapsed = (now - last_seen).total_seconds()
                    if elapsed >= self._grace_seconds:
                        newly_stale_ids.append(session.id)
                    else:
                        logger.info(
                            "health.gpu_session.within_grace",
                            session_id=str(session.id),
                            elapsed_seconds=elapsed,
                            grace_seconds=self._grace_seconds,
                        )

        # Batch DB writes — one UPDATE each, not per-session.
        # Pass `now` so the decision timestamp and stored values are aligned.
        if reachable_ids:
            await self._update_last_reachable_batch(reachable_ids, now=now)

        if newly_stale_ids:
            await self._mark_stale(newly_stale_ids, now=now)
            logger.warning(
                "health.gpu_sessions.stale_detected",
                count=len(newly_stale_ids),
                session_ids=[str(sid) for sid in newly_stale_ids],
            )

        if recovered_ids:
            await self._clear_stale_batch(recovered_ids, now=now)
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

    async def _probe_session(self, session: GpuSession) -> bool:
        """Probe ComfyUI /object_info on a session's tunnel endpoint.

        Returns True if reachable, False otherwise. Invalid or missing
        tunnel_hostname logs health.gpu_session.invalid_hostname and returns
        False — subject to the same grace period as a network failure.
        """
        try:
            base_url = session_comfyui_base_url(
                session.tunnel_hostname,
                tunnel_domain=self._tunnel_domain,
                allowed_prefix=self._tunnel_prefix,
            )
        except InvalidTunnelHostnameError:
            logger.warning(
                "health.gpu_session.invalid_hostname",
                session_id=str(session.id),
                tunnel_hostname=session.tunnel_hostname,
            )
            return False

        try:
            resp = await self._client.get(
                f"{base_url}{_COMFYUI_PROBE_PATH}",
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.debug(
                "health.gpu_session.probe_failed",
                session_id=str(session.id),
                tunnel_hostname=session.tunnel_hostname,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return False
        else:
            return resp.status_code == 200

    async def _update_last_reachable_batch(
        self,
        session_ids: list[UUID],
        *,
        now: datetime,
    ) -> None:
        """Set last_reachable_at = now for all reachable sessions.

        ``now`` must be the same timestamp used for grace-window calculations so
        the decision time and stored value stay aligned across a single run.
        """
        async with self._session_factory() as session:
            stmt = (
                update(GpuSession)
                .where(GpuSession.id.in_(session_ids))
                .values(last_reachable_at=now)
            )
            await session.execute(stmt)
            await session.commit()

    async def _mark_stale(self, session_ids: list[UUID], *, now: datetime) -> None:
        """Mark sessions as stale in the DB.

        Sets stale_detected_at to ``now``, but only for sessions where it is
        still NULL (idempotent — won't overwrite an earlier detection).
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
                    stale_detected_at=now,
                    status=GpuSessionStatus.stale.value,
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def _clear_stale_batch(self, session_ids: list[UUID], *, now: datetime) -> None:
        """Batch-clear staleness for sessions that have recovered.

        Transitions status back to 'active', resets stale tracking, and
        stamps last_reachable_at to ``now``. Handles transient network blips
        — if nodes come back, they don't stay stuck in 'stale'.
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
                    last_reachable_at=now,
                )
            )
            await session.execute(stmt)
            await session.commit()
