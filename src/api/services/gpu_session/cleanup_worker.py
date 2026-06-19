"""Background worker: reconciles CF tunnels and Vast.ai instances against gpu_sessions.

Tunnel sweep: detects tunnels that are (a) ours (by name prefix), (b) older than the
grace period, (c) not referenced by any non-terminal gpu_sessions row, and deletes them.

Instance sweep: detects sessions in terminal state with potentially-undestroyed Vast.ai
instances (vastai_instance_id IS NOT NULL AND vastai_instance_destroyed_at IS NULL),
verifies each against the Vast.ai API, and destroys any that are still running.

Both sweeps run once every ``settings.orphaned_tunnel_cleanup_interval_minutes`` (default 60).
Intentionally slow — orphans are not an emergency.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from src.api.services.cloudflare.schemas import TunnelListEntry
from src.api.services.vastai.exceptions import InstanceNotFoundError, VastAIRateLimitError
from src.core.enums import TERMINAL_GPU_SESSION_STATUSES, GpuSessionStatus
from src.db.repositories.gpu_session import GpuSessionRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.cloudflare.client import CloudflareTunnelClient
    from src.api.services.vastai.client import VastAIClient
    from src.core.config import Settings
    from src.db.models.gpu_session import GpuSession

logger = structlog.get_logger(__name__)

# Statuses whose tunnels are still actively claimed (including stopping/stale)
_LIVE_STATUSES = (
    GpuSessionStatus.pending,
    GpuSessionStatus.provisioning,
    GpuSessionStatus.active,
    GpuSessionStatus.paused,
    GpuSessionStatus.resuming,
    GpuSessionStatus.stale,
    GpuSessionStatus.stopping,
)


class OrphanedTunnelCleanupWorker:
    """Reconciles CF tunnels and Vast.ai instances against the gpu_sessions table.

    Tunnel sweep: detects tunnels that are (a) ours (by name prefix), (b) older than
    the grace period, (c) not referenced by any non-terminal gpu_sessions row,
    and deletes them.

    Instance sweep: finds terminal sessions with potentially-undestroyed Vast.ai
    instances, verifies via API, and destroys any still-running instances.

    Runs once every ``settings.orphaned_tunnel_cleanup_interval_minutes``
    (default 60). Intentionally slow — orphans are not an emergency.
    """

    _TUNNEL_NAME_PREFIX = "gpu-session-"

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        cf_client: CloudflareTunnelClient,
        vastai_client: VastAIClient,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._cf = cf_client
        self._vastai = vastai_client
        self._settings = settings
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the cleanup worker."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "gpu_orphan_cleanup_worker.started",
            interval_minutes=self._settings.orphaned_tunnel_cleanup_interval_minutes,
        )

    async def stop(self) -> None:
        """Stop the cleanup worker."""
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("gpu_orphan_cleanup_worker.stopped")

    async def _run_loop(self) -> None:
        """Main periodic loop."""
        while self._running:
            try:
                await self._sweep_once()
            except Exception:
                logger.exception("gpu_orphan_cleanup_worker.sweep_error")
            await asyncio.sleep(self._settings.orphaned_tunnel_cleanup_interval_minutes * 60)

    async def _sweep_once(self) -> None:
        """One cleanup sweep: orphaned tunnels AND orphaned Vast.ai instances."""
        await self._sweep_tunnels()
        await self._sweep_instances()

    async def _sweep_tunnels(self) -> None:
        """Find and delete CF tunnels not claimed by any live session."""
        started = time.monotonic()

        # 1. Pull all CF tunnels we own (by name prefix, not deleted).
        # Client-side filter on deleted_at is defense-in-depth: list_tunnels
        # already sends is_deleted=false to Cloudflare, but we don't want to
        # re-delete (and log noise for) tunnels that slip through.
        all_tunnels = await self._cf.list_tunnels(name_prefix=self._TUNNEL_NAME_PREFIX)
        session_tunnels = [
            t
            for t in all_tunnels
            if t.name.startswith(self._TUNNEL_NAME_PREFIX) and t.deleted_at is None
        ]

        # 2. Pull all non-terminal sessions' tunnel IDs in one shot.
        #    Terminal sessions (stopped/failed) don't claim their tunnels —
        #    their tunnels should already have been deleted.
        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            live_sessions = await repo.list_by_status(*_LIVE_STATUSES)

        claimed_tunnel_ids = {s.cf_tunnel_id for s in live_sessions if s.cf_tunnel_id}

        # 3. Grace period: skip tunnels younger than this cutoff
        cutoff = datetime.now(UTC) - timedelta(
            minutes=self._settings.orphaned_tunnel_cleanup_grace_period_minutes
        )

        candidates = [
            t
            for t in session_tunnels
            if t.id not in claimed_tunnel_ids and t.created_at is not None and t.created_at < cutoff
        ]

        # 4. Delete each orphan (best-effort, log all outcomes)
        for tunnel in candidates:
            try:
                await self._delete_orphan(tunnel)
                logger.info(
                    "gpu_session.orphan_cleanup.deleted",
                    tunnel_id=tunnel.id,
                    tunnel_name=tunnel.name,
                )
            except Exception:
                logger.exception(
                    "gpu_session.orphan_cleanup.delete_failed",
                    tunnel_id=tunnel.id,
                )

        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "gpu_session.orphan_cleanup.done",
            total_tunnels=len(session_tunnels),
            claimed=len(claimed_tunnel_ids),
            candidates=len(candidates),
            duration_ms=elapsed_ms,
        )

    async def _delete_orphan(self, tunnel: TunnelListEntry) -> None:
        """Look up DNS record for tunnel and delete both.

        If DNS record is not found, deletes the tunnel only and logs a warning.
        Hostname is reconstructed from the tunnel name:
        "gpu-session-{short_id}" → "{short_id}.{aisha_cf_tunnel_domain}".
        """
        short_id = tunnel.name[len(self._TUNNEL_NAME_PREFIX) :]
        hostname = f"{short_id}.{self._settings.aisha_cf_tunnel_domain}"

        dns_record_id = await self._cf.find_dns_record_by_hostname(hostname)

        if dns_record_id is not None:
            await self._cf.delete_session_tunnel(tunnel.id, dns_record_id)
        else:
            logger.warning(
                "gpu_session.orphan_cleanup.dns_record_not_found",
                tunnel_id=tunnel.id,
                hostname=hostname,
            )
            await self._cf.delete_tunnel(tunnel.id)

    async def _sweep_instances(self) -> None:
        """Find terminal sessions with potentially-undestroyed Vast.ai instances.

        Selects: status IN (stopped, failed) AND vastai_instance_id IS NOT NULL
        AND vastai_instance_destroyed_at IS NULL AND created_at > horizon
        AND (stopped_at IS NULL OR stopped_at < grace_cutoff).

        For each candidate, calls get_instance:
          - 404 / null envelope → mark destroyed_at (instance already gone)
          - non-terminal actual_status → destroy_instance, then mark on success
          - VastAIRateLimitError → skip this sweep, next sweep retries
          - other error → log, leave for next sweep
        """
        grace_minutes = self._settings.orphaned_instance_cleanup_grace_period_minutes
        horizon_minutes = self._settings.orphaned_instance_cleanup_horizon_minutes
        now = datetime.now(UTC)
        stopped_before = now - timedelta(minutes=grace_minutes)
        created_after = now - timedelta(minutes=horizon_minutes)

        async with self._session_factory() as db:
            repo = GpuSessionRepository(db)
            candidates = await repo.list_orphaned_instance_candidates(
                terminal_statuses=tuple(TERMINAL_GPU_SESSION_STATUSES),
                stopped_before=stopped_before,
                created_after=created_after,
            )

        logger.info(
            "gpu_session.orphan_instance_cleanup.candidates",
            count=len(candidates),
        )

        processed = 0
        try:
            for session in candidates:
                await self._cleanup_orphan_instance(session)
                processed += 1
        except VastAIRateLimitError:
            logger.warning(
                "gpu_session.orphan_instance_cleanup.sweep_aborted_rate_limited",
                processed=processed,
                remaining=len(candidates) - processed,
            )
            # Don't re-raise — the run-loop should continue scheduling.
            # Next sweep cycle will pick up the remaining candidates.

    async def _cleanup_orphan_instance(self, session: GpuSession) -> None:
        """Verify and destroy one orphaned instance candidate."""
        try:
            instance = await self._vastai.get_instance(session.vastai_instance_id)  # type: ignore[arg-type]
        except InstanceNotFoundError:
            await self._mark_session_instance_destroyed(session.id)
            logger.info(
                "gpu_session.orphan_instance_cleanup.already_gone",
                session_id=str(session.id),
                instance_id=session.vastai_instance_id,
            )
            return
        except VastAIRateLimitError:
            logger.info(
                "gpu_session.orphan_instance_cleanup.skip_rate_limited",
                session_id=str(session.id),
                instance_id=session.vastai_instance_id,
            )
            raise
        except Exception:
            logger.exception(
                "gpu_session.orphan_instance_cleanup.get_instance_error",
                session_id=str(session.id),
                instance_id=session.vastai_instance_id,
            )
            return

        if instance is None:
            # Null envelope — Vast.ai garbage-collected, treat as gone.
            await self._mark_session_instance_destroyed(session.id)
            logger.info(
                "gpu_session.orphan_instance_cleanup.null_envelope_assume_gone",
                session_id=str(session.id),
            )
            return

        # Instance still exists. Destroy it.
        try:
            await self._vastai.destroy_instance(session.vastai_instance_id)  # type: ignore[arg-type]
            await self._mark_session_instance_destroyed(session.id)
            logger.warning(
                "gpu_session.orphan_instance_cleanup.destroyed",
                session_id=str(session.id),
                instance_id=session.vastai_instance_id,
                actual_status=instance.actual_status,
                cur_state=instance.cur_state,
            )
        except Exception:
            logger.exception(
                "gpu_session.orphan_instance_cleanup.destroy_failed",
                session_id=str(session.id),
                instance_id=session.vastai_instance_id,
            )

    async def _mark_session_instance_destroyed(self, session_id: UUID) -> None:
        """Stamp vastai_instance_destroyed_at on a session row."""
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            await repo.mark_instance_destroyed(session_id, datetime.now(UTC))
