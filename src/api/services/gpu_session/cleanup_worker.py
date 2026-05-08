"""Background worker: reconciles CF tunnels against the gpu_sessions table.

Detects tunnels that are (a) ours (by name prefix), (b) older than the grace
period, (c) not referenced by any non-terminal gpu_sessions row, and deletes them.

Runs once every ``settings.orphaned_tunnel_cleanup_interval_minutes`` (default 60).
Intentionally slow — orphans are not an emergency.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from src.api.services.cloudflare.schemas import TunnelListEntry
from src.core.enums import GpuSessionStatus
from src.db.repositories.gpu_session import GpuSessionRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.cloudflare.client import CloudflareTunnelClient
    from src.core.config import Settings

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
    """Reconciles CF tunnels against the gpu_sessions table.

    Detects tunnels that are (a) ours (by name prefix), (b) older than
    the grace period, (c) not referenced by any non-terminal gpu_sessions row,
    and deletes them.

    Runs once every ``settings.orphaned_tunnel_cleanup_interval_minutes``
    (default 60). Intentionally slow — orphans are not an emergency.
    """

    _TUNNEL_NAME_PREFIX = "gpu-session-"

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        cf_client: CloudflareTunnelClient,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._cf = cf_client
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
        """One cleanup sweep: find orphaned tunnels and delete them."""
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
