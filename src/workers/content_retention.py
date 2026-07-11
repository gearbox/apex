"""Periodic background task that enforces content retention (expires_at)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from src.api.services.content_retention import ContentRetentionService

logger = structlog.get_logger(__name__)


class ContentRetentionWorker(PeriodicWorker):
    """Worker that periodically sweeps expired outputs/uploads (DB + R2).

    No leader lock (D5): sweep deletes are idempotent — a row already gone
    is simply not selected again, and an R2 delete of a missing key is a
    no-op — so duplicate ticks across instances are harmless. Revisit if
    Apex goes multi-replica and duplicate R2 delete traffic becomes a
    concern worth avoiding.
    """

    def __init__(self, *, service: ContentRetentionService, interval: int) -> None:
        super().__init__(
            name="content_retention",
            interval_seconds=interval,
            jitter_seconds=5.0,
            use_leader_lease=False,
        )
        self._service = service

    async def run_once(self) -> None:
        """Execute a single content retention sweep."""
        try:
            await self._service.sweep()
        except Exception as e:
            logger.exception("content_retention.error", error=str(e))
