"""Publishing run events - strictly after the transaction that wrote them committed.

Delivery is best-effort UX: the row is already durable, so a Redis failure is logged and swallowed,
never surfaced to the request or the worker. Payloads carry ids, kinds, stages and codes only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.api.schemas.events import EventType, StudioRunUpdatedPayload

if TYPE_CHECKING:
    from src.api.services.billing import BalanceEvent
    from src.api.services.event_bus import EventBus
    from src.db.models.studio import StudioRun

__all__ = ["StudioRunEvents"]

logger = structlog.get_logger(__name__)


class StudioRunEvents:
    """Post-commit publisher of ``studio.run.updated`` and the balance events of a run."""

    def __init__(self, event_bus: EventBus | None) -> None:
        self._bus = event_bus

    async def run_updated(self, run: StudioRun) -> None:
        """Publish the run's current state to its owner. Never raises."""
        if self._bus is None:
            return
        try:
            await self._bus.publish(
                user_id=run.user_id,
                event_type=EventType.STUDIO_RUN_UPDATED,
                payload=StudioRunUpdatedPayload(
                    run_id=run.id,
                    track_id=run.track_id,
                    kind=run.kind,
                    status=run.status,
                    stage=run.stage,
                    revision=run.revision,
                    error_code=run.error_code,
                ),
            )
        except Exception:
            logger.warning("studio.run.event_publish_failed", run_id=str(run.id))

    async def balance(self, event: BalanceEvent | None) -> None:
        """Publish a ledger change built by ``BillingService`` (a no-op on ``None``)."""
        if self._bus is not None:
            await self._bus.publish_balance(event)
