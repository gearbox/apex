"""Factory for the settlement primitive shared by read-through and worker paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.services.job_state_transition import JobStateTransitionService

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BillingService
    from src.api.services.event_bus import EventBus
    from src.api.services.ops_event_bus import OpsEventBus


def make_transition_service_factory(
    *,
    billing_service: BillingService | None,
    event_bus: EventBus | None,
    ops_event_bus: OpsEventBus | None,
) -> Callable[[AsyncSession], JobStateTransitionService]:
    """Return a per-session ``JobStateTransitionService`` constructor."""
    from src.api.services.billing import BillingService

    billing = billing_service or BillingService()

    def factory(session: AsyncSession) -> JobStateTransitionService:
        return JobStateTransitionService(
            session=session,
            event_bus=event_bus,
            billing_service=billing,
            ops_event_bus=ops_event_bus,
        )

    return factory
