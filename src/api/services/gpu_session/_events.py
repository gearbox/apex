"""Shared SSE event publishing for GPU session lifecycle.

Both GpuSessionService and GpuProvisioningWorker emit the same
GPU_SESSION_STATUS_CHANGED event shape at every transition. This helper
centralizes the payload construction and error-swallowing so the event
contract lives in one place.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.events import EventType, GpuSessionStatusPayload
from src.api.schemas.ops_events import GpuNodeStartedOpsPayload, OpsEventType
from src.core.enums import GpuSessionStatus

if TYPE_CHECKING:
    from src.api.services.event_bus import EventBus
    from src.api.services.ops_event_bus import OpsEventBus
    from src.db.models.gpu_session import GpuSession

logger = structlog.get_logger(__name__)

FIRE_AND_FORGET_TIMEOUT_SECS = 5


async def publish_status_event(
    event_bus: EventBus | None,
    session: GpuSession,
    *,
    previous_status: str,
    error_message: str | None = None,
    reason: str | None = None,
    ops_event_bus: OpsEventBus | None = None,
) -> None:
    """Fire-and-forget SSE publish. Failures are logged but never propagate.

    A None ``event_bus`` is a legitimate "no-op" case (the GPU stack can run
    without SSE when Redis isn't configured). Any other exception is caught
    and logged so that an unhealthy EventBus never breaks a user-visible
    state transition.

    Also fires the ``GPU_NODE_STARTED`` ops event, but only on the
    ``provisioning -> active`` edge (D9) — ``resuming -> active`` is a
    resume, not a new node, so it deliberately does NOT fire.
    """
    if event_bus is not None:
        try:
            await asyncio.wait_for(
                event_bus.publish(
                    user_id=session.user_id,
                    event_type=EventType.GPU_SESSION_STATUS_CHANGED,
                    payload=GpuSessionStatusPayload(
                        session_id=session.id,
                        # session.status is already a string (Mapped[str] on the model);
                        # this str() is defensive against enum-typed test mocks.
                        status=str(session.status),
                        previous_status=previous_status,
                        # session.model_type is stored as the enum `.value` already;
                        # use it directly for consistency across all call sites.
                        model_type=session.model_type,
                        tunnel_hostname=session.tunnel_hostname,
                        error_message=error_message,
                        reason=reason,
                    ),
                ),
                timeout=FIRE_AND_FORGET_TIMEOUT_SECS,
            )
        except TimeoutError:
            logger.warning(
                "gpu_session.event_publish_timeout",
                session_id=str(session.id),
                timeout=FIRE_AND_FORGET_TIMEOUT_SECS,
            )
        except Exception:
            logger.exception("gpu_session.event_publish_failed", session_id=str(session.id))

    if (
        ops_event_bus is not None
        and str(session.status) == GpuSessionStatus.active
        and previous_status == GpuSessionStatus.provisioning
    ):
        await ops_event_bus.publish(
            event_type=OpsEventType.GPU_NODE_STARTED,
            product_id=session.product_id,
            payload=GpuNodeStartedOpsPayload(
                session_id=session.id,
                user_id=session.user_id,
                model_type=session.model_type,
            ),
        )
