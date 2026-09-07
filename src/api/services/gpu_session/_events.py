"""Shared SSE event publishing for GPU session lifecycle.

Both GpuSessionService and GpuProvisioningWorker emit the same
GPU_SESSION_STATUS_CHANGED event shape at every transition. This helper
centralizes the payload construction and error-swallowing so the event
contract lives in one place.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from src.api.schemas.events import EventType, GpuDeploymentStatusPayload, GpuSessionStatusPayload
from src.api.schemas.ops_events import GpuNodeStartedOpsPayload, OpsEventType
from src.core.enums import GpuSessionStatus

if TYPE_CHECKING:
    from uuid import UUID

    from src.api.services.event_bus import EventBus
    from src.api.services.ops_event_bus import OpsEventBus
    from src.db.models.gpu_session import GpuSession
    from src.db.models.gpu_session_deployment import GpuSessionDeployment
    from src.db.models.gpu_session_operation import GpuSessionOperation

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


_OPERATION_ID_NOT_GIVEN: Any = object()


async def publish_deployment_event(
    event_bus: EventBus | None,
    deployment: GpuSessionDeployment,
    *,
    operation: GpuSessionOperation | None = None,
    operation_id: UUID | None = _OPERATION_ID_NOT_GIVEN,
    error_message: str | None = None,
) -> None:
    """Fire-and-forget SSE publish for a P4 deployment state change (Part 4).

    ``operation`` is whichever operation currently governs the deployment's progress
    (its provision or restart operation) — pass it when already loaded in the same
    tick as the state write, so ``operation_phase``/``operation_progress`` reflect
    the node's own telemetry. ``operation_id`` lets a caller name the governing
    operation explicitly when it hasn't loaded the row, *including as None* when
    there deliberately is none to report — e.g. a stranded removal reaper, whose
    governing operation must never fall back to the deployment's stale
    ``restart_operation_id`` from some earlier, already-terminal restart (S1). A
    caller that passes neither keyword gets the historical N4 behavior: falls
    back to ``deployment.restart_operation_id or deployment.provision_operation_id``,
    so every such event still carries an id the client can poll — restart_operation_id
    takes precedence there because, once it's set, the restart (not the original
    provision) is what currently governs the deployment's progress. The default
    is a private sentinel rather than ``None`` precisely so "explicitly None" and
    "not given at all" can be told apart.
    """
    if event_bus is None:
        return
    resolved_operation_id: UUID | None
    if operation is not None:
        resolved_operation_id = operation.id
    elif operation_id is not _OPERATION_ID_NOT_GIVEN:
        resolved_operation_id = operation_id
    else:
        resolved_operation_id = deployment.restart_operation_id or deployment.provision_operation_id
    try:
        await asyncio.wait_for(
            event_bus.publish(
                user_id=deployment.user_id,
                event_type=EventType.GPU_DEPLOYMENT_STATUS_CHANGED,
                payload=GpuDeploymentStatusPayload(
                    deployment_id=deployment.id,
                    session_id=deployment.session_id,
                    model_type=deployment.model_type,
                    status=str(deployment.status),
                    pending_restart=deployment.pending_restart,
                    routing_suspended=deployment.routing_suspended,
                    operation_id=resolved_operation_id,
                    operation_phase=operation.phase if operation is not None else None,
                    operation_progress=operation.progress if operation is not None else None,
                    error_message=error_message,
                ),
            ),
            timeout=FIRE_AND_FORGET_TIMEOUT_SECS,
        )
    except TimeoutError:
        logger.warning(
            "gpu_session.deployment.event_publish_timeout",
            deployment_id=str(deployment.id),
            timeout=FIRE_AND_FORGET_TIMEOUT_SECS,
        )
    except Exception:
        logger.exception(
            "gpu_session.deployment.event_publish_failed", deployment_id=str(deployment.id)
        )
