"""HTTP schemas for GPU session endpoints. Service-layer DTOs stay in gpu_session/schemas.py."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import msgspec

from src.core.enums import ModelType, OperationKind, OperationStatus, ProvisioningPhase

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.db.models.gpu_session import GpuSession
    from src.db.models.gpu_session_deployment import GpuSessionDeployment
    from src.db.models.gpu_session_operation import GpuSessionOperation


class OperationBatchBody(msgspec.Struct, kw_only=True):
    """Optional batch position in an operation envelope."""

    batch_id: str
    index: int
    total: int


class OperationTargetBody(msgspec.Struct, kw_only=True):
    """Bundle target attached to an operation envelope."""

    bundle: str
    bundle_version: str | None
    mode: str


class OperationEventBody(msgspec.Struct, kw_only=True):
    """Tolerant-reader schema for an Aisha telemetry v2 operation event."""

    schema_version: int
    event_id: str
    session_id: UUID
    operation_id: UUID
    operation_kind: OperationKind
    batch: OperationBatchBody | None
    sequence: int
    target: OperationTargetBody | None
    status: OperationStatus
    phase: ProvisioningPhase | None
    started_at: datetime
    ts: datetime
    elapsed_seconds: float
    phase_elapsed_seconds: float | None
    progress: dict[str, Any] | None
    plan: dict[str, Any] | None
    summary: dict[str, Any] | None
    message: str
    error: str | None


class ClaimCommandRequest(msgspec.Struct, kw_only=True):
    """Tolerant-reader schema for the node agent's command claim request."""

    agent_id: str
    schema_version: int


class StartSessionRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    model: ModelType
    """The model to provision a GPU session for."""

    bundle_override: str | None = None
    """Admin-only: pin a specific bundle 'name' or 'name:version'. Ignored for non-admins."""


class DeploymentResponse(msgspec.Struct, kw_only=True):
    """Read-only projection of one gpu_session_deployments row.

    P4 adds attach/remove actions; the read model shipped early in P2 so
    frontend work on the shape could start in parallel — see D19.
    """

    id: UUID
    model_type: str
    bundle_name: str
    bundle_version: str | None
    status: str
    pending_restart: bool
    is_primary: bool
    created_at: datetime
    activated_at: datetime | None
    restart_operation_id: UUID | None = None
    """The restart operation that will make this deployment routable, once pending_restart (D33)."""

    @classmethod
    def from_model(cls, m: GpuSessionDeployment) -> DeploymentResponse:
        return cls(
            id=m.id,
            model_type=m.model_type,
            bundle_name=m.bundle_name,
            bundle_version=m.bundle_version,
            status=m.status,
            pending_restart=m.pending_restart,
            is_primary=m.is_primary,
            created_at=m.created_at,
            activated_at=m.activated_at,
            restart_operation_id=m.restart_operation_id,
        )


class AttachDeploymentRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    model: ModelType
    """The model to attach to this already-running session."""


class AttachDeploymentResponse(msgspec.Struct, kw_only=True):
    deployment: DeploymentResponse
    operation_id: UUID
    """The provision operation the client can poll/subscribe to for progress."""


class GpuSessionResponse(msgspec.Struct, kw_only=True):
    id: UUID
    user_id: UUID
    product_id: str
    status: str
    model_type: str
    tunnel_hostname: str | None
    vastai_gpu_name: str | None
    vastai_cost_per_hour_micros: int | None
    created_at: datetime
    started_at: datetime | None = None
    paused_at: datetime | None = None
    resumed_at: datetime | None = None
    stopped_at: datetime | None = None
    error_message: str | None = None
    in_flight_job_count: int = 0
    """Number of QUEUED/RUNNING Aisha jobs on this session. Non-zero only for active
    sessions. Used by the frontend to gate the Pause button."""
    provisioning_status: str | None = None
    """Latest status of this session's current bootstrap operation."""
    provisioning_phase: str | None = None
    """Latest phase of this session's current bootstrap operation."""
    provisioning_progress: dict[str, Any] | None = None
    """Latest generic progress object from this session's bootstrap operation."""
    deployments: list[DeploymentResponse] = []
    """This session's deployments — always exactly one in P2 (D19)."""

    @classmethod
    def from_model(
        cls,
        m: GpuSession,
        *,
        bootstrap_operation: GpuSessionOperation | None = None,
        in_flight_job_count: int = 0,
        deployments: Sequence[GpuSessionDeployment] = (),
    ) -> GpuSessionResponse:
        return cls(
            id=m.id,
            user_id=m.user_id,
            product_id=m.product_id,
            status=str(m.status),
            model_type=m.model_type,
            tunnel_hostname=m.tunnel_hostname,
            vastai_gpu_name=m.vastai_gpu_name,
            vastai_cost_per_hour_micros=m.vastai_cost_per_hour_micros,
            created_at=m.created_at,
            started_at=m.started_at,
            paused_at=m.paused_at,
            resumed_at=m.resumed_at,
            stopped_at=m.stopped_at,
            error_message=m.error_message,
            in_flight_job_count=in_flight_job_count,
            provisioning_status=(bootstrap_operation.status if bootstrap_operation else None),
            provisioning_phase=(bootstrap_operation.phase if bootstrap_operation else None),
            provisioning_progress=bootstrap_operation.progress if bootstrap_operation else None,
            deployments=[DeploymentResponse.from_model(d) for d in deployments],
        )


class StopConfirmationResponse(msgspec.Struct, kw_only=True):
    session_id: UUID
    model_type: str
    vastai_gpu_name: str | None
    vastai_cost_per_hour_micros: int | None
    active_duration_seconds: int
    paused_duration_seconds: int
    """Cumulative time the session spent paused. Useful for UX ("5m active, 20m paused")
    and billed in a later phase at the storage rate."""
    estimated_final_tokens: int
    """Estimated total token cost if stopped now (including overage)."""
    message: str


class StopSessionRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    confirmed: bool = False


class ListSessionsResponse(msgspec.Struct, kw_only=True):
    sessions: list[GpuSessionResponse]
