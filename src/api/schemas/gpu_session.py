"""HTTP schemas for GPU session endpoints. Service-layer DTOs stay in gpu_session/schemas.py."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import msgspec

from src.api.schemas.operation import OperationResponse
from src.core.enums import (
    DeploymentStatus,
    GpuSessionStatus,
    ModelType,
    OperationKind,
    OperationStatus,
    ProvisioningPhase,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    model_type: ModelType
    bundle_name: str
    bundle_version: str | None
    status: DeploymentStatus
    pending_restart: bool
    routing_suspended: bool
    is_primary: bool
    created_at: datetime
    activated_at: datetime | None
    current_operation: OperationResponse | None = None
    """Newest operation for this deployment, including terminal operations."""

    @classmethod
    def from_model(
        cls, m: GpuSessionDeployment, *, current_operation: GpuSessionOperation | None = None
    ) -> DeploymentResponse:
        return cls(
            id=m.id,
            model_type=ModelType(m.model_type),
            bundle_name=m.bundle_name,
            bundle_version=m.bundle_version,
            status=DeploymentStatus(m.status),
            pending_restart=m.pending_restart,
            routing_suspended=m.routing_suspended,
            is_primary=m.is_primary,
            created_at=m.created_at,
            activated_at=m.activated_at,
            current_operation=(
                OperationResponse.from_model(current_operation)
                if current_operation is not None
                else None
            ),
        )


class DeploymentSummaryResponse(msgspec.Struct, kw_only=True):
    """Deployment fields needed on the lightweight session list endpoint."""

    id: UUID
    model_type: ModelType
    status: DeploymentStatus
    is_primary: bool

    @classmethod
    def from_model(cls, m: GpuSessionDeployment) -> DeploymentSummaryResponse:
        return cls(
            id=m.id,
            model_type=ModelType(m.model_type),
            status=DeploymentStatus(m.status),
            is_primary=m.is_primary,
        )


class AttachDeploymentRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    model: ModelType
    """The model to attach to this already-running session."""


class DeploymentMutationResponse(msgspec.Struct, kw_only=True):
    deployment: DeploymentResponse
    operation: OperationResponse


class GpuSessionResponse(msgspec.Struct, kw_only=True):
    id: UUID
    user_id: UUID
    product_id: str
    status: GpuSessionStatus
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
    bootstrap_operation: OperationResponse | None = None
    deployments: list[DeploymentResponse] = msgspec.field(default_factory=list)
    """This session's deployments — always exactly one in P2 (D19)."""

    @classmethod
    def from_model(
        cls,
        m: GpuSession,
        *,
        bootstrap_operation: GpuSessionOperation | None = None,
        in_flight_job_count: int = 0,
        deployments: Sequence[GpuSessionDeployment] = (),
        current_operations: Mapping[UUID, GpuSessionOperation] | None = None,
    ) -> GpuSessionResponse:
        return cls(
            id=m.id,
            user_id=m.user_id,
            product_id=m.product_id,
            status=GpuSessionStatus(m.status),
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
            bootstrap_operation=(
                OperationResponse.from_model(bootstrap_operation)
                if bootstrap_operation is not None
                else None
            ),
            deployments=[
                DeploymentResponse.from_model(
                    deployment,
                    current_operation=(
                        current_operations.get(deployment.id)
                        if current_operations is not None
                        else None
                    ),
                )
                for deployment in deployments
            ],
        )


class GpuSessionListItemResponse(msgspec.Struct, kw_only=True):
    """Compact session list projection intentionally free of operation bodies."""

    id: UUID
    status: GpuSessionStatus
    product_id: str
    created_at: datetime
    started_at: datetime | None
    deployments: list[DeploymentSummaryResponse]

    @classmethod
    def from_model(
        cls, m: GpuSession, *, deployments: Sequence[GpuSessionDeployment] = ()
    ) -> GpuSessionListItemResponse:
        return cls(
            id=m.id,
            status=GpuSessionStatus(m.status),
            product_id=m.product_id,
            created_at=m.created_at,
            started_at=m.started_at,
            deployments=[
                DeploymentSummaryResponse.from_model(deployment) for deployment in deployments
            ],
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
    sessions: list[GpuSessionListItemResponse]
