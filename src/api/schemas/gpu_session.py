"""HTTP schemas for GPU session endpoints. Service-layer DTOs stay in gpu_session/schemas.py."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import msgspec

from src.core.enums import ModelType

if TYPE_CHECKING:
    from src.db.models.gpu_session import GpuSession


class DownloadProgressBody(msgspec.Struct, kw_only=True):
    """Download-progress snapshot sent by aisha during the 'downloading' phase."""

    bytes_done: int
    bytes_total: int
    files_done: int
    files_total: int


class StartSessionRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    model: ModelType
    """The model to provision a GPU session for."""

    bundle_override: str | None = None
    """Admin-only: pin a specific bundle 'name' or 'name:version'. Ignored for non-admins."""


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
    provisioning_phase: str | None = None
    """Latest provisioning phase reported via callback (e.g. 'downloading', 'ready')."""
    provisioning_progress: dict[str, Any] | None = None
    """Latest progress blob from node callback (download bytes, message, etc.)."""

    @classmethod
    def from_model(cls, m: GpuSession, *, in_flight_job_count: int = 0) -> GpuSessionResponse:
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
            provisioning_phase=m.provisioning_phase,
            provisioning_progress=m.provisioning_progress,
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
