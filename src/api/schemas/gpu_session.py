"""HTTP schemas for GPU session endpoints. Service-layer DTOs stay in gpu_session/schemas.py."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

import msgspec

from src.core.enums import ModelType

if TYPE_CHECKING:
    from src.db.models.gpu_session import GpuSession


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
    bundle_name: str
    bundle_version: str | None
    tunnel_hostname: str | None
    vastai_gpu_name: str | None
    vastai_cost_per_hour_micros: int | None
    created_at: datetime
    started_at: datetime | None = None
    paused_at: datetime | None = None
    resumed_at: datetime | None = None
    stopped_at: datetime | None = None
    error_message: str | None = None

    @classmethod
    def from_model(cls, m: GpuSession) -> GpuSessionResponse:
        return cls(
            id=m.id,
            user_id=m.user_id,
            product_id=m.product_id,
            status=str(m.status),
            model_type=m.model_type,
            bundle_name=m.bundle_name,
            bundle_version=m.bundle_version,
            tunnel_hostname=m.tunnel_hostname,
            vastai_gpu_name=m.vastai_gpu_name,
            vastai_cost_per_hour_micros=m.vastai_cost_per_hour_micros,
            created_at=m.created_at,
            started_at=m.started_at,
            paused_at=m.paused_at,
            resumed_at=m.resumed_at,
            stopped_at=m.stopped_at,
            error_message=m.error_message,
        )


class StopConfirmationResponse(msgspec.Struct, kw_only=True):
    session_id: UUID
    model_type: str
    bundle_name: str
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
