"""Public, defensive projection of durable GPU operation state."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import msgspec
import structlog

from src.core.enums import (
    OperationKind,
    OperationStatus,
    ProvisioningPhase,
    RateUnit,
    WorkUnit,
)

if TYPE_CHECKING:
    from src.db.models.gpu_session_operation import GpuSessionOperation

logger = structlog.get_logger(__name__)


class OperationWorkResponse(msgspec.Struct, kw_only=True):
    completed: int
    total: int | None
    unit: WorkUnit


class OperationRateResponse(msgspec.Struct, kw_only=True):
    value: float
    unit: RateUnit


class OperationTargetResponse(msgspec.Struct, kw_only=True):
    bundle: str
    bundle_version: str | None
    mode: str | None


class OperationProgressResponse(msgspec.Struct, kw_only=True):
    progress_pct: float | None
    work: OperationWorkResponse | None
    items: OperationWorkResponse | None
    rate: OperationRateResponse | None
    eta_seconds: float | None


class OperationErrorResponse(msgspec.Struct, kw_only=True):
    message: str


def _warn_unprojectable(operation_id: UUID, field: str) -> None:
    """Record foreign telemetry that cannot safely cross the public boundary."""
    logger.warning("operation.progress.unprojectable", operation_id=str(operation_id), field=field)


def _as_int(value: object) -> int | None:
    """Accept JSON integers, but not bool (a Python int subclass)."""
    return value if type(value) is int else None


def _as_number(value: object) -> float | None:
    """Accept JSON numeric values while rejecting bool and non-numeric types."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _work_from_json(
    value: object, *, operation_id: UUID, field: str
) -> OperationWorkResponse | None:
    if not isinstance(value, dict):
        _warn_unprojectable(operation_id, field)
        return None
    completed = _as_int(value.get("completed"))
    unit_value = value.get("unit")
    if completed is None or not isinstance(unit_value, str):
        _warn_unprojectable(operation_id, field)
        return None
    total_value = value.get("total")
    total = None if total_value is None else _as_int(total_value)
    if total_value is not None and total is None:
        _warn_unprojectable(operation_id, field)
        return None
    try:
        unit = WorkUnit(unit_value)
    except ValueError:
        _warn_unprojectable(operation_id, field)
        return None
    return OperationWorkResponse(completed=completed, total=total, unit=unit)


def _rate_from_json(value: object, *, operation_id: UUID) -> OperationRateResponse | None:
    if not isinstance(value, dict):
        _warn_unprojectable(operation_id, "rate")
        return None
    raw_value = _as_number(value.get("value"))
    unit_value = value.get("unit")
    if raw_value is None or not isinstance(unit_value, str):
        _warn_unprojectable(operation_id, "rate")
        return None
    try:
        unit = RateUnit(unit_value)
    except ValueError:
        _warn_unprojectable(operation_id, "rate")
        return None
    return OperationRateResponse(value=raw_value, unit=unit)


def _progress_from_json(value: object, *, operation_id: UUID) -> OperationProgressResponse | None:
    """Safely project Aisha's untrusted JSONB envelope into the public shape."""
    if value is None:
        return None
    if not isinstance(value, dict):
        _warn_unprojectable(operation_id, "progress")
        return None

    raw_work = value.get("work")
    work = (
        _work_from_json(raw_work, operation_id=operation_id, field="work")
        if raw_work is not None
        else None
    )
    raw_items = value.get("items")
    items = (
        _work_from_json(raw_items, operation_id=operation_id, field="items")
        if raw_items is not None
        else None
    )
    raw_rate = value.get("rate")
    rate = _rate_from_json(raw_rate, operation_id=operation_id) if raw_rate is not None else None
    raw_eta = value.get("eta_seconds")
    eta_seconds = _as_number(raw_eta) if raw_eta is not None else None
    if raw_eta is not None and eta_seconds is None:
        _warn_unprojectable(operation_id, "eta_seconds")

    progress_pct = (
        round(work.completed / work.total * 100, 1)
        if work is not None and work.total is not None and work.total > 0
        else None
    )
    if all(item is None for item in (work, items, rate, eta_seconds)):
        return None
    return OperationProgressResponse(
        progress_pct=progress_pct,
        work=work,
        items=items,
        rate=rate,
        eta_seconds=eta_seconds,
    )


class OperationResponse(msgspec.Struct, kw_only=True):
    """Public state of one asynchronous session/deployment operation."""

    id: UUID
    session_id: UUID
    deployment_id: Annotated[
        UUID | None,
        msgspec.Meta(
            description=(
                "Informational target deployment for deployment-scoped operations. "
                "Null for session-scoped operations including cohort restarts. Never "
                "use this field to associate an operation-update frame with a "
                "deployment; patch every cached deployment whose current_operation.id "
                "matches this operation's id instead."
            )
        ),
    ]
    kind: OperationKind
    status: OperationStatus
    phase: ProvisioningPhase | None
    sequence: int
    target: OperationTargetResponse | None
    progress: OperationProgressResponse | None
    message: str | None
    error: OperationErrorResponse | None
    started_at: datetime | None
    updated_at: datetime
    finished_at: datetime | None

    @classmethod
    def from_model(cls, m: GpuSessionOperation) -> OperationResponse:
        """Project a durable row without exposing opaque Aisha diagnostics."""
        try:
            phase = ProvisioningPhase(m.phase) if m.phase is not None else None
        except ValueError:
            _warn_unprojectable(m.id, "phase")
            phase = None
        target = (
            OperationTargetResponse(
                bundle=m.target_bundle,
                bundle_version=m.target_bundle_version,
                mode=m.target_mode,
            )
            if m.target_bundle is not None
            else None
        )
        return cls(
            id=m.id,
            session_id=m.session_id,
            deployment_id=m.deployment_id,
            kind=OperationKind(m.kind),
            status=OperationStatus(m.status),
            phase=phase,
            sequence=m.last_sequence,
            target=target,
            progress=_progress_from_json(m.progress, operation_id=m.id),
            message=m.message,
            error=OperationErrorResponse(message=m.error) if m.error else None,
            started_at=m.node_started_at,
            updated_at=m.updated_at,
            finished_at=m.terminal_at,
        )
