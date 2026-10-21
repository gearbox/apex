"""Wire and domain types for the Alibaba Model Studio (DashScope) video API."""

from __future__ import annotations

from enum import StrEnum

import msgspec


class ModelStudioTaskStatus(StrEnum):
    """Task lifecycle as reported by ``GET /api/v1/tasks/{task_id}``."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    UNKNOWN = "UNKNOWN"
    """Upstream: the task does not exist or has expired (24 h)."""


class ModelStudioCancelResult(StrEnum):
    """Outcome of ``POST /api/v1/tasks/{task_id}/cancel``."""

    CANCELLED = "cancelled"
    NOT_CANCELLABLE = "not_cancellable"
    """Upstream ``UnsupportedOperation``: only ``PENDING`` tasks can be cancelled."""


class ModelStudioTaskRef(msgspec.Struct, frozen=True, kw_only=True):
    """Result of accepting a create call."""

    task_id: str
    request_id: str | None = None


class ModelStudioVideoUsage(msgspec.Struct, kw_only=True):
    """Billing-relevant usage; present for successful tasks only.

    Timestamps (``submit_time`` etc.) are deliberately not modelled: upstream
    sends them as UTC+8 strings without an offset and nothing needs them.
    """

    duration: float | None = None
    input_video_duration: float | None = None
    output_video_duration: float | None = None
    sr: int | str | None = msgspec.field(default=None, name="SR")
    ratio: str | None = None
    fps: int | None = None


class ModelStudioTask(msgspec.Struct, frozen=True, kw_only=True):
    """A task as seen by one poll."""

    task_id: str
    task_status: ModelStudioTaskStatus
    video_url: str | None = None
    code: str | None = None
    message: str | None = None
    usage: ModelStudioVideoUsage | None = None
    request_id: str | None = None


# ---- raw response shapes (unknown fields are ignored by msgspec's default) ----


class WireOutput(msgspec.Struct, kw_only=True):
    task_id: str | None = None
    task_status: str | None = None
    video_url: str | None = None
    code: str | None = None
    message: str | None = None


class WireTaskResponse(msgspec.Struct, kw_only=True):
    request_id: str | None = None
    output: WireOutput | None = None
    usage: ModelStudioVideoUsage | None = None
    code: str | None = None
    message: str | None = None


class WireError(msgspec.Struct, kw_only=True):
    code: str | None = None
    message: str | None = None
    request_id: str | None = None
