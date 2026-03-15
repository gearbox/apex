"""Unified job schemas for the cross-provider jobs API.

All providers (Grok, ComfyUI) surface jobs through this single schema set
so the frontend only needs to handle one response shape.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import msgspec

from src.core.enums import GenerationType, JobStatus


class JobCreatedResponse(msgspec.Struct, kw_only=True):
    """Creation receipt returned immediately after a job is submitted.

    The frontend should use ``job_id`` to poll ``GET /v1/jobs/{job_id}``
    for the full ``UnifiedJobResponse`` including outputs.
    """

    job_id: UUID
    """Unique job identifier."""

    status: JobStatus
    """Initial status (typically ``queued`` or ``completed`` for sync jobs)."""

    name: str
    """Human-readable job name."""

    model: str
    """Model identifier used for this job."""

    generation_type: GenerationType
    """Workflow type: t2i, i2i, t2v, i2v, v2v."""

    created_at: datetime
    """Job creation timestamp."""

    message: str | None = None
    """Optional human-readable status message (e.g. 'Poll job status for results.')."""

    tokens_charged: int | None = None
    """Tokens charged at submission time."""

    balance_remaining: int | None = None
    """Token balance after the charge."""


class JobOutputItem(msgspec.Struct, kw_only=True):
    """A single generated output (image or video) within a job."""

    id: UUID
    """Output record UUID."""

    url: str
    """Presigned URL for downloading the full-resolution output.
    Valid for ~1 hour; caller should not cache."""

    content_type: str
    """MIME type, e.g. ``image/jpeg`` or ``video/mp4``."""

    format: str
    """File format string, e.g. ``jpeg``, ``webp``, ``mp4``."""

    size_bytes: int
    """File size in bytes."""

    output_index: int
    """Position within the batch (0-based). -1 is reserved for thumbnails."""

    is_thumbnail: bool = False
    """True for the extracted first-frame/poster image of a video."""


class UnifiedJobResponse(msgspec.Struct, kw_only=True):
    """Full job detail — generation params + current status + outputs.

    Returned by both the single-job and list endpoints.
    The ``outputs`` list is empty while the job is still processing.
    """

    id: UUID
    """Job UUID."""

    name: str
    """Human-readable job name (auto-generated from prompt if not supplied)."""

    status: JobStatus
    """Current lifecycle status."""

    provider: str
    """Generation provider identifier, e.g. ``grok`` or ``aisha``."""

    model: str | None = None
    """Model identifier, e.g. ``grok-imagine-image``. None for legacy ComfyUI jobs."""

    generation_type: GenerationType
    """Workflow type: t2i, i2i, t2v, i2v, v2v."""

    # Generation parameters — preserved for gallery display and re-generation
    prompt: str
    negative_prompt: str | None = None
    aspect_ratio: str | None = None
    """Aspect ratio string, e.g. ``16:9``. None when not applicable."""

    # Billing
    token_cost: int | None = None
    """Tokens charged for this job. None while pending/running."""

    # Timestamps
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Results
    outputs: list[JobOutputItem] = msgspec.field(default_factory=list)
    """Ordered list of outputs. Thumbnail (if any) is first (``is_thumbnail=True``)."""

    thumbnail_url: str | None = None
    """Convenience shortcut — presigned URL of the thumbnail output.
    Equivalent to ``outputs[0].url`` when ``outputs[0].is_thumbnail is True``."""

    error: str | None = None
    """Error message for failed jobs."""


class UnifiedJobListResponse(msgspec.Struct, kw_only=True):
    """Paginated list of jobs."""

    items: list[UnifiedJobResponse]
    total: int
    limit: int
    offset: int
