"""Unified job schemas for the cross-provider jobs API.

All providers (Grok, ComfyUI) surface jobs through this single schema set
so the frontend only needs to handle one response shape.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import msgspec

from src.api.schemas.media import MediaObject
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

    output_index: int
    """Position within the batch (0-based)."""

    media: MediaObject
    """Full media envelope: original asset + preview variants (sm/md WEBP).
    For video: original is the MP4 source; variants are poster-frame rasters."""


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
    """Ordered list of outputs. Empty while the job is processing."""

    error: str | None = None
    """Public-safe error message for failed jobs; never internal diagnostics."""

    failure_code: str | None = None
    """Stable public failure code for failed jobs, when available."""
