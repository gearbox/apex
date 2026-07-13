"""msgspec schemas for the video frame extraction API (/v1/frames)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

import msgspec

from src.api.schemas.media import MediaObject
from src.core.constants import MAX_EXTRACT_TIMESTAMPS, MAX_PREVIEW_FRAME_COUNT

# -----------------------------------------------------------------------------
# Requests
# -----------------------------------------------------------------------------


class FramePreviewRequest(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    """POST /v1/frames/preview — request a low-res, N-frame preview strip."""

    source_output_id: UUID | None = None
    source_upload_id: UUID | None = None
    frame_count: Annotated[int, msgspec.Meta(ge=2, le=MAX_PREVIEW_FRAME_COUNT)] = 12


class FrameExtractRequest(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    """POST /v1/frames/extract — request full-resolution frames at given timestamps."""

    source_output_id: UUID | None = None
    source_upload_id: UUID | None = None
    timestamps_ms: Annotated[
        list[Annotated[int, msgspec.Meta(ge=0)]],
        msgspec.Meta(min_length=1, max_length=MAX_EXTRACT_TIMESTAMPS),
    ]
    """Each must be < the source video's duration — checked by the worker
    once it probes the file (see FrameExtractionWorker._run_extract); the
    upper bound is exclusive, matching the preview strip's timestamps."""


# -----------------------------------------------------------------------------
# Responses
# -----------------------------------------------------------------------------


class FrameJobCreatedResponse(msgspec.Struct, kw_only=True):
    """202 response for POST /v1/frames/preview and /v1/frames/extract."""

    job_id: UUID
    status: str


class FrameJobSource(msgspec.Struct, kw_only=True):
    """Identifies the video a frame extraction job was run against."""

    type: str
    """'output' or 'upload'."""

    id: UUID


class PreviewFrame(msgspec.Struct, kw_only=True):
    """One frame in a completed preview strip."""

    index: int
    timestamp_ms: int
    url: str
    """Presigned R2 URL, generated per-read — TTL-bounded, never persisted."""


class FramePreviewResult(msgspec.Struct, kw_only=True):
    """Preview payload — present when kind=preview and status=completed."""

    frames: list[PreviewFrame]
    expires_in_seconds: int
    duration_ms: int
    """Server-probed (ffprobe) duration of the source video. Valid extract
    timestamps are exactly [0, duration_ms) — upper bound exclusive."""


class ExtractedFrame(msgspec.Struct, kw_only=True):
    """One saved frame from a completed extract job."""

    timestamp_ms: int
    upload_id: UUID
    media: MediaObject


class FrameExtractResult(msgspec.Struct, kw_only=True):
    """Extract payload — present when kind=extract and status=completed."""

    frames: list[ExtractedFrame]


class FrameJobResponse(msgspec.Struct, kw_only=True):
    """GET /v1/frames/jobs/{job_id} response."""

    job_id: UUID
    kind: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    source: FrameJobSource
    preview: FramePreviewResult | None = None
    extracted: FrameExtractResult | None = None
