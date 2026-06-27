"""Gallery endpoint schemas — grid items, detail views, lineage."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import msgspec

from src.api.schemas.media import MediaObject
from src.core.enums import (
    GalleryBadge,
    GallerySourceType,
    GenerationType,
    OutputMediaType,
)


class GalleryGridItem(msgspec.Struct, kw_only=True):
    """Single cell in the gallery grid — represents one generation group (job)."""

    job_id: UUID
    """Generation group identifier."""

    cover: MediaObject
    """Media envelope for the grid cover — always the job's own primary output.
    For image jobs: the primary output image with sm/md WEBP variants.
    For video jobs: original is the MP4; variants are poster-frame rasters."""

    badge: GalleryBadge
    """'image' if input-driven (i2i, i2v, flf2v, v2v), 'prompt' if text-only."""

    output_count: int
    """Number of non-thumbnail outputs in this group."""

    generation_type: GenerationType

    model: str | None = None

    aspect_ratio: str | None = None
    """Aspect ratio string, e.g. '16:9'. From the parent GenerationJob."""

    prompt_snippet: str
    """First 100 characters of the prompt for preview/search."""

    created_at: datetime


class GalleryOutputItem(msgspec.Struct, kw_only=True):
    """Single output within a gallery group detail view."""

    id: UUID

    output_index: int

    created_at: datetime

    media: MediaObject
    """Full media envelope: original asset + preview variants (sm/md WEBP).
    For video: original is the MP4 source; variants are poster-frame rasters."""


class GalleryLineage(msgspec.Struct, kw_only=True):
    """Single-level remix lineage."""

    source_type: GallerySourceType

    source_upload_id: UUID | None = None
    """If source was a direct upload."""

    source_job_id: UUID | None = None
    """If source was a previous generation's output."""

    source_job_name: str | None = None
    """Human-readable name of the source job."""

    source_output_id: UUID | None = None
    """Specific output used as input."""


class GalleryGroupDetail(msgspec.Struct, kw_only=True):
    """Full detail view of a generation group."""

    job_id: UUID

    # --- Header ---
    badge: GalleryBadge
    """'image' if input-driven, 'prompt' if text-only."""

    input_media: MediaObject | None = None
    """Media envelope for the source input (upload or remixed output).
    Present when badge == 'image'."""

    prompt: str
    negative_prompt: str | None = None

    # --- Outputs grid ---
    outputs: list[GalleryOutputItem]
    """All non-thumbnail outputs, ordered by output_index."""

    # --- Metadata ---
    media_type: OutputMediaType
    model: str | None = None
    provider: str
    generation_type: GenerationType
    aspect_ratio: str | None = None
    """Aspect ratio string, e.g. '16:9'."""
    token_cost: int | None = None
    created_at: datetime
    completed_at: datetime | None = None

    # --- Lineage ---
    lineage: GalleryLineage | None = None
