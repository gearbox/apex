"""Gallery endpoint schemas — grid items, detail views, lineage."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import msgspec

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

    cover_url: str
    """Content proxy path for the grid cover.

    Resolution logic:
    - Image-input types (i2i, i2v, flf2v, v2v):
      → source output image OR uploaded input image.
    - Text-only types (t2i): → last generated output.
    - Text-only video (t2v): → video thumbnail (poster frame).

    Format: "/v1/content/outputs/{id}" or "/v1/content/uploads/{id}"
    """

    video_url: str | None = None
    """Content proxy path for the full video (autoplay).
    Present only for video generation types.
    Frontend uses cover_url (thumbnail) for fast grid load,
    then replaces with autoplaying video_url."""

    badge: GalleryBadge
    """'image' if input-driven (i2i, i2v, flf2v, v2v), 'prompt' if text-only."""

    media_type: OutputMediaType
    """Output media type: 'image' or 'video'. Derived from generation_type.is_video."""

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

    url: str
    """Content proxy path: '/v1/content/outputs/{id}'."""

    thumbnail_url: str | None = None
    """Content proxy path for video poster frame (if applicable)."""

    content_type: str
    """MIME type: 'image/jpeg', 'video/mp4', etc."""

    media_type: OutputMediaType
    """'image' or 'video' — derived from content_type."""

    format: str
    size_bytes: int
    output_index: int
    created_at: datetime


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

    input_image_url: str | None = None
    """Content proxy path for the input image/output.
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
