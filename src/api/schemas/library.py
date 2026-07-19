"""Library endpoint schemas — asset items, detail views, group detail, patch."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import msgspec

from src.api.schemas.media import MediaObject
from src.api.services.library_capabilities import LibraryAction
from src.core.enums import GenerationType, LibraryBadge, LibraryGroupSourceType, OutputMediaType
from src.core.library_ref import LibraryAssetSource


class LibraryAssetItem(msgspec.Struct, kw_only=True):
    """Single cell in the library grid — one upload or one generation output."""

    asset_ref: str
    """Wire-format asset reference, e.g. ``"upload:<uuid>"`` / ``"output:<uuid>"``."""

    source: LibraryAssetSource

    media: MediaObject

    created_at: datetime

    expires_at: datetime
    """UTC timestamp when this asset's content is deleted by retention cleanup."""

    display_title: str | None = None
    """User-set display name, if any (see PATCH /assets/{asset_ref})."""

    original_filename: str | None = None
    """Upload-only: the originally uploaded filename."""

    is_favorite: bool

    duration_ms: int | None = None
    """Upload-only: video duration for uploaded videos."""

    job_id: UUID | None = None
    """Output-only: the generation job this output belongs to."""

    output_count: int | None = None
    """Output-only: number of non-thumbnail outputs in the same job."""

    model: str | None = None
    """Output-only: generation model."""

    generation_type: GenerationType | None = None
    """Output-only."""

    available_actions: tuple[LibraryAction, ...]


class LibraryLineage(msgspec.Struct, kw_only=True):
    """Single-level frame-extraction lineage for a library asset.

    Populated from UserImage.source_output_id / source_upload_id /
    source_timestamp_ms — distinct from LibraryGroupLineage (job-level remix
    lineage, used in LibraryGroupDetail)."""

    source_asset_ref: str | None = None
    """Wire-format ref of the source asset the frame was extracted from."""

    source_job_id: UUID | None = None
    """Set when the source was a generation output — its owning job."""

    source_timestamp_ms: int | None = None
    """Timestamp within the source video this frame was extracted at."""


class LibraryDescendants(msgspec.Struct, kw_only=True):
    """Counts of assets derived from this one."""

    job_count: int
    """Generation jobs referencing this asset as input (input_image_id / source_output_id)."""

    frame_count: int
    """Extracted frames (user_images) referencing this asset as their source."""


class LibraryAssetDetail(LibraryAssetItem, kw_only=True):
    """Full detail view of a single library asset."""

    prompt: str | None = None
    negative_prompt: str | None = None
    provider: str | None = None
    aspect_ratio: str | None = None
    token_cost: int | None = None
    completed_at: datetime | None = None

    lineage: LibraryLineage | None = None
    descendants: LibraryDescendants


class LibraryAssetPatch(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to update mutable library asset fields (Phase 1: display_title only)."""

    display_title: str | None | msgspec.UnsetType = msgspec.UNSET
    """Absent = leave unchanged. ``null`` = clear. String = set (max 255 chars)."""


# ---------------------------------------------------------------------------
# Group detail — ported from schemas/gallery.py::GalleryGroupDetail (D6).
#
# Deliberately NOT importing from schemas/gallery.py — gallery schemas are
# removed in prompt 03; sharing types here would block that deletion.
# ---------------------------------------------------------------------------


class LibraryOutputItem(msgspec.Struct, kw_only=True):
    """Single output within a library group detail view."""

    id: UUID

    asset_ref: str
    """Wire-format asset reference for this output — always ``"output:<id>"``."""

    output_index: int

    created_at: datetime

    expires_at: datetime
    """UTC timestamp when this output is deleted by retention cleanup."""

    media: MediaObject


class LibraryGroupLineage(msgspec.Struct, kw_only=True):
    """Job-level remix lineage — which asset this generation job was based on."""

    source_type: LibraryGroupSourceType

    source_upload_id: UUID | None = None
    """If source was a direct upload."""

    source_job_id: UUID | None = None
    """If source was a previous generation's output."""

    source_job_name: str | None = None
    """Human-readable name of the source job."""

    source_output_id: UUID | None = None
    """Specific output used as input."""


class LibraryGroupDetail(msgspec.Struct, kw_only=True):
    """Full detail view of a generation group (job) — relocated from GalleryGroupDetail."""

    job_id: UUID

    badge: LibraryBadge
    """'image' if input-driven, 'prompt' if text-only."""

    input_media: MediaObject | None = None
    """Media envelope for the source input (upload or remixed output).
    Present when badge == 'image'."""

    prompt: str
    negative_prompt: str | None = None

    outputs: list[LibraryOutputItem]
    """All non-thumbnail outputs, ordered by output_index."""

    media_type: OutputMediaType
    model: str | None = None
    provider: str
    generation_type: GenerationType
    aspect_ratio: str | None = None
    token_cost: int | None = None
    created_at: datetime
    completed_at: datetime | None = None

    lineage: LibraryGroupLineage | None = None
