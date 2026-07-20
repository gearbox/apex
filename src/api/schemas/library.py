"""Library endpoint schemas — asset items, detail views, group detail, patch."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
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

    project_id: UUID | None = None
    """Project this asset is assigned to, if any (P1: at most one)."""

    project_name: str | None = None
    """Denormalized name of ``project_id``, resolved via a batched lookup."""


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
    """Request to update mutable library asset fields."""

    display_title: str | None | msgspec.UnsetType = msgspec.UNSET
    """Absent = leave unchanged. ``null`` = clear. String = set (max 255 chars)."""

    project_id: UUID | None | msgspec.UnsetType = msgspec.UNSET
    """Absent = leave unchanged. ``null`` = unassign. UUID = assign (must be owned by caller)."""


# ---------------------------------------------------------------------------
# Projects (Phase 2)
# ---------------------------------------------------------------------------


class LibraryProject(msgspec.Struct, kw_only=True):
    """A user-created grouping for library assets."""

    id: UUID
    name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime


class LibraryProjectListItem(LibraryProject, kw_only=True):
    """Single row in the project list view — adds the batched asset count."""

    asset_count: int


class LibraryProjectCreate(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to create a new project."""

    name: Annotated[str, msgspec.Meta(min_length=1, max_length=100)]
    description: str | None = None


class LibraryProjectPatch(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to rename/redescribe a project. Both fields are tri-state."""

    name: Annotated[str, msgspec.Meta(min_length=1, max_length=100)] | msgspec.UnsetType = (
        msgspec.UNSET
    )
    description: str | None | msgspec.UnsetType = msgspec.UNSET


# ---------------------------------------------------------------------------
# Bulk operations (Phase 2)
# ---------------------------------------------------------------------------

_BulkAssetRefs = Annotated[list[str], msgspec.Meta(min_length=1, max_length=100)]
"""Shared bound for every bulk op's asset_refs — never more than 100 (P4)."""


class BulkSetFavorite(msgspec.Struct, tag="set_favorite", kw_only=True):
    """Bulk-set (or clear) the favorite flag on a batch of assets."""

    asset_refs: _BulkAssetRefs
    value: bool


class BulkSetProject(msgspec.Struct, tag="set_project", kw_only=True):
    """Bulk-assign (or unassign, if ``project_id`` is ``null``) a batch of assets."""

    asset_refs: _BulkAssetRefs
    project_id: UUID | None


class BulkDelete(msgspec.Struct, tag="delete", kw_only=True):
    """Bulk-delete a batch of assets."""

    asset_refs: _BulkAssetRefs


BulkOperation = BulkSetFavorite | BulkSetProject | BulkDelete
"""Tagged union decoded from the ``op`` discriminant field — see P4."""


class BulkOperationItemResult(msgspec.Struct, kw_only=True):
    """Per-item outcome of one asset_ref within a bulk operation."""

    asset_ref: str
    success: bool


class BulkOperationResult(msgspec.Struct, kw_only=True):
    """Response body for POST /v1/library/assets/bulk.

    Only returned once every asset_ref has passed validation (P5) — a
    validation failure never reaches this shape, it short-circuits to a
    400 listing the offending refs instead.
    """

    op: str
    results: list[BulkOperationItemResult]
    succeeded: int
    failed: int


# ---------------------------------------------------------------------------
# Group detail — ported from the now-removed schemas/gallery.py::GalleryGroupDetail.
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
    """Full detail view of a generation group (job)."""

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
