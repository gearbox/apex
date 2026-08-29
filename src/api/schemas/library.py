"""Library endpoint schemas — asset items, detail views, group detail, patch."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID

import annotated_types
import msgspec

from src.api.schemas.media import MediaObject
from src.api.services.library_capabilities import LibraryAction
from src.core.enums import GenerationType, LibraryBadge, LibraryGroupSourceType, MediaKind
from src.core.library_limits import MAX_TAGS_PER_ASSET
from src.core.library_ref import LibraryAssetSource


class LibraryTagRef(msgspec.Struct, kw_only=True):
    """Minimal tag reference embedded in asset items — id + name only."""

    id: UUID
    name: str


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
    """Upload-only: canonical system filename (``{uuid}.{ext}``), kept for compatibility."""

    display_filename: str | None = None
    """Upload-only: sanitized, human-readable original filename, for display/search."""

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

    tags: tuple[LibraryTagRef, ...] = ()
    """Tags assigned to this asset (T1: many-to-many), name-ordered."""


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

    # The redundant annotated_types.MaxLen marker gets the bound onto the
    # oneOf branch's own schema, not just the wrapper — see the S2 comment
    # on LibraryTagPatch.name for why msgspec.Meta alone isn't enough here.
    tag_ids: (
        Annotated[
            list[UUID],
            msgspec.Meta(max_length=MAX_TAGS_PER_ASSET),
            annotated_types.MaxLen(MAX_TAGS_PER_ASSET),
        ]
        | msgspec.UnsetType
    ) = msgspec.UNSET
    """Absent = leave unchanged. Replace-set semantics: ``[]`` clears every
    tag, a list sets the exact tag set (<=20, all must be owned by caller).
    No ``None`` arm — clearing an asset's tags is expressed as ``[]``."""


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


# The redundant annotated_types.MinLen/MaxLen markers are inert for msgspec
# itself (validation is governed entirely by msgspec.Meta) but are what
# Litestar's OpenAPI generator recognizes when building the inner member of
# a tri-state field's `oneOf` — without them the bounds only land on the
# wrapper schema (a sibling of `oneOf`), never on the `{"type": "string"}`
# branch a oneOf-strict validator/codegen tool actually inspects. Shared by
# *Create and *Patch so the bound lives in exactly one place.
_ProjectName = Annotated[
    str,
    msgspec.Meta(min_length=1, max_length=100),
    annotated_types.MinLen(1),
    annotated_types.MaxLen(100),
]


class LibraryProjectCreate(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to create a new project."""

    name: _ProjectName
    description: str | None = None


class LibraryProjectPatch(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to rename/redescribe a project. Both fields are tri-state."""

    name: _ProjectName | msgspec.UnsetType = msgspec.UNSET
    description: str | None | msgspec.UnsetType = msgspec.UNSET


# ---------------------------------------------------------------------------
# Tags (Phase 3)
# ---------------------------------------------------------------------------


class LibraryTag(msgspec.Struct, kw_only=True):
    """A user-created tag, assignable to many assets (T1: many-to-many)."""

    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime


class LibraryTagListItem(LibraryTag, kw_only=True):
    """Single row in the tag list view — adds the batched asset count."""

    asset_count: int


# See the _ProjectName comment above for why the annotated_types markers
# are needed alongside msgspec.Meta. Shared by LibraryTagCreate/Patch so the
# bound lives in exactly one place.
_TagName = Annotated[
    str,
    msgspec.Meta(min_length=1, max_length=50),
    annotated_types.MinLen(1),
    annotated_types.MaxLen(50),
]


class LibraryTagCreate(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to create a new tag."""

    name: _TagName


class LibraryTagPatch(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request to rename a tag."""

    name: _TagName | msgspec.UnsetType = msgspec.UNSET


# ---------------------------------------------------------------------------
# Bulk operations (Phase 2/3)
# ---------------------------------------------------------------------------

_BulkAssetRefs = Annotated[list[str], msgspec.Meta(min_length=1, max_length=100)]
"""Shared bound for every bulk op's asset_refs — never more than 100 (P4)."""

_BulkTagIds = Annotated[list[UUID], msgspec.Meta(min_length=1, max_length=10)]
"""Shared bound for bulk tag ops' tag_ids — never more than 10 (T5)."""


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


class BulkAddTags(msgspec.Struct, tag="add_tags", kw_only=True):
    """Bulk-add a batch of tags to a batch of assets."""

    asset_refs: _BulkAssetRefs
    tag_ids: _BulkTagIds


class BulkRemoveTags(msgspec.Struct, tag="remove_tags", kw_only=True):
    """Bulk-remove a batch of tags from a batch of assets."""

    asset_refs: _BulkAssetRefs
    tag_ids: _BulkTagIds


BulkOperation = BulkSetFavorite | BulkSetProject | BulkDelete | BulkAddTags | BulkRemoveTags
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


class LibrarySourceMediaItem(msgspec.Struct, kw_only=True):
    """One ordered source used by a generation job.

    A source row outlives the referenced asset: retention nulls the foreign
    key but preserves ``asset_ref`` and its position, allowing clients to
    show an unavailable placeholder instead of replaying a changed request.
    """

    position: int
    asset_ref: str
    available: bool
    media: MediaObject | None = None
    """Resolved media envelope; absent when the source is unavailable."""


class LibraryGroupDetail(msgspec.Struct, kw_only=True):
    """Full detail view of a generation group (job)."""

    job_id: UUID

    badge: LibraryBadge
    """'image' if input-driven, 'prompt' if text-only."""

    input_media: MediaObject | None = None
    """Media envelope for the source input (upload or remixed output).
    Present when badge == 'image'."""

    source_media: list[LibrarySourceMediaItem] = msgspec.field(default_factory=list)
    """Ordered source assets used for this job, including unavailable positions."""

    prompt: str
    negative_prompt: str | None = None

    outputs: list[LibraryOutputItem]
    """All non-thumbnail outputs, ordered by output_index."""

    media_type: MediaKind
    model: str | None = None
    provider: str
    generation_type: GenerationType
    aspect_ratio: str | None = None
    token_cost: int | None = None
    created_at: datetime
    completed_at: datetime | None = None

    lineage: LibraryGroupLineage | None = None


# ---------------------------------------------------------------------------
# Lineage graph (Phase 3) — GET /v1/library/assets/{asset_ref}/lineage
# ---------------------------------------------------------------------------


class LineageRelation(StrEnum):
    """How a lineage edge's node relates to the node it's attached to."""

    GENERATED_FROM_UPLOAD = "generated_from_upload"
    """Ancestor edge: an output's job used this upload as its input image."""

    GENERATED_FROM_OUTPUT = "generated_from_output"
    """Ancestor/descendant edge: a job remixed this output as its source."""

    FRAME_OF_OUTPUT = "frame_of_output"
    """Ancestor/descendant edge: an upload is a frame extracted from this output."""

    FRAME_OF_UPLOAD = "frame_of_upload"
    """Ancestor/descendant edge: an upload is a frame extracted from this upload."""


class LineageNode(msgspec.Struct, kw_only=True):
    """Single asset in a lineage graph — enough to render a thumbnail + link."""

    asset_ref: str
    source: LibraryAssetSource
    media: MediaObject
    created_at: datetime
    model: str | None = None
    """Output-only: generation model."""

    generation_type: GenerationType | None = None
    """Output-only."""


class LineageEdge(msgspec.Struct, kw_only=True):
    """One hop in the lineage graph: the relation plus the node it leads to."""

    relation: LineageRelation
    node: LineageNode
    source_timestamp_ms: int | None = None
    """Frame edges only: timestamp within the source video this frame was extracted at."""


class LibraryLineageGraph(msgspec.Struct, kw_only=True):
    """Bounded ancestor/descendant graph for a single library asset (T8)."""

    focus: LineageNode

    ancestors: tuple[LineageEdge, ...]
    """Nearest-first chain, one parent per step, depth-capped."""

    descendants: tuple[LineageEdge, ...]
    """Immediate descendants only (not recursive), per-relation capped."""

    descendant_totals: LibraryDescendants
    """Full descendant counts, independent of the capped ``descendants`` list."""

    ancestors_truncated: bool
    """True if the ancestor walk stopped at the depth cap rather than a real root."""

    descendants_truncated: bool
    """True if either descendant relation was clipped by its per-relation cap."""
