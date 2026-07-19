"""Library read-model queries — UNION of uploads and outputs with per-asset metadata.

The list query unions two independently-filtered branches (one per source
table) ranked against each other via a fixed per-branch ``source_rank``
constant, so a single keyset cursor can page across both tables in a stable
``created_at DESC, source_rank DESC, id DESC`` order. See ``list_assets``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import (
    Integer,
    String,
    and_,
    cast,
    delete,
    false,
    func,
    literal,
    null,
    or_,
    select,
    text,
    tuple_,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import Select
from sqlalchemy.sql import union_all as sa_union_all

from src.core.enums import JobStatus, LibrarySort, OutputMediaType
from src.core.library_ref import LibraryAssetSource
from src.core.uid import new_id
from src.db.models.library import LibraryAssetMetadata
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class _UnsetUpdate:
    """Sentinel for an omitted metadata field update (not the same as an explicit None)."""

    __slots__ = ()


UNSET_UPDATE = _UnsetUpdate()
type OptionalUpdate[T] = T | _UnsetUpdate

# Fixed per-source ranking constants. Arbitrary but distinct — only used to
# keep the union's ORDER BY / keyset comparison deterministic across the two
# branches; the actual values carry no meaning outside this module.
_SOURCE_RANK: dict[LibraryAssetSource, int] = {
    LibraryAssetSource.UPLOAD: 0,
    LibraryAssetSource.OUTPUT: 1,
}


@dataclass
class LibraryAssetRow:
    """Normalized row shape shared by both UNION branches."""

    source: LibraryAssetSource
    id: UUID
    created_at: datetime
    expires_at: datetime
    content_type: str
    size_bytes: int
    width: int | None
    height: int | None
    duration_ms: int | None
    original_filename: str | None
    job_id: UUID | None
    model: str | None
    generation_type: str | None
    display_title: str | None
    is_favorite: bool


class LibraryRepository:
    """Data access layer for the unified library read model."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    # List
    # -------------------------------------------------------------------------

    async def list_assets(
        self,
        user_id: UUID,
        product_id: str,
        *,
        limit: int,
        cursor: tuple[datetime, str, UUID] | None = None,
        source: LibraryAssetSource | None = None,
        media_type: OutputMediaType | None = None,
        model: str | None = None,
        favorite: bool | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        sort: LibrarySort = LibrarySort.NEWEST,
    ) -> Sequence[LibraryAssetRow]:
        """List library assets (uploads + outputs) via a UNION ALL keyset query.

        Uses the limit+1 fetch pattern — caller checks ``len(result) > limit``
        to determine ``has_more``.

        Args:
            user_id: Owner of the assets.
            product_id: Product scope.
            limit: Page size (fetches limit+1 internally).
            cursor: ``(created_at, source, id)`` of the last item on the
                previous page — see ``schemas.pagination.decode_library_cursor``.
            source: Restrict to a single source table.
            media_type: Filter to image or video content types.
            model: Filter to a specific generation model. Implies
                ``source=OUTPUT`` semantics — uploads never match a model
                filter and are excluded from the branch set entirely.
            favorite: Filter to favorited (True) or non-favorited (False) assets.
            created_from: Lower bound (inclusive) on ``created_at``.
            created_to: Upper bound (inclusive) on ``created_at``.
            sort: ``newest`` (default) or ``oldest``.

        Returns:
            Sequence of LibraryAssetRow, ordered per ``sort``.
        """
        cursor_ts: datetime | None = None
        cursor_rank: int | None = None
        cursor_id: UUID | None = None
        if cursor is not None:
            cursor_ts, cursor_source_raw, cursor_id = cursor
            cursor_rank = _SOURCE_RANK[LibraryAssetSource(cursor_source_raw)]

        include_upload = model is None and source in (None, LibraryAssetSource.UPLOAD)
        include_output = source in (None, LibraryAssetSource.OUTPUT)

        branches: list[Select] = []  # type: ignore[type-arg]
        if include_upload:
            branches.append(
                self._build_upload_branch(
                    user_id,
                    product_id,
                    media_type=media_type,
                    favorite=favorite,
                    created_from=created_from,
                    created_to=created_to,
                    cursor_ts=cursor_ts,
                    cursor_rank=cursor_rank,
                    cursor_id=cursor_id,
                    sort=sort,
                )
            )
        if include_output:
            branches.append(
                self._build_output_branch(
                    user_id,
                    product_id,
                    media_type=media_type,
                    model=model,
                    favorite=favorite,
                    created_from=created_from,
                    created_to=created_to,
                    cursor_ts=cursor_ts,
                    cursor_rank=cursor_rank,
                    cursor_id=cursor_id,
                    sort=sort,
                )
            )

        if not branches:
            return []

        union_query = sa_union_all(*branches) if len(branches) > 1 else branches[0]
        subq = union_query.subquery()

        order_cols = (
            (subq.c.created_at.desc(), subq.c.source_rank.desc(), subq.c.id.desc())
            if sort == LibrarySort.NEWEST
            else (subq.c.created_at.asc(), subq.c.source_rank.asc(), subq.c.id.asc())
        )
        final_query = select(subq).order_by(*order_cols).limit(limit + 1)

        result = await self._session.execute(final_query)
        rows = result.mappings().all()
        return [
            LibraryAssetRow(
                source=LibraryAssetSource(row["source"]),
                id=row["id"],
                created_at=row["created_at"],
                expires_at=row["expires_at"],
                content_type=row["content_type"],
                size_bytes=row["size_bytes"],
                width=row["width"],
                height=row["height"],
                duration_ms=row["duration_ms"],
                original_filename=row["original_filename"],
                job_id=row["job_id"],
                model=row["model"],
                generation_type=row["generation_type"],
                display_title=row["display_title"],
                is_favorite=bool(row["is_favorite"]),
            )
            for row in rows
        ]

    def _build_upload_branch(
        self,
        user_id: UUID,
        product_id: str,
        *,
        media_type: OutputMediaType | None,
        favorite: bool | None,
        created_from: datetime | None,
        created_to: datetime | None,
        cursor_ts: datetime | None,
        cursor_rank: int | None,
        cursor_id: UUID | None,
        sort: LibrarySort,
    ) -> Select:  # type: ignore[type-arg]
        rank = _SOURCE_RANK[LibraryAssetSource.UPLOAD]
        meta_join_cond = and_(
            LibraryAssetMetadata.user_id == user_id,
            LibraryAssetMetadata.product_id == product_id,
            LibraryAssetMetadata.asset_type == LibraryAssetSource.UPLOAD.value,
            LibraryAssetMetadata.asset_id == UserImage.id,
        )

        query = (
            select(
                literal(LibraryAssetSource.UPLOAD.value).label("source"),
                UserImage.id.label("id"),
                UserImage.created_at.label("created_at"),
                UserImage.expires_at.label("expires_at"),
                UserImage.content_type.label("content_type"),
                UserImage.size_bytes.label("size_bytes"),
                UserImage.width.label("width"),
                UserImage.height.label("height"),
                UserImage.duration_ms.label("duration_ms"),
                UserImage.original_filename.label("original_filename"),
                cast(null(), PG_UUID(as_uuid=True)).label("job_id"),
                cast(null(), String(50)).label("model"),
                cast(null(), String(20)).label("generation_type"),
                LibraryAssetMetadata.display_title.label("display_title"),
                func.coalesce(LibraryAssetMetadata.is_favorite, false()).label("is_favorite"),
                literal(rank).label("source_rank"),
            )
            .outerjoin(LibraryAssetMetadata, meta_join_cond)
            .where(
                UserImage.user_id == user_id,
                UserImage.product_id == product_id,
                UserImage.is_thumbnail.is_(False),
            )
        )

        if media_type == OutputMediaType.VIDEO:
            query = query.where(UserImage.content_type.like("video/%"))
        elif media_type == OutputMediaType.IMAGE:
            query = query.where(~UserImage.content_type.like("video/%"))

        if favorite is True:
            query = query.where(LibraryAssetMetadata.is_favorite.is_(True))
        elif favorite is False:
            query = query.where(func.coalesce(LibraryAssetMetadata.is_favorite, false()).is_(False))

        if created_from is not None:
            query = query.where(UserImage.created_at >= created_from)
        if created_to is not None:
            query = query.where(UserImage.created_at <= created_to)

        if cursor_ts is not None and cursor_rank is not None and cursor_id is not None:
            current = tuple_(UserImage.created_at, literal(rank), UserImage.id)
            cursor_tuple = tuple_(literal(cursor_ts), literal(cursor_rank), literal(cursor_id))
            query = query.where(
                current < cursor_tuple if sort == LibrarySort.NEWEST else current > cursor_tuple
            )

        return query

    def _build_output_branch(
        self,
        user_id: UUID,
        product_id: str,
        *,
        media_type: OutputMediaType | None,
        model: str | None,
        favorite: bool | None,
        created_from: datetime | None,
        created_to: datetime | None,
        cursor_ts: datetime | None,
        cursor_rank: int | None,
        cursor_id: UUID | None,
        sort: LibrarySort,
    ) -> Select:  # type: ignore[type-arg]
        rank = _SOURCE_RANK[LibraryAssetSource.OUTPUT]
        meta_join_cond = and_(
            LibraryAssetMetadata.user_id == user_id,
            LibraryAssetMetadata.product_id == product_id,
            LibraryAssetMetadata.asset_type == LibraryAssetSource.OUTPUT.value,
            LibraryAssetMetadata.asset_id == GenerationOutput.id,
        )

        query = (
            select(
                literal(LibraryAssetSource.OUTPUT.value).label("source"),
                GenerationOutput.id.label("id"),
                GenerationOutput.created_at.label("created_at"),
                GenerationOutput.expires_at.label("expires_at"),
                GenerationOutput.content_type.label("content_type"),
                GenerationOutput.size_bytes.label("size_bytes"),
                GenerationOutput.width.label("width"),
                GenerationOutput.height.label("height"),
                cast(null(), Integer).label("duration_ms"),
                cast(null(), String(255)).label("original_filename"),
                GenerationOutput.job_id.label("job_id"),
                GenerationJob.model.label("model"),
                GenerationJob.generation_type.label("generation_type"),
                LibraryAssetMetadata.display_title.label("display_title"),
                func.coalesce(LibraryAssetMetadata.is_favorite, false()).label("is_favorite"),
                literal(rank).label("source_rank"),
            )
            .join(GenerationJob, GenerationJob.id == GenerationOutput.job_id)
            .outerjoin(LibraryAssetMetadata, meta_join_cond)
            .where(
                GenerationOutput.user_id == user_id,
                GenerationOutput.product_id == product_id,
                GenerationOutput.is_thumbnail.is_(False),
                GenerationJob.status == JobStatus.COMPLETED,
                GenerationJob.is_deleted.is_(False),
            )
        )

        if media_type == OutputMediaType.VIDEO:
            query = query.where(GenerationOutput.content_type.like("video/%"))
        elif media_type == OutputMediaType.IMAGE:
            query = query.where(~GenerationOutput.content_type.like("video/%"))

        if model is not None:
            query = query.where(GenerationJob.model == model)

        if favorite is True:
            query = query.where(LibraryAssetMetadata.is_favorite.is_(True))
        elif favorite is False:
            query = query.where(func.coalesce(LibraryAssetMetadata.is_favorite, false()).is_(False))

        if created_from is not None:
            query = query.where(GenerationOutput.created_at >= created_from)
        if created_to is not None:
            query = query.where(GenerationOutput.created_at <= created_to)

        if cursor_ts is not None and cursor_rank is not None and cursor_id is not None:
            current = tuple_(GenerationOutput.created_at, literal(rank), GenerationOutput.id)
            cursor_tuple = tuple_(literal(cursor_ts), literal(cursor_rank), literal(cursor_id))
            query = query.where(
                current < cursor_tuple if sort == LibrarySort.NEWEST else current > cursor_tuple
            )

        return query

    # -------------------------------------------------------------------------
    # Output-count batch (list path — grouped COUNT, not per-row)
    # -------------------------------------------------------------------------

    async def batch_output_counts(self, job_ids: Sequence[UUID]) -> dict[UUID, int]:
        """Count non-thumbnail outputs per job, for a batch of job ids.

        Args:
            job_ids: Job ids appearing in the current page's output rows.

        Returns:
            Mapping from job_id to output count. Job ids with zero counted
            outputs are absent from the result.
        """
        if not job_ids:
            return {}

        result = await self._session.execute(
            select(GenerationOutput.job_id, func.count(GenerationOutput.id))
            .where(
                GenerationOutput.job_id.in_(job_ids),
                GenerationOutput.is_thumbnail.is_(False),
            )
            .group_by(GenerationOutput.job_id)
        )
        return dict(result.tuples().all())

    # -------------------------------------------------------------------------
    # Group detail + descendants
    # -------------------------------------------------------------------------

    async def get_group_job(
        self,
        job_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> GenerationJob | None:
        """Get a single completed job with eager-loaded relationships for group detail.

        Mirrors GalleryRepository.get_gallery_job byte-for-byte — copied
        rather than imported so Library stays independent of the Gallery
        module, which is removed in a later phase.

        Args:
            job_id: Job to fetch.
            user_id: Owner check.
            product_id: Product scope check.

        Returns:
            GenerationJob with relationships loaded; None if not found.
        """
        result = await self._session.execute(
            select(GenerationJob)
            .where(
                GenerationJob.id == job_id,
                GenerationJob.user_id == user_id,
                GenerationJob.product_id == product_id,
                GenerationJob.status == JobStatus.COMPLETED,
                GenerationJob.is_deleted.is_(False),
            )
            .options(
                selectinload(GenerationJob.outputs),
                selectinload(GenerationJob.source_job),
                selectinload(GenerationJob.source_output).selectinload(
                    GenerationOutput.derivatives
                ),
                selectinload(GenerationJob.input_image).selectinload(UserImage.derivatives),
            )
        )
        return result.scalar_one_or_none()

    async def count_descendants(
        self,
        source: LibraryAssetSource,
        asset_id: UUID,
    ) -> tuple[int, int]:
        """Count descendants of a single asset: two aggregate queries, max.

        Args:
            source: Which table ``asset_id`` belongs to.
            asset_id: The asset's primary key.

        Returns:
            ``(job_count, frame_count)`` — jobs that used this asset as
            input, and extracted-frame uploads sourced from this asset.
        """
        if source == LibraryAssetSource.UPLOAD:
            job_count_query = select(func.count(GenerationJob.id)).where(
                GenerationJob.input_image_id == asset_id
            )
            frame_count_query = select(func.count(UserImage.id)).where(
                UserImage.source_upload_id == asset_id
            )
        else:
            job_count_query = select(func.count(GenerationJob.id)).where(
                GenerationJob.source_output_id == asset_id
            )
            frame_count_query = select(func.count(UserImage.id)).where(
                UserImage.source_output_id == asset_id
            )

        job_count = (await self._session.execute(job_count_query)).scalar_one()
        frame_count = (await self._session.execute(frame_count_query)).scalar_one()
        return job_count, frame_count

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    async def get_metadata(
        self,
        user_id: UUID,
        product_id: str,
        source: LibraryAssetSource,
        asset_id: UUID,
    ) -> LibraryAssetMetadata | None:
        """Fetch the metadata row for a single asset, if one exists yet."""
        result = await self._session.execute(
            select(LibraryAssetMetadata).where(
                LibraryAssetMetadata.user_id == user_id,
                LibraryAssetMetadata.product_id == product_id,
                LibraryAssetMetadata.asset_type == source.value,
                LibraryAssetMetadata.asset_id == asset_id,
            )
        )
        return result.scalar_one_or_none()

    async def upsert_metadata(
        self,
        user_id: UUID,
        product_id: str,
        source: LibraryAssetSource,
        asset_id: UUID,
        *,
        is_favorite: OptionalUpdate[bool] = UNSET_UPDATE,
        display_title: OptionalUpdate[str | None] = UNSET_UPDATE,
    ) -> LibraryAssetMetadata:
        """Race-safe lazy create-or-update of a library_asset_metadata row.

        Ownership of the underlying asset must be verified by the caller
        BEFORE calling this — this method only enforces the DB-level
        uniqueness of (product_id, user_id, asset_type, asset_id).

        Args:
            user_id: Owner.
            product_id: Product scope.
            source: Which table the asset belongs to.
            asset_id: The asset's primary key.
            is_favorite: New favorite value, or ``UNSET_UPDATE`` to leave
                unchanged (defaults to ``False`` on first insert).
            display_title: New display title (``None`` clears it), or
                ``UNSET_UPDATE`` to leave unchanged (defaults to ``None`` on
                first insert).

        Returns:
            The resulting LibraryAssetMetadata row.
        """
        insert_is_favorite = False if isinstance(is_favorite, _UnsetUpdate) else is_favorite
        insert_display_title = None if isinstance(display_title, _UnsetUpdate) else display_title

        update_values: dict[str, object] = {"updated_at": text("CURRENT_TIMESTAMP")}
        if not isinstance(is_favorite, _UnsetUpdate):
            update_values["is_favorite"] = is_favorite
        if not isinstance(display_title, _UnsetUpdate):
            update_values["display_title"] = display_title

        stmt = (
            pg_insert(LibraryAssetMetadata)
            .values(
                id=new_id(),
                product_id=product_id,
                user_id=user_id,
                asset_type=source.value,
                asset_id=asset_id,
                is_favorite=insert_is_favorite,
                display_title=insert_display_title,
            )
            .on_conflict_do_update(
                constraint="uq_library_asset_metadata_asset",
                set_=update_values,
            )
            .returning(LibraryAssetMetadata)
            .execution_options(populate_existing=True)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one()
        await self._session.flush()
        return row

    async def delete_metadata_for_assets(self, pairs: Sequence[tuple[str, UUID]]) -> int:
        """Bulk-delete metadata rows matching (asset_type, asset_id) pairs.

        Deliberately not scoped to (user_id, product_id) — callers (asset
        deletion, retention sweep) already establish ownership/scope before
        collecting the pairs to purge, and a bulk ``IN`` over composite pairs
        is simplest expressed as an OR of per-pair AND clauses.

        Args:
            pairs: (asset_type, asset_id) tuples to purge.

        Returns:
            Number of rows deleted.
        """
        if not pairs:
            return 0

        conditions = [
            and_(
                LibraryAssetMetadata.asset_type == asset_type,
                LibraryAssetMetadata.asset_id == asset_id,
            )
            for asset_type, asset_id in pairs
        ]
        result = await self._session.execute(delete(LibraryAssetMetadata).where(or_(*conditions)))
        return result.rowcount or 0  # type: ignore[attr-defined]
