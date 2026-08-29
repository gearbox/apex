"""Library read-model queries — UNION of uploads and outputs with per-asset metadata.

The list query unions two independently-filtered branches (one per source
table) ranked against each other via a fixed per-branch ``source_rank``
constant, so a single keyset cursor can page across both tables in a stable
``created_at DESC, source_rank DESC, id DESC`` order. See ``list_assets``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import (
    String,
    and_,
    cast,
    delete,
    exists,
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

from src.core.enums import JobStatus, LibrarySort, MediaKind
from src.core.library_ref import AssetRef, LibraryAssetSource
from src.core.uid import new_id
from src.db.models.library import LibraryAssetMetadata, LibraryAssetTag
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)

# Window used by the ``expiring`` filter / ``expiring_soon`` sort (P6).
_EXPIRING_SOON_WINDOW = timedelta(days=7)


def _escape_like_term(term: str) -> str:
    """Escape LIKE metacharacters so user search input is matched literally.

    Order matters: the escape character itself must be escaped first, or a
    literal backslash in the input would be mis-paired with the escaping of
    ``%``/``_`` that follows.
    """
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


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
    display_filename: str | None
    job_id: UUID | None
    model: str | None
    generation_type: str | None
    display_title: str | None
    is_favorite: bool
    project_id: UUID | None


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
        media_type: MediaKind | None = None,
        model: str | None = None,
        favorite: bool | None = None,
        project_id: UUID | None = None,
        tag_id: UUID | None = None,
        expiring: bool | None = None,
        query: str | None = None,
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
            cursor: ``(created_at_or_expires_at, source, id)`` of the last
                item on the previous page — see
                ``schemas.pagination.decode_library_cursor``. The first
                component is ``expires_at`` when ``sort=EXPIRING_SOON``,
                otherwise ``created_at``.
            source: Restrict to a single source table.
            media_type: Filter to image or video content types.
            model: Filter to a specific generation model. Implies
                ``source=OUTPUT`` semantics — uploads never match a model
                filter and are excluded from the branch set entirely.
            favorite: Filter to favorited (True) or non-favorited (False) assets.
            project_id: Filter to assets assigned to this project.
            tag_id: Filter to assets tagged with this tag (T6: single tag only).
            expiring: Filter to assets expiring within (True) or beyond
                (False) the ``expiring_soon`` window (P6, 7 days).
            query: Case-insensitive substring search over display_title,
                display_filename (falling back to original_filename on
                pre-030 uploads), and the owning job's prompt (outputs).
                Escaped before use — never interpolated raw.
            created_from: Lower bound (inclusive) on ``created_at``.
            created_to: Upper bound (inclusive) on ``created_at``.
            sort: ``newest`` (default), ``oldest``, or ``expiring_soon``.

        Returns:
            Sequence of LibraryAssetRow, ordered per ``sort``.
        """
        cursor_ts: datetime | None = None
        cursor_rank: int | None = None
        cursor_id: UUID | None = None
        if cursor is not None:
            cursor_ts, cursor_source_raw, cursor_id = cursor
            cursor_rank = _SOURCE_RANK[LibraryAssetSource(cursor_source_raw)]

        expiring_threshold = datetime.now(UTC) + _EXPIRING_SOON_WINDOW
        search_term = _escape_like_term(query) if query else None

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
                    project_id=project_id,
                    tag_id=tag_id,
                    expiring=expiring,
                    expiring_threshold=expiring_threshold,
                    search_term=search_term,
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
                    project_id=project_id,
                    tag_id=tag_id,
                    expiring=expiring,
                    expiring_threshold=expiring_threshold,
                    search_term=search_term,
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

        if sort == LibrarySort.NEWEST:
            order_cols = (subq.c.created_at.desc(), subq.c.source_rank.desc(), subq.c.id.desc())
        elif sort == LibrarySort.OLDEST:
            order_cols = (subq.c.created_at.asc(), subq.c.source_rank.asc(), subq.c.id.asc())
        else:
            order_cols = (subq.c.expires_at.asc(), subq.c.source_rank.asc(), subq.c.id.asc())
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
                display_filename=row["display_filename"],
                job_id=row["job_id"],
                model=row["model"],
                generation_type=row["generation_type"],
                display_title=row["display_title"],
                is_favorite=bool(row["is_favorite"]),
                project_id=row["project_id"],
            )
            for row in rows
        ]

    def _build_upload_branch(
        self,
        user_id: UUID,
        product_id: str,
        *,
        media_type: MediaKind | None,
        favorite: bool | None,
        project_id: UUID | None,
        tag_id: UUID | None,
        expiring: bool | None,
        expiring_threshold: datetime,
        search_term: str | None,
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
                UserImage.display_filename.label("display_filename"),
                cast(null(), PG_UUID(as_uuid=True)).label("job_id"),
                cast(null(), String(50)).label("model"),
                cast(null(), String(20)).label("generation_type"),
                LibraryAssetMetadata.display_title.label("display_title"),
                func.coalesce(LibraryAssetMetadata.is_favorite, false()).label("is_favorite"),
                LibraryAssetMetadata.project_id.label("project_id"),
                literal(rank).label("source_rank"),
            )
            .outerjoin(LibraryAssetMetadata, meta_join_cond)
            .where(
                UserImage.user_id == user_id,
                UserImage.product_id == product_id,
                UserImage.is_thumbnail.is_(False),
            )
        )

        if media_type == MediaKind.VIDEO:
            query = query.where(UserImage.content_type.like("video/%"))
        elif media_type == MediaKind.IMAGE:
            query = query.where(~UserImage.content_type.like("video/%"))

        if favorite is True:
            query = query.where(LibraryAssetMetadata.is_favorite.is_(True))
        elif favorite is False:
            query = query.where(func.coalesce(LibraryAssetMetadata.is_favorite, false()).is_(False))

        if project_id is not None:
            query = query.where(LibraryAssetMetadata.project_id == project_id)

        if tag_id is not None:
            query = query.where(
                exists(
                    select(LibraryAssetTag.tag_id).where(
                        LibraryAssetTag.asset_type == LibraryAssetSource.UPLOAD.value,
                        LibraryAssetTag.asset_id == UserImage.id,
                        LibraryAssetTag.tag_id == tag_id,
                        LibraryAssetTag.user_id == user_id,
                        LibraryAssetTag.product_id == product_id,
                    )
                )
            )

        if expiring is True:
            query = query.where(UserImage.expires_at <= expiring_threshold)
        elif expiring is False:
            query = query.where(UserImage.expires_at > expiring_threshold)

        if search_term is not None:
            pattern = f"%{search_term}%"
            query = query.where(
                or_(
                    # display_filename is the sanitized client name on rows written
                    # since migration 030; it is NULL for every upload predating it,
                    # whose real name still lives in original_filename. COALESCE (not
                    # OR) keeps those searchable without also matching the {uuid}.{ext}
                    # canonical name on new rows.
                    func.coalesce(
                        UserImage.display_filename,
                        UserImage.original_filename,
                    ).ilike(pattern, escape="\\"),
                    LibraryAssetMetadata.display_title.ilike(pattern, escape="\\"),
                )
            )

        if created_from is not None:
            query = query.where(UserImage.created_at >= created_from)
        if created_to is not None:
            query = query.where(UserImage.created_at <= created_to)

        if cursor_ts is not None and cursor_rank is not None and cursor_id is not None:
            ts_col = (
                UserImage.expires_at if sort == LibrarySort.EXPIRING_SOON else UserImage.created_at
            )
            ascending = sort in (LibrarySort.OLDEST, LibrarySort.EXPIRING_SOON)
            current = tuple_(ts_col, literal(rank), UserImage.id)
            cursor_tuple = tuple_(literal(cursor_ts), literal(cursor_rank), literal(cursor_id))
            query = query.where(current > cursor_tuple if ascending else current < cursor_tuple)

        return query

    def _build_output_branch(
        self,
        user_id: UUID,
        product_id: str,
        *,
        media_type: MediaKind | None,
        model: str | None,
        favorite: bool | None,
        project_id: UUID | None,
        tag_id: UUID | None,
        expiring: bool | None,
        expiring_threshold: datetime,
        search_term: str | None,
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
                GenerationOutput.duration_ms.label("duration_ms"),
                cast(null(), String(255)).label("original_filename"),
                cast(null(), String(255)).label("display_filename"),
                GenerationOutput.job_id.label("job_id"),
                GenerationJob.model.label("model"),
                GenerationJob.generation_type.label("generation_type"),
                LibraryAssetMetadata.display_title.label("display_title"),
                func.coalesce(LibraryAssetMetadata.is_favorite, false()).label("is_favorite"),
                LibraryAssetMetadata.project_id.label("project_id"),
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

        if media_type == MediaKind.VIDEO:
            query = query.where(GenerationOutput.content_type.like("video/%"))
        elif media_type == MediaKind.IMAGE:
            query = query.where(~GenerationOutput.content_type.like("video/%"))

        if model is not None:
            query = query.where(GenerationJob.model == model)

        if favorite is True:
            query = query.where(LibraryAssetMetadata.is_favorite.is_(True))
        elif favorite is False:
            query = query.where(func.coalesce(LibraryAssetMetadata.is_favorite, false()).is_(False))

        if project_id is not None:
            query = query.where(LibraryAssetMetadata.project_id == project_id)

        if tag_id is not None:
            query = query.where(
                exists(
                    select(LibraryAssetTag.tag_id).where(
                        LibraryAssetTag.asset_type == LibraryAssetSource.OUTPUT.value,
                        LibraryAssetTag.asset_id == GenerationOutput.id,
                        LibraryAssetTag.tag_id == tag_id,
                        LibraryAssetTag.user_id == user_id,
                        LibraryAssetTag.product_id == product_id,
                    )
                )
            )

        if expiring is True:
            query = query.where(GenerationOutput.expires_at <= expiring_threshold)
        elif expiring is False:
            query = query.where(GenerationOutput.expires_at > expiring_threshold)

        if search_term is not None:
            pattern = f"%{search_term}%"
            query = query.where(
                or_(
                    GenerationJob.prompt.ilike(pattern, escape="\\"),
                    LibraryAssetMetadata.display_title.ilike(pattern, escape="\\"),
                )
            )

        if created_from is not None:
            query = query.where(GenerationOutput.created_at >= created_from)
        if created_to is not None:
            query = query.where(GenerationOutput.created_at <= created_to)

        if cursor_ts is not None and cursor_rank is not None and cursor_id is not None:
            ts_col = (
                GenerationOutput.expires_at
                if sort == LibrarySort.EXPIRING_SOON
                else GenerationOutput.created_at
            )
            ascending = sort in (LibrarySort.OLDEST, LibrarySort.EXPIRING_SOON)
            current = tuple_(ts_col, literal(rank), GenerationOutput.id)
            cursor_tuple = tuple_(literal(cursor_ts), literal(cursor_rank), literal(cursor_id))
            query = query.where(current > cursor_tuple if ascending else current < cursor_tuple)

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

        Mirrors the now-removed GalleryRepository.get_gallery_job byte-for-byte.

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
    # Lineage graph (Phase 3, T8) — immediate descendants only
    # -------------------------------------------------------------------------

    async def list_output_descendants(
        self,
        source: LibraryAssetSource,
        asset_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        limit: int,
    ) -> Sequence[GenerationOutput]:
        """Immediate output descendants: outputs of completed jobs remixing this asset.

        Args:
            source: Which table ``asset_id`` belongs to.
            asset_id: The asset's primary key.
            user_id: Owner scope.
            product_id: Product scope.
            limit: Row cap — caller fetches limit+1 to detect truncation.

        Returns:
            Non-thumbnail GenerationOutput rows, newest first.
        """
        job_filter = (
            GenerationJob.input_image_id == asset_id
            if source == LibraryAssetSource.UPLOAD
            else GenerationJob.source_output_id == asset_id
        )
        result = await self._session.execute(
            select(GenerationOutput)
            .join(GenerationJob, GenerationJob.id == GenerationOutput.job_id)
            .where(
                job_filter,
                GenerationJob.status == JobStatus.COMPLETED,
                GenerationJob.is_deleted.is_(False),
                GenerationOutput.user_id == user_id,
                GenerationOutput.product_id == product_id,
                GenerationOutput.is_thumbnail.is_(False),
            )
            .order_by(GenerationOutput.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def list_frame_descendants(
        self,
        source: LibraryAssetSource,
        asset_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        limit: int,
    ) -> Sequence[UserImage]:
        """Immediate frame descendants: uploads extracted from this asset.

        Args:
            source: Which table ``asset_id`` belongs to.
            asset_id: The asset's primary key.
            user_id: Owner scope.
            product_id: Product scope.
            limit: Row cap — caller fetches limit+1 to detect truncation.

        Returns:
            Non-thumbnail UserImage rows, newest first.
        """
        frame_filter = (
            UserImage.source_upload_id == asset_id
            if source == LibraryAssetSource.UPLOAD
            else UserImage.source_output_id == asset_id
        )
        result = await self._session.execute(
            select(UserImage)
            .where(
                frame_filter,
                UserImage.user_id == user_id,
                UserImage.product_id == product_id,
                UserImage.is_thumbnail.is_(False),
            )
            .order_by(UserImage.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

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
        project_id: OptionalUpdate[UUID | None] = UNSET_UPDATE,
    ) -> LibraryAssetMetadata:
        """Race-safe lazy create-or-update of a library_asset_metadata row.

        Ownership of the underlying asset must be verified by the caller
        BEFORE calling this — this method only enforces the DB-level
        uniqueness of (product_id, user_id, asset_type, asset_id). Likewise,
        ownership of ``project_id`` (if provided) must be verified by the
        caller (P8) — this method does not re-check it.

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
            project_id: New project assignment (``None`` unassigns), or
                ``UNSET_UPDATE`` to leave unchanged (defaults to ``None`` on
                first insert).

        Returns:
            The resulting LibraryAssetMetadata row.
        """
        insert_is_favorite = False if isinstance(is_favorite, _UnsetUpdate) else is_favorite
        insert_display_title = None if isinstance(display_title, _UnsetUpdate) else display_title
        insert_project_id = None if isinstance(project_id, _UnsetUpdate) else project_id

        update_values: dict[str, object] = {"updated_at": text("CURRENT_TIMESTAMP")}
        if not isinstance(is_favorite, _UnsetUpdate):
            update_values["is_favorite"] = is_favorite
        if not isinstance(display_title, _UnsetUpdate):
            update_values["display_title"] = display_title
        if not isinstance(project_id, _UnsetUpdate):
            update_values["project_id"] = project_id

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
                project_id=insert_project_id,
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

    async def bulk_set_favorite(
        self,
        user_id: UUID,
        product_id: str,
        refs: Sequence[AssetRef],
        value: bool,
    ) -> None:
        """Set the favorite flag on a batch of assets in a single statement.

        Race-safe lazy create-or-update, same semantics as
        ``upsert_metadata`` but for many assets in one round trip — a
        single multi-row ``INSERT ... ON CONFLICT DO UPDATE``, never a
        per-ref loop. Ownership of every ref must already be verified by
        the caller (see LibraryService.bulk_apply / P5).

        Uniqueness of ``refs`` (no duplicate (source, asset_id) pairs) must
        also already be guaranteed by the caller — a duplicate here would
        make the multi-row VALUES list affect the same conflict target
        twice, which PostgreSQL rejects with a CardinalityViolationError.
        Single source of truth for dedup is LibraryService.bulk_apply; do
        NOT add a second dedup pass here.
        """
        if not refs:
            return

        rows = [
            {
                "id": new_id(),
                "product_id": product_id,
                "user_id": user_id,
                "asset_type": ref.source.value,
                "asset_id": ref.asset_id,
                "is_favorite": value,
            }
            for ref in refs
        ]
        stmt = (
            pg_insert(LibraryAssetMetadata)
            .values(rows)
            .on_conflict_do_update(
                constraint="uq_library_asset_metadata_asset",
                set_={"is_favorite": value, "updated_at": text("CURRENT_TIMESTAMP")},
            )
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def bulk_set_project(
        self,
        user_id: UUID,
        product_id: str,
        refs: Sequence[AssetRef],
        project_id: UUID | None,
    ) -> None:
        """Assign (or unassign, if ``project_id`` is None) a batch of assets to a project.

        Same single-statement bulk-upsert shape as ``bulk_set_favorite``.
        Existence/ownership of ``project_id`` itself must already be
        verified by the caller (P8) — this method does not re-check it.

        Uniqueness of ``refs`` must also already be guaranteed by the
        caller, same reasoning as ``bulk_set_favorite`` — do NOT add a
        second dedup pass here.
        """
        if not refs:
            return

        rows = [
            {
                "id": new_id(),
                "product_id": product_id,
                "user_id": user_id,
                "asset_type": ref.source.value,
                "asset_id": ref.asset_id,
                "project_id": project_id,
            }
            for ref in refs
        ]
        stmt = (
            pg_insert(LibraryAssetMetadata)
            .values(rows)
            .on_conflict_do_update(
                constraint="uq_library_asset_metadata_asset",
                set_={"project_id": project_id, "updated_at": text("CURRENT_TIMESTAMP")},
            )
        )
        await self._session.execute(stmt)
        await self._session.flush()

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
