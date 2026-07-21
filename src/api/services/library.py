"""Library service — unified read model over uploads + generation outputs.

Business logic for the library endpoints: list/detail/favorite/patch/delete
for individual assets, plus group detail (ported from the now-removed
GalleryService).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import msgspec
import structlog

from src.api.schemas.library import (
    BulkAddTags,
    BulkDelete,
    BulkOperationItemResult,
    BulkOperationResult,
    BulkRemoveTags,
    BulkSetFavorite,
    BulkSetProject,
    LibraryAssetDetail,
    LibraryAssetItem,
    LibraryAssetPatch,
    LibraryDescendants,
    LibraryGroupDetail,
    LibraryGroupLineage,
    LibraryLineage,
    LibraryLineageGraph,
    LibraryOutputItem,
    LibraryTagRef,
    LineageEdge,
    LineageNode,
    LineageRelation,
)
from src.api.schemas.media import ImageVariant, MediaObject, MediaOriginal
from src.api.schemas.pagination import CursorPage, decode_library_cursor, encode_library_cursor
from src.api.services.content_proxy import ContentNotFoundError
from src.api.services.library_capabilities import resolve_library_actions
from src.api.services.media import OUTPUT_PREFIX, UPLOAD_PREFIX
from src.core.enums import (
    GenerationType,
    JobStatus,
    LibraryBadge,
    LibraryGroupSourceType,
    LibrarySort,
    OutputMediaType,
)
from src.core.library_limits import MAX_TAGS_PER_ASSET as _MAX_TAGS_PER_ASSET
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref, parse_asset_ref
from src.core.thumbnails import label_for_max_edge
from src.db.models.storage import GenerationOutput, UserImage
from src.db.repositories.job import JobRepository
from src.db.repositories.library import UNSET_UPDATE, LibraryRepository, _UnsetUpdate
from src.db.repositories.library_project import LibraryProjectRepository
from src.db.repositories.library_tag import LibraryTagRepository, TagRef
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.library import BulkOperation
    from src.api.services.content_proxy import ContentProxyService
    from src.core.product import ProductConfig
    from src.db.models.storage import GenerationJob
    from src.db.repositories.library import LibraryAssetRow, OptionalUpdate

logger = structlog.get_logger(__name__)

# Lineage graph bounds (T8) — ancestor walk depth and per-relation
# descendant cap. Deliberately small constants for an on-demand, bounded
# read; see LibraryService.get_lineage_graph.
_LINEAGE_DEPTH_CAP = 10
_LINEAGE_DESCENDANTS_CAP = 50


class LibraryValidationError(Exception):
    """Raised when a library asset patch fails validation."""


class LibraryProjectNotFoundError(Exception):
    """Raised when a patch/bulk op references a project_id that doesn't exist / isn't owned. → HTTP 404"""

    def __init__(self, project_id: UUID) -> None:
        self.project_id = project_id
        super().__init__(f"Project {project_id} not found")


class LibraryTagNotFoundError(Exception):
    """Raised when a patch/bulk op references tag_id(s) that don't exist / aren't owned. → HTTP 404"""

    def __init__(self, tag_ids: Sequence[UUID]) -> None:
        self.tag_ids = list(tag_ids)
        super().__init__(f"{len(self.tag_ids)} tag id(s) not found: {self.tag_ids}")


class LibraryBulkValidationError(Exception):
    """Raised when one or more asset_refs in a bulk op are malformed/missing/unowned. → HTTP 400

    Per P5, this always covers the whole batch: validation runs for every
    ref before anything executes, so ``invalid_refs`` lists every offender
    at once rather than failing fast on the first one.
    """

    def __init__(self, invalid_refs: Sequence[str]) -> None:
        self.invalid_refs = list(invalid_refs)
        super().__init__(f"{len(self.invalid_refs)} invalid asset_ref(s): {self.invalid_refs}")


class LibraryBulkTagCapError(Exception):
    """Raised when a bulk ``add_tags`` op would push an asset past ``_MAX_TAGS_PER_ASSET``. → HTTP 422

    Distinct from the 400 ``LibraryBulkValidationError`` (malformed/unknown
    refs) and the 404 ``LibraryTagNotFoundError`` (unknown tags) — this is a
    422 (unprocessable entity) since ``app.py`` already establishes that
    status for a well-formed-but-semantically-rejected request (see
    ``ModerationError``), so it's the existing convention rather than a new
    one. The offending refs and the cap are carried through to the error
    envelope's ``detail`` so the FE can point at exactly which assets are
    over the limit.
    """

    def __init__(self, over_cap_refs: Sequence[str], cap: int) -> None:
        self.over_cap_refs = list(over_cap_refs)
        self.cap = cap
        super().__init__(
            f"{len(self.over_cap_refs)} asset(s) would exceed the {cap}-tag cap: "
            f"{self.over_cap_refs}"
        )


class LibraryService:
    """Business logic for library endpoints."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    # List
    # -------------------------------------------------------------------------

    async def list_assets(
        self,
        user_id: UUID,
        product_id: str,
        product_config: ProductConfig,
        *,
        session: AsyncSession,
        limit: int = 30,
        cursor: str | None = None,
        source: LibraryAssetSource | None = None,
        media_type: OutputMediaType | None = None,
        model: str | None = None,
        favorite: bool | None = None,
        project_id: UUID | None = None,
        tag_id: UUID | None = None,
        expiring: bool | None = None,
        query: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        sort: LibrarySort = LibrarySort.NEWEST,
    ) -> CursorPage[LibraryAssetItem]:
        """Return a paginated library grid mixing uploads and outputs.

        Args:
            user_id: Requesting user.
            product_id: Product scope.
            product_config: Resolved product config (for action gating).
            session: Database session.
            limit: Page size.
            cursor: Opaque library cursor from a previous page.
            source: Optional restriction to a single source table.
            media_type: Optional media type filter.
            model: Optional model filter (implies output-only).
            favorite: Optional favorite filter.
            project_id: Optional filter to assets assigned to this project.
            tag_id: Optional filter to assets tagged with this tag.
            expiring: Optional filter to assets expiring within (True) or
                beyond (False) the expiring-soon window.
            query: Optional case-insensitive substring search over
                display_title / original_filename / prompt.
            created_from: Optional lower bound on created_at.
            created_to: Optional upper bound on created_at.
            sort: newest (default), oldest, or expiring_soon.

        Returns:
            CursorPage of LibraryAssetItem.

        Raises:
            ValueError: If ``cursor`` is malformed or was created under a
                different sort.
        """
        repo = LibraryRepository(session)

        decoded_cursor = (
            decode_library_cursor(cursor, expected_sort=sort.value) if cursor is not None else None
        )

        rows = await repo.list_assets(
            user_id,
            product_id,
            limit=limit,
            cursor=decoded_cursor,
            source=source,
            media_type=media_type,
            model=model,
            favorite=favorite,
            project_id=project_id,
            tag_id=tag_id,
            expiring=expiring,
            query=query,
            created_from=created_from,
            created_to=created_to,
            sort=sort,
        )

        has_more = len(rows) > limit
        page_rows = list(rows[:limit])

        output_ids = [r.id for r in page_rows if r.source == LibraryAssetSource.OUTPUT]
        upload_ids = [r.id for r in page_rows if r.source == LibraryAssetSource.UPLOAD]

        output_repo = OutputRepository(session)
        image_repo = UserImageRepository(session)
        output_derivatives = await output_repo.batch_derivatives(output_ids)
        upload_derivatives = await image_repo.batch_derivatives(upload_ids)

        job_ids = list({r.job_id for r in page_rows if r.job_id is not None})
        output_counts = await repo.batch_output_counts(job_ids)

        project_ids = list({r.project_id for r in page_rows if r.project_id is not None})
        project_names = await LibraryProjectRepository(session).batch_names(
            project_ids, user_id=user_id, product_id=product_id
        )

        tag_pairs = [(row.source.value, row.id) for row in page_rows]
        asset_tags = await LibraryTagRepository(session).batch_tags_for_assets(
            tag_pairs, user_id=user_id, product_id=product_id
        )

        items = [
            self._build_item_from_row(
                row,
                derivatives=(
                    output_derivatives.get(row.id, [])
                    if row.source == LibraryAssetSource.OUTPUT
                    else upload_derivatives.get(row.id, [])
                ),
                output_count=(output_counts.get(row.job_id, 0) if row.job_id is not None else None),
                product_config=product_config,
                project_names=project_names,
                asset_tags=asset_tags,
            )
            for row in page_rows
        ]

        next_cursor: str | None = None
        if has_more and page_rows:
            last = page_rows[-1]
            last_ts = last.expires_at if sort == LibrarySort.EXPIRING_SOON else last.created_at
            next_cursor = encode_library_cursor(
                last_ts, last.source.value, last.id, sort=sort.value
            )

        logger.info(
            "library.list",
            user_id=str(user_id),
            product_id=product_id,
            count=len(items),
            has_more=has_more,
        )
        if query is not None:
            # Search terms are user content — never logged verbatim, length only.
            logger.info("library.search", user_id=str(user_id), query_length=len(query))

        return CursorPage(items=items, limit=limit, has_more=has_more, next_cursor=next_cursor)

    def _build_item_from_row(
        self,
        row: LibraryAssetRow,
        *,
        derivatives: Sequence[GenerationOutput | UserImage],
        output_count: int | None,
        product_config: ProductConfig,
        project_names: dict[UUID, str],
        asset_tags: Mapping[tuple[str, UUID], list[TagRef]],
    ) -> LibraryAssetItem:
        media = _build_media_object(
            source=row.source,
            asset_id=row.id,
            width=row.width,
            height=row.height,
            content_type=row.content_type,
            size_bytes=row.size_bytes,
            derivatives=derivatives,
        )
        has_generation_metadata = row.source == LibraryAssetSource.OUTPUT
        actions = resolve_library_actions(
            media_type=media.media_type,
            source=row.source,
            has_generation_metadata=has_generation_metadata,
            product_config=product_config,
        )
        return LibraryAssetItem(
            asset_ref=format_asset_ref(row.source, row.id),
            source=row.source,
            media=media,
            created_at=row.created_at,
            expires_at=row.expires_at,
            display_title=row.display_title,
            original_filename=row.original_filename,
            is_favorite=row.is_favorite,
            duration_ms=row.duration_ms,
            job_id=row.job_id,
            output_count=output_count,
            model=row.model,
            generation_type=GenerationType(row.generation_type)
            if row.generation_type is not None
            else None,
            available_actions=actions,
            project_id=row.project_id,
            project_name=project_names.get(row.project_id) if row.project_id is not None else None,
            tags=tuple(
                LibraryTagRef(id=t.id, name=t.name)
                for t in asset_tags.get((row.source.value, row.id), [])
            ),
        )

    # -------------------------------------------------------------------------
    # Asset detail
    # -------------------------------------------------------------------------

    async def get_asset_detail(
        self,
        asset_ref: str,
        user_id: UUID,
        product_id: str,
        product_config: ProductConfig,
        *,
        session: AsyncSession,
    ) -> LibraryAssetDetail | None:
        """Return full detail for a single library asset, or None if missing/not owned."""
        try:
            ref = parse_asset_ref(asset_ref)
        except ValueError:
            return None

        if ref.source == LibraryAssetSource.UPLOAD:
            return await self._get_upload_detail(ref, user_id, product_id, product_config, session)
        return await self._get_output_detail(ref, user_id, product_id, product_config, session)

    async def _get_upload_detail(
        self,
        ref: AssetRef,
        user_id: UUID,
        product_id: str,
        product_config: ProductConfig,
        session: AsyncSession,
    ) -> LibraryAssetDetail | None:
        image_repo = UserImageRepository(session)
        image = await image_repo.get(ref.asset_id, user_id=user_id)
        if image is None or image.product_id != product_id or image.is_thumbnail:
            return None

        repo = LibraryRepository(session)
        metadata = await repo.get_metadata(user_id, product_id, LibraryAssetSource.UPLOAD, image.id)
        project_id = metadata.project_id if metadata is not None else None
        project_name = await self._resolve_project_name(project_id, user_id, product_id, session)
        tags = await self._resolve_asset_tags(
            LibraryAssetSource.UPLOAD, image.id, user_id, product_id, session
        )
        derivatives = await image_repo.list_derivatives(image.id)

        media = _build_media_object(
            source=LibraryAssetSource.UPLOAD,
            asset_id=image.id,
            width=image.width,
            height=image.height,
            content_type=image.content_type,
            size_bytes=image.size_bytes,
            derivatives=derivatives,
        )

        lineage: LibraryLineage | None = None
        if image.source_output_id is not None:
            source_output = await OutputRepository(session).get(
                image.source_output_id, user_id=user_id
            )
            source_ref = (
                format_asset_ref(LibraryAssetSource.OUTPUT, source_output.id)
                if source_output is not None
                else None
            )
            lineage = LibraryLineage(
                source_asset_ref=source_ref,
                source_job_id=source_output.job_id if source_output is not None else None,
                source_timestamp_ms=image.source_timestamp_ms,
            )
        elif image.source_upload_id is not None:
            lineage = LibraryLineage(
                source_asset_ref=format_asset_ref(
                    LibraryAssetSource.UPLOAD, image.source_upload_id
                ),
                source_job_id=None,
                source_timestamp_ms=image.source_timestamp_ms,
            )

        job_count, frame_count = await repo.count_descendants(LibraryAssetSource.UPLOAD, image.id)

        actions = resolve_library_actions(
            media_type=media.media_type,
            source=LibraryAssetSource.UPLOAD,
            has_generation_metadata=False,
            product_config=product_config,
        )

        return LibraryAssetDetail(
            asset_ref=format_asset_ref(LibraryAssetSource.UPLOAD, image.id),
            source=LibraryAssetSource.UPLOAD,
            media=media,
            created_at=image.created_at,
            expires_at=image.expires_at,
            display_title=metadata.display_title if metadata is not None else None,
            original_filename=image.original_filename,
            is_favorite=metadata.is_favorite if metadata is not None else False,
            duration_ms=image.duration_ms,
            job_id=None,
            output_count=None,
            model=None,
            generation_type=None,
            available_actions=actions,
            project_id=project_id,
            project_name=project_name,
            tags=tags,
            prompt=None,
            negative_prompt=None,
            provider=None,
            aspect_ratio=None,
            token_cost=None,
            completed_at=None,
            lineage=lineage,
            descendants=LibraryDescendants(job_count=job_count, frame_count=frame_count),
        )

    async def _get_output_detail(
        self,
        ref: AssetRef,
        user_id: UUID,
        product_id: str,
        product_config: ProductConfig,
        session: AsyncSession,
    ) -> LibraryAssetDetail | None:
        output_repo = OutputRepository(session)
        output = await output_repo.get(ref.asset_id, user_id=user_id)
        if output is None or output.product_id != product_id or output.is_thumbnail:
            return None

        job = await JobRepository(session).get(output.job_id, user_id=user_id)
        if job is None or job.product_id != product_id:
            return None

        repo = LibraryRepository(session)
        metadata = await repo.get_metadata(
            user_id, product_id, LibraryAssetSource.OUTPUT, output.id
        )
        project_id = metadata.project_id if metadata is not None else None
        project_name = await self._resolve_project_name(project_id, user_id, product_id, session)
        tags = await self._resolve_asset_tags(
            LibraryAssetSource.OUTPUT, output.id, user_id, product_id, session
        )
        derivatives = await output_repo.list_derivatives(output.id)

        media = _build_media_object(
            source=LibraryAssetSource.OUTPUT,
            asset_id=output.id,
            width=output.width,
            height=output.height,
            content_type=output.content_type,
            size_bytes=output.size_bytes,
            derivatives=derivatives,
        )

        job_count, frame_count = await repo.count_descendants(LibraryAssetSource.OUTPUT, output.id)
        output_counts = await repo.batch_output_counts([job.id])

        actions = resolve_library_actions(
            media_type=media.media_type,
            source=LibraryAssetSource.OUTPUT,
            has_generation_metadata=True,
            product_config=product_config,
        )

        return LibraryAssetDetail(
            asset_ref=format_asset_ref(LibraryAssetSource.OUTPUT, output.id),
            source=LibraryAssetSource.OUTPUT,
            media=media,
            created_at=output.created_at,
            expires_at=output.expires_at,
            display_title=metadata.display_title if metadata is not None else None,
            original_filename=None,
            is_favorite=metadata.is_favorite if metadata is not None else False,
            duration_ms=None,
            job_id=job.id,
            output_count=output_counts.get(job.id, 0),
            model=job.model,
            generation_type=GenerationType(job.generation_type),
            available_actions=actions,
            project_id=project_id,
            project_name=project_name,
            tags=tags,
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            provider=job.provider,
            aspect_ratio=job.aspect_ratio,
            token_cost=job.token_cost,
            completed_at=job.completed_at,
            lineage=None,
            descendants=LibraryDescendants(job_count=job_count, frame_count=frame_count),
        )

    @staticmethod
    async def _resolve_project_name(
        project_id: UUID | None,
        user_id: UUID,
        product_id: str,
        session: AsyncSession,
    ) -> str | None:
        """Resolve a single project's name, if assigned. None short-circuits with no query."""
        if project_id is None:
            return None
        names = await LibraryProjectRepository(session).batch_names(
            [project_id], user_id=user_id, product_id=product_id
        )
        return names.get(project_id)

    @staticmethod
    async def _resolve_asset_tags(
        source: LibraryAssetSource,
        asset_id: UUID,
        user_id: UUID,
        product_id: str,
        session: AsyncSession,
    ) -> tuple[LibraryTagRef, ...]:
        """Resolve a single asset's tags for detail views — one query, not per-row."""
        tag_map = await LibraryTagRepository(session).batch_tags_for_assets(
            [(source.value, asset_id)], user_id=user_id, product_id=product_id
        )
        return tuple(
            LibraryTagRef(id=t.id, name=t.name) for t in tag_map.get((source.value, asset_id), [])
        )

    # -------------------------------------------------------------------------
    # Mutations
    # -------------------------------------------------------------------------

    async def set_favorite(
        self,
        asset_ref: str,
        value: bool,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> bool:
        """Set (or clear) the favorite flag on an asset. Idempotent.

        Returns:
            True if the asset was found and owned; False otherwise.
        """
        try:
            ref = parse_asset_ref(asset_ref)
        except ValueError:
            return False

        if not await self._asset_exists(ref, user_id, product_id, session):
            return False

        repo = LibraryRepository(session)
        await repo.upsert_metadata(user_id, product_id, ref.source, ref.asset_id, is_favorite=value)
        logger.info(
            "library.favorite_set",
            asset_ref=asset_ref,
            user_id=str(user_id),
            value=value,
        )
        return True

    async def patch_asset(
        self,
        asset_ref: str,
        patch: LibraryAssetPatch,
        user_id: UUID,
        product_id: str,
        product_config: ProductConfig,
        *,
        session: AsyncSession,
    ) -> LibraryAssetDetail | None:
        """Apply a partial update to an asset's mutable metadata.

        Supports ``display_title`` (tri-state: absent = no-op, ``null`` =
        clear, string = set — validated to <=255 chars, stripped, empty
        string normalized to ``None``), ``project_id`` (tri-state: absent =
        no-op, ``null`` = unassign, UUID = assign — the referenced project
        must be owned by ``user_id`` under ``product_id``, P8), and
        ``tag_ids`` (tri-state: absent = no-op, list = replace-set — every
        id deduped order-preservingly, capped at 20, and must be owned by
        ``user_id`` under ``product_id``; no ``null`` arm, clearing is
        ``[]``, T4).

        Returns:
            Updated LibraryAssetDetail, or None if the asset doesn't exist /
            isn't owned by ``user_id``.

        Raises:
            LibraryValidationError: If ``display_title`` fails validation or
                ``tag_ids`` has more than 20 entries.
            LibraryProjectNotFoundError: If ``project_id`` doesn't exist /
                isn't owned by ``user_id``.
            LibraryTagNotFoundError: If any ``tag_ids`` entry doesn't exist /
                isn't owned by ``user_id``.
        """
        try:
            ref = parse_asset_ref(asset_ref)
        except ValueError:
            return None

        if not await self._asset_exists(ref, user_id, product_id, session):
            return None

        # Validate-then-mutate: every tri-state field is validated up front
        # (may raise before any DB write) so e.g. a project_id 404 never
        # leaves a display_title change applied — see M2. display_title and
        # project_id are then combined into a single upsert_metadata call;
        # tag_ids applies via a separate set_asset_tags call against a
        # different table, same transaction, same log event.
        display_title_update: OptionalUpdate[str | None] = UNSET_UPDATE
        if patch.display_title is not msgspec.UNSET:
            display_title_update = self._normalize_display_title(patch.display_title)

        project_id_update: OptionalUpdate[UUID | None] = UNSET_UPDATE
        if patch.project_id is not msgspec.UNSET:
            if patch.project_id is not None:
                project = await LibraryProjectRepository(session).get(
                    patch.project_id, user_id=user_id, product_id=product_id
                )
                if project is None:
                    raise LibraryProjectNotFoundError(patch.project_id)
            project_id_update = patch.project_id

        tag_ids_update: OptionalUpdate[list[UUID]] = UNSET_UPDATE
        if patch.tag_ids is not msgspec.UNSET:
            deduped_tag_ids = list(dict.fromkeys(patch.tag_ids))
            if len(deduped_tag_ids) > _MAX_TAGS_PER_ASSET:
                raise LibraryValidationError(
                    f"tag_ids must contain at most {_MAX_TAGS_PER_ASSET} tags"
                )
            if deduped_tag_ids:
                owned_tags = await LibraryTagRepository(session).get_many(
                    deduped_tag_ids, user_id=user_id, product_id=product_id
                )
                if missing_tags := [tid for tid in deduped_tag_ids if tid not in owned_tags]:
                    raise LibraryTagNotFoundError(missing_tags)
            tag_ids_update = deduped_tag_ids

        changed_fields: list[str] = []
        if not isinstance(display_title_update, _UnsetUpdate):
            changed_fields.append("display_title")
        if not isinstance(project_id_update, _UnsetUpdate):
            changed_fields.append("project_id")
        if not isinstance(tag_ids_update, _UnsetUpdate):
            changed_fields.append("tag_ids")

        if changed_fields:
            if not isinstance(display_title_update, _UnsetUpdate) or not isinstance(
                project_id_update, _UnsetUpdate
            ):
                repo = LibraryRepository(session)
                await repo.upsert_metadata(
                    user_id,
                    product_id,
                    ref.source,
                    ref.asset_id,
                    display_title=display_title_update,
                    project_id=project_id_update,
                )
            if not isinstance(tag_ids_update, _UnsetUpdate):
                await LibraryTagRepository(session).set_asset_tags(
                    ref.source.value,
                    ref.asset_id,
                    tag_ids_update,
                    user_id=user_id,
                    product_id=product_id,
                )
            logger.info(
                "library.metadata_patched",
                asset_ref=asset_ref,
                user_id=str(user_id),
                fields=changed_fields,
            )

        return await self.get_asset_detail(
            asset_ref, user_id, product_id, product_config, session=session
        )

    @staticmethod
    def _normalize_display_title(raw: str | None) -> str | None:
        stripped = raw.strip() if raw is not None else None
        if stripped == "":
            stripped = None
        if stripped is not None and len(stripped) > 255:
            raise LibraryValidationError("display_title must be at most 255 characters")
        return stripped

    async def delete_asset(
        self,
        asset_ref: str,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
        content_proxy: ContentProxyService,
    ) -> bool:
        """Purge the asset's metadata and tag rows, then delete it via ContentProxyService.

        A typed pre-check (fetch by the ref's declared source) runs before
        delegating to ``ContentProxyService.delete_content`` — a malformed
        claim (ref says upload but the id belongs to an output) is treated
        as not-found rather than falling through to the other table.

        The metadata and tag purges are flushed BEFORE ``delete_content`` so
        they ride in the same commit as the content-row delete —
        ``delete_content`` commits internally before its best-effort R2
        cleanup, and a purge issued after that commit would run in a second
        transaction whose failure could leak the metadata/tag rows forever
        (mirrors ``ContentRetentionService._purge_metadata``). If
        ``delete_content`` then raises ``ContentNotFoundError`` (race: asset
        vanished between the pre-check and delegation), this method rolls
        the session back itself before returning — ``delete_asset`` swallows
        the exception to preserve its bool-return contract (bulk delete
        needs a per-ref result, not an abort), so the DI session provider's
        own rollback-on-exception never triggers here; without an explicit
        rollback the flushed purges would ride along on the next unrelated
        commit. The metadata/tag rows survive unless the concurrent deleter
        already purged them too.

        Returns:
            True if deleted; False if not found / not owned.
        """
        try:
            ref = parse_asset_ref(asset_ref)
        except ValueError:
            return False

        if not await self._asset_exists(ref, user_id, product_id, session):
            return False

        repo = LibraryRepository(session)
        await repo.delete_metadata_for_assets([(ref.source.value, ref.asset_id)])
        await LibraryTagRepository(session).delete_tags_for_assets(
            [(ref.source.value, ref.asset_id)]
        )

        try:
            await content_proxy.delete_content(
                ref.asset_id,
                user_id=user_id,
                product_id=product_id,
                session=session,
            )
        except ContentNotFoundError:
            await session.rollback()
            return False

        logger.info(
            "library.asset_deleted",
            asset_ref=asset_ref,
            user_id=str(user_id),
            source=ref.source.value,
        )
        return True

    async def _asset_exists(
        self,
        ref: AssetRef,
        user_id: UUID,
        product_id: str,
        session: AsyncSession,
    ) -> bool:
        if ref.source == LibraryAssetSource.UPLOAD:
            image = await UserImageRepository(session).get(ref.asset_id, user_id=user_id)
            return image is not None and image.product_id == product_id and not image.is_thumbnail
        output = await OutputRepository(session).get(ref.asset_id, user_id=user_id)
        return output is not None and output.product_id == product_id and not output.is_thumbnail

    # -------------------------------------------------------------------------
    # Bulk operations (P4/P5)
    # -------------------------------------------------------------------------

    async def bulk_apply(
        self,
        op: BulkOperation,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
        content_proxy: ContentProxyService,
    ) -> BulkOperationResult:
        """Execute a bulk operation across up to 100 asset_refs.

        Per P5, every ref is parsed and ownership-validated BEFORE anything
        executes — a single bad ref fails the whole request with
        ``LibraryBulkValidationError`` (→ 400 listing every offender), never
        a silent partial skip. Once validation passes, favorite/project
        assignment run as one bulk statement each (single transaction);
        delete reuses ``delete_asset`` per ref (O(n) R2 calls, bounded by
        the 100-ref cap) since it already handles the DB-first/R2-best-effort
        ordering and per-asset commit.

        Idempotency: favorite/project bulk ops are naturally idempotent —
        replaying the same request converges to the same end state, so no
        Idempotency-Key handling is wired here. Bulk delete is idempotent up
        to the point an asset is actually gone: retrying after a partial
        client-side failure re-validates and, for any ref already deleted,
        surfaces it as a 400 in ``invalid_refs`` (not silently ignored) —
        an explicit signal rather than the ambiguous both-outcomes-look-the-same
        behavior a shared create-once IdempotencyService is built around.

        Duplicate refs (by parsed identity, so mixed-case UUID duplicates
        collapse too) are silently collapsed to their first occurrence right
        after parsing — this is not a P5 "skip" since the operation on that
        asset still executes exactly once; it just preserves the advertised
        idempotent semantics instead of hitting a multi-row ``ON CONFLICT``
        cardinality violation. ``results`` (and the ``library.bulk_applied``
        count) reflect unique assets only.

        Raises:
            LibraryBulkValidationError: If any asset_ref is malformed,
                missing, not owned, or belongs to a different product.
            LibraryProjectNotFoundError: If a ``set_project`` op's
                ``project_id`` doesn't exist / isn't owned by ``user_id``.
            LibraryTagNotFoundError: If an ``add_tags``/``remove_tags`` op's
                ``tag_ids`` contains an id that doesn't exist / isn't owned
                by ``user_id``.
            LibraryBulkTagCapError: If an ``add_tags`` op would push any
                asset's tag count past ``_MAX_TAGS_PER_ASSET`` (counting the
                dedup union of its existing tags and the new ones — an
                idempotent re-add of an already-assigned tag never consumes
                a slot).
        """
        parsed: list[AssetRef] = []
        malformed: list[str] = []
        for raw in op.asset_refs:
            try:
                parsed.append(parse_asset_ref(raw))
            except ValueError:
                malformed.append(raw)
        if malformed:
            raise LibraryBulkValidationError(malformed)

        seen: set[tuple[LibraryAssetSource, UUID]] = set()
        deduped: list[AssetRef] = []
        for ref in parsed:
            key = (ref.source, ref.asset_id)
            if key not in seen:
                seen.add(key)
                deduped.append(ref)
        parsed = deduped

        invalid = await self._validate_refs(parsed, user_id, product_id, session)
        if invalid:
            raise LibraryBulkValidationError(
                [format_asset_ref(r.source, r.asset_id) for r in invalid]
            )

        if isinstance(op, BulkSetProject) and op.project_id is not None:
            project = await LibraryProjectRepository(session).get(
                op.project_id, user_id=user_id, product_id=product_id
            )
            if project is None:
                raise LibraryProjectNotFoundError(op.project_id)

        tag_ids_deduped: list[UUID] = []
        if isinstance(op, BulkAddTags | BulkRemoveTags):
            tag_ids_deduped = list(dict.fromkeys(op.tag_ids))
            owned_tags = await LibraryTagRepository(session).get_many(
                tag_ids_deduped, user_id=user_id, product_id=product_id
            )
            if missing_tags := [tid for tid in tag_ids_deduped if tid not in owned_tags]:
                raise LibraryTagNotFoundError(missing_tags)

        if isinstance(op, BulkAddTags):
            # Cap enforcement is validate-then-mutate (M2): computed here,
            # before any op branch executes, so a cap violation on one asset
            # never leaves the batch partially tagged. The count is the
            # dedup UNION of existing + new ids — a re-add of an
            # already-assigned tag doesn't consume a slot, matching
            # bulk_add_tags' ON CONFLICT DO NOTHING idempotency.
            pairs = [(r.source.value, r.asset_id) for r in parsed]
            existing = await LibraryTagRepository(session).batch_tags_for_assets(
                pairs, user_id=user_id, product_id=product_id
            )
            new_ids = set(tag_ids_deduped)
            over_cap = [
                format_asset_ref(r.source, r.asset_id)
                for r in parsed
                if len({t.id for t in existing.get((r.source.value, r.asset_id), [])} | new_ids)
                > _MAX_TAGS_PER_ASSET
            ]
            if over_cap:
                raise LibraryBulkTagCapError(over_cap, _MAX_TAGS_PER_ASSET)

        if isinstance(op, BulkSetFavorite):
            op_name = "set_favorite"
            await LibraryRepository(session).bulk_set_favorite(
                user_id, product_id, parsed, op.value
            )
            results = [
                BulkOperationItemResult(
                    asset_ref=format_asset_ref(r.source, r.asset_id), success=True
                )
                for r in parsed
            ]
        elif isinstance(op, BulkSetProject):
            op_name = "set_project"
            await LibraryRepository(session).bulk_set_project(
                user_id, product_id, parsed, op.project_id
            )
            results = [
                BulkOperationItemResult(
                    asset_ref=format_asset_ref(r.source, r.asset_id), success=True
                )
                for r in parsed
            ]
        elif isinstance(op, BulkAddTags):
            op_name = "add_tags"
            await LibraryTagRepository(session).bulk_add_tags(
                parsed, tag_ids_deduped, user_id=user_id, product_id=product_id
            )
            results = [
                BulkOperationItemResult(
                    asset_ref=format_asset_ref(r.source, r.asset_id), success=True
                )
                for r in parsed
            ]
        elif isinstance(op, BulkRemoveTags):
            op_name = "remove_tags"
            await LibraryTagRepository(session).bulk_remove_tags(
                parsed, tag_ids_deduped, user_id=user_id, product_id=product_id
            )
            results = [
                BulkOperationItemResult(
                    asset_ref=format_asset_ref(r.source, r.asset_id), success=True
                )
                for r in parsed
            ]
        else:
            assert isinstance(op, BulkDelete)  # noqa: S101 — exhaustiveness over the tagged union
            op_name = "delete"
            results = []
            for r in parsed:
                ref_str = format_asset_ref(r.source, r.asset_id)
                deleted = await self.delete_asset(
                    ref_str, user_id, product_id, session=session, content_proxy=content_proxy
                )
                results.append(BulkOperationItemResult(asset_ref=ref_str, success=deleted))

        succeeded = sum(bool(r.success) for r in results)
        logger.info(
            "library.bulk_applied",
            op=op_name,
            user_id=str(user_id),
            count=len(parsed),
            succeeded=succeeded,
        )
        return BulkOperationResult(
            op=op_name,
            results=results,
            succeeded=succeeded,
            failed=len(results) - succeeded,
        )

    async def _validate_refs(
        self,
        refs: Sequence[AssetRef],
        user_id: UUID,
        product_id: str,
        session: AsyncSession,
    ) -> list[AssetRef]:
        """Return the subset of refs that are missing / not owned / wrong product / thumbnail.

        Batched by source (at most 2 queries total, regardless of how many
        refs are in the batch) rather than one existence check per ref.
        """
        upload_ids = [r.asset_id for r in refs if r.source == LibraryAssetSource.UPLOAD]
        output_ids = [r.asset_id for r in refs if r.source == LibraryAssetSource.OUTPUT]

        uploads = (
            await UserImageRepository(session).get_many(upload_ids, user_id=user_id)
            if upload_ids
            else {}
        )
        outputs = (
            await OutputRepository(session).get_many(output_ids, user_id=user_id)
            if output_ids
            else {}
        )

        invalid: list[AssetRef] = []
        for ref in refs:
            if ref.source == LibraryAssetSource.UPLOAD:
                image = uploads.get(ref.asset_id)
                if image is None or image.product_id != product_id or image.is_thumbnail:
                    invalid.append(ref)
            else:
                output = outputs.get(ref.asset_id)
                if output is None or output.product_id != product_id or output.is_thumbnail:
                    invalid.append(ref)
        return invalid

    # -------------------------------------------------------------------------
    # Group detail — ported from the now-removed GalleryService.get_gallery_detail.
    # -------------------------------------------------------------------------

    async def get_group_detail(
        self,
        job_id: UUID,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> LibraryGroupDetail | None:
        """Return full detail view for a generation group (job)."""
        repo = LibraryRepository(session)
        job = await repo.get_group_job(job_id, user_id, product_id)
        if job is None:
            return None

        gt = GenerationType(job.generation_type)

        full_outputs = [o for o in job.outputs if not o.is_thumbnail]
        derivatives_map: dict[UUID, list[GenerationOutput]] = defaultdict(list)
        for o in job.outputs:
            if o.is_thumbnail and o.parent_output_id is not None:
                derivatives_map[o.parent_output_id].append(o)

        output_items = [
            self._build_group_output_item(o, list(derivatives_map.get(o.id, [])))
            for o in sorted(full_outputs, key=lambda o: o.output_index)
        ]

        input_media: MediaObject | None = None
        if job.source_output is not None:
            input_media = _build_media_object(
                source=LibraryAssetSource.OUTPUT,
                asset_id=job.source_output.id,
                width=job.source_output.width,
                height=job.source_output.height,
                content_type=job.source_output.content_type,
                size_bytes=job.source_output.size_bytes,
                derivatives=list(job.source_output.derivatives),
            )
        elif job.input_image is not None:
            input_media = _build_media_object(
                source=LibraryAssetSource.UPLOAD,
                asset_id=job.input_image.id,
                width=job.input_image.width,
                height=job.input_image.height,
                content_type=job.input_image.content_type,
                size_bytes=job.input_image.size_bytes,
                derivatives=list(job.input_image.derivatives),
            )

        return LibraryGroupDetail(
            job_id=job.id,
            badge=self._resolve_group_badge(gt),
            input_media=input_media,
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            outputs=output_items,
            media_type=OutputMediaType.VIDEO if gt.is_video else OutputMediaType.IMAGE,
            model=job.model,
            provider=job.provider,
            generation_type=gt,
            aspect_ratio=job.aspect_ratio,
            token_cost=job.token_cost,
            created_at=job.created_at,
            completed_at=job.completed_at,
            lineage=self._build_group_lineage(job),
        )

    @staticmethod
    def _resolve_group_badge(gt: GenerationType) -> LibraryBadge:
        if gt.requires_image_input or gt.requires_video_input:
            return LibraryBadge.IMAGE
        return LibraryBadge.PROMPT

    @staticmethod
    def _build_group_lineage(job: GenerationJob) -> LibraryGroupLineage | None:
        if job.source_job_id is not None:
            source_job = job.source_job
            return LibraryGroupLineage(
                source_type=LibraryGroupSourceType.OUTPUT,
                source_job_id=job.source_job_id,
                source_job_name=source_job.name if source_job is not None else None,
                source_output_id=job.source_output_id,
            )
        if job.input_image_id is not None:
            return LibraryGroupLineage(
                source_type=LibraryGroupSourceType.UPLOAD,
                source_upload_id=job.input_image_id,
            )
        return None

    # -------------------------------------------------------------------------
    # Lineage graph (Phase 3, T8) — GET /v1/library/assets/{asset_ref}/lineage
    # -------------------------------------------------------------------------

    async def get_lineage_graph(
        self,
        asset_ref: str,
        user_id: UUID,
        product_id: str,
        *,
        session: AsyncSession,
    ) -> LibraryLineageGraph | None:
        """Return a bounded ancestor/descendant lineage graph for a single asset.

        Total queries: <= depth-cap single-row lookups for the ancestor walk
        (one job lookup + one parent lookup per output-type hop, one lookup
        per upload-type hop) + 2 descendant queries + 2 count queries +
        <=2 derivative batch calls + 1 job batch call. Acceptable for an
        on-demand endpoint — this is not the list hot path.

        Returns:
            LibraryLineageGraph, or None if the asset doesn't exist / isn't
            owned by ``user_id`` (identical 404 semantics to asset detail).
        """
        try:
            ref = parse_asset_ref(asset_ref)
        except ValueError:
            return None

        focus_row = await self._fetch_lineage_row(ref, user_id, product_id, session)
        if focus_row is None:
            return None

        ancestor_edges, ancestors_truncated = await self._walk_ancestors(
            focus_row, user_id, product_id, session
        )

        library_repo = LibraryRepository(session)

        raw_outputs = await library_repo.list_output_descendants(
            ref.source,
            ref.asset_id,
            user_id=user_id,
            product_id=product_id,
            limit=_LINEAGE_DESCENDANTS_CAP + 1,
        )
        outputs_truncated = len(raw_outputs) > _LINEAGE_DESCENDANTS_CAP
        descendant_outputs = list(raw_outputs[:_LINEAGE_DESCENDANTS_CAP])

        raw_frames = await library_repo.list_frame_descendants(
            ref.source,
            ref.asset_id,
            user_id=user_id,
            product_id=product_id,
            limit=_LINEAGE_DESCENDANTS_CAP + 1,
        )
        frames_truncated = len(raw_frames) > _LINEAGE_DESCENDANTS_CAP
        descendant_frames = list(raw_frames[:_LINEAGE_DESCENDANTS_CAP])

        job_count, frame_count = await library_repo.count_descendants(ref.source, ref.asset_id)

        # Batch-hydrate media + job info for every node in the graph — a
        # fixed handful of batch calls total, never per-node (T8).
        all_output_rows: list[GenerationOutput] = list(descendant_outputs)
        all_upload_rows: list[UserImage] = list(descendant_frames)
        if isinstance(focus_row, GenerationOutput):
            all_output_rows.append(focus_row)
        else:
            all_upload_rows.append(focus_row)
        for _relation, _source, row, _ts in ancestor_edges:
            if isinstance(row, GenerationOutput):
                all_output_rows.append(row)
            else:
                all_upload_rows.append(row)

        output_repo = OutputRepository(session)
        image_repo = UserImageRepository(session)
        output_derivatives = await output_repo.batch_derivatives([o.id for o in all_output_rows])
        upload_derivatives = await image_repo.batch_derivatives([u.id for u in all_upload_rows])

        job_ids = list({o.job_id for o in all_output_rows})
        jobs = await JobRepository(session).get_many(job_ids, user_id=user_id)

        def build_node(
            source: LibraryAssetSource, row: GenerationOutput | UserImage
        ) -> LineageNode:
            derivatives = (
                output_derivatives.get(row.id, [])
                if isinstance(row, GenerationOutput)
                else upload_derivatives.get(row.id, [])
            )
            job = jobs.get(row.job_id) if isinstance(row, GenerationOutput) else None
            return LineageNode(
                asset_ref=format_asset_ref(source, row.id),
                source=source,
                media=_build_media_object(
                    source=source,
                    asset_id=row.id,
                    width=row.width,
                    height=row.height,
                    content_type=row.content_type,
                    size_bytes=row.size_bytes,
                    derivatives=derivatives,
                ),
                created_at=row.created_at,
                model=job.model if job is not None else None,
                generation_type=(GenerationType(job.generation_type) if job is not None else None),
            )

        ancestors = tuple(
            LineageEdge(relation=relation, node=build_node(source, row), source_timestamp_ms=ts)
            for relation, source, row, ts in ancestor_edges
        )

        descendant_output_relation = (
            LineageRelation.GENERATED_FROM_UPLOAD
            if ref.source == LibraryAssetSource.UPLOAD
            else LineageRelation.GENERATED_FROM_OUTPUT
        )
        descendant_frame_relation = (
            LineageRelation.FRAME_OF_UPLOAD
            if ref.source == LibraryAssetSource.UPLOAD
            else LineageRelation.FRAME_OF_OUTPUT
        )
        descendants = tuple(
            LineageEdge(
                relation=descendant_output_relation,
                node=build_node(LibraryAssetSource.OUTPUT, o),
                source_timestamp_ms=None,
            )
            for o in descendant_outputs
        ) + tuple(
            LineageEdge(
                relation=descendant_frame_relation,
                node=build_node(LibraryAssetSource.UPLOAD, f),
                source_timestamp_ms=f.source_timestamp_ms,
            )
            for f in descendant_frames
        )

        return LibraryLineageGraph(
            focus=build_node(ref.source, focus_row),
            ancestors=ancestors,
            descendants=descendants,
            descendant_totals=LibraryDescendants(job_count=job_count, frame_count=frame_count),
            ancestors_truncated=ancestors_truncated,
            descendants_truncated=outputs_truncated or frames_truncated,
        )

    @staticmethod
    async def _lineage_output_visible(
        output: GenerationOutput,
        user_id: UUID,
        product_id: str,
        job_repo: JobRepository,
    ) -> GenerationJob | None:
        """Resolve ``output``'s owning job iff it's visible; else None.

        Visible means: present, same-product, completed, and not
        soft-deleted — the exact predicate the list/detail UNION applies
        (``GenerationJob.status == COMPLETED AND is_deleted == False``, see
        ``LibraryRepository._build_output_branch``). Lineage must not
        surface an output the list path would hide, whether as the focus
        node or as an ancestor/descendant.

        Returns the job itself (not just a bool) so callers already walking
        the ancestor chain can reuse it for parent-column resolution
        instead of fetching it again on the next step.
        """
        job = await job_repo.get(output.job_id, user_id=user_id)
        if (
            job is None
            or job.product_id != product_id
            or job.status != JobStatus.COMPLETED
            or job.is_deleted
        ):
            return None
        return job

    @staticmethod
    async def _fetch_lineage_row(
        ref: AssetRef,
        user_id: UUID,
        product_id: str,
        session: AsyncSession,
    ) -> GenerationOutput | UserImage | None:
        """Typed, owner+product-scoped fetch of the lineage focus asset.

        For an OUTPUT ref, the owning job must also be visible (completed,
        not soft-deleted) — an invisible focus returns None here, which the
        caller turns into an identical 404 to a missing asset (no leak of
        *why*). The UPLOAD branch is unchanged: uploads have no owning job.
        """
        if ref.source == LibraryAssetSource.UPLOAD:
            image = await UserImageRepository(session).get(ref.asset_id, user_id=user_id)
            if image is None or image.product_id != product_id or image.is_thumbnail:
                return None
            return image
        output = await OutputRepository(session).get(ref.asset_id, user_id=user_id)
        if output is None or output.product_id != product_id or output.is_thumbnail:
            return None
        job_repo = JobRepository(session)
        if (
            await LibraryService._lineage_output_visible(output, user_id, product_id, job_repo)
            is None
        ):
            return None
        return output

    @staticmethod
    async def _walk_ancestors(
        focus_row: GenerationOutput | UserImage,
        user_id: UUID,
        product_id: str,
        session: AsyncSession,
    ) -> tuple[
        list[tuple[LineageRelation, LibraryAssetSource, GenerationOutput | UserImage, int | None]],
        bool,
    ]:
        """Iteratively resolve one parent per step, nearest-first (T8).

        Every fetch is owner+product scoped. Stops on: no parent column
        set, parent row missing (SET NULL raced), a resolved output ancestor
        whose owning job isn't visible (not completed / soft-deleted — a
        natural graph boundary, not truncation, matching the list/detail
        hide rule), a cycle (impossible by construction today, cheap
        insurance forever), or the depth cap — only the depth cap sets
        ``truncated=True``. Deliberately an iterative per-table walk, not a
        recursive CTE (T8): two polymorphic tables plus jobs make a CTE
        unreadable and unbounded-by-default.

        Each output node's owning job is fetched exactly once, right when
        the node is resolved as a parent, and carried forward as
        ``current_job`` for the next step's parent-column resolution — never
        re-fetched, so the walk's query count stays linear in depth.

        Returns:
            ``(edges, truncated)`` where edges are
            ``(relation, source, row, source_timestamp_ms)`` in
            nearest-first order.
        """
        output_repo = OutputRepository(session)
        image_repo = UserImageRepository(session)
        job_repo = JobRepository(session)

        edges: list[
            tuple[LineageRelation, LibraryAssetSource, GenerationOutput | UserImage, int | None]
        ] = []
        focus_source = (
            LibraryAssetSource.OUTPUT
            if isinstance(focus_row, GenerationOutput)
            else LibraryAssetSource.UPLOAD
        )
        visited: set[tuple[LibraryAssetSource, UUID]] = {(focus_source, focus_row.id)}
        current_row: GenerationOutput | UserImage = focus_row
        current_job: GenerationJob | None = None

        for _ in range(_LINEAGE_DEPTH_CAP):
            parent_source: LibraryAssetSource
            parent_id: UUID
            relation: LineageRelation
            ts: int | None

            if isinstance(current_row, GenerationOutput):
                if current_job is None:
                    current_job = await LibraryService._lineage_output_visible(
                        current_row, user_id, product_id, job_repo
                    )
                if current_job is None:
                    break
                if current_job.input_image_id is not None:
                    parent_source = LibraryAssetSource.UPLOAD
                    parent_id = current_job.input_image_id
                    relation = LineageRelation.GENERATED_FROM_UPLOAD
                    ts = None
                elif current_job.source_output_id is not None:
                    parent_source = LibraryAssetSource.OUTPUT
                    parent_id = current_job.source_output_id
                    relation = LineageRelation.GENERATED_FROM_OUTPUT
                    ts = None
                else:
                    break
            elif current_row.source_output_id is not None:
                parent_source = LibraryAssetSource.OUTPUT
                parent_id = current_row.source_output_id
                relation = LineageRelation.FRAME_OF_OUTPUT
                ts = current_row.source_timestamp_ms
            elif current_row.source_upload_id is not None:
                parent_source = LibraryAssetSource.UPLOAD
                parent_id = current_row.source_upload_id
                relation = LineageRelation.FRAME_OF_UPLOAD
                ts = current_row.source_timestamp_ms
            else:
                break

            if (parent_source, parent_id) in visited:
                break

            parent_row: GenerationOutput | UserImage | None
            if parent_source == LibraryAssetSource.UPLOAD:
                parent_row = await image_repo.get(parent_id, user_id=user_id)
            else:
                parent_row = await output_repo.get(parent_id, user_id=user_id)
            if parent_row is None or parent_row.product_id != product_id or parent_row.is_thumbnail:
                break

            parent_job: GenerationJob | None = None
            if isinstance(parent_row, GenerationOutput):
                parent_job = await LibraryService._lineage_output_visible(
                    parent_row, user_id, product_id, job_repo
                )
                if parent_job is None:
                    break

            edges.append((relation, parent_source, parent_row, ts))
            visited.add((parent_source, parent_id))
            current_row = parent_row
            current_job = parent_job
        else:
            return edges, True

        return edges, False

    @staticmethod
    def _build_group_output_item(
        output: GenerationOutput,
        derivatives: list[GenerationOutput],
    ) -> LibraryOutputItem:
        return LibraryOutputItem(
            id=output.id,
            asset_ref=format_asset_ref(LibraryAssetSource.OUTPUT, output.id),
            output_index=output.output_index,
            created_at=output.created_at,
            expires_at=output.expires_at,
            media=_build_media_object(
                source=LibraryAssetSource.OUTPUT,
                asset_id=output.id,
                width=output.width,
                height=output.height,
                content_type=output.content_type,
                size_bytes=output.size_bytes,
                derivatives=derivatives,
            ),
        )


def _build_media_object(
    *,
    source: LibraryAssetSource,
    asset_id: UUID,
    width: int | None,
    height: int | None,
    content_type: str,
    size_bytes: int,
    derivatives: Sequence[GenerationOutput | UserImage],
) -> MediaObject:
    """Build a MediaObject from normalized fields shared by both source tables.

    Distinct from build_output_media/build_upload_media in services/media.py:
    those take a concrete GenerationOutput/UserImage ORM instance, but the
    library list path works from LibraryAssetRow — a cross-table projection
    that isn't either concrete type. Same construction logic, different input
    shape; the URL prefix constants are still reused from services/media.py.
    """
    prefix = OUTPUT_PREFIX if source == LibraryAssetSource.OUTPUT else UPLOAD_PREFIX
    media_type = (
        OutputMediaType.VIDEO if content_type.startswith("video/") else OutputMediaType.IMAGE
    )

    original = MediaOriginal(
        url=f"{prefix}/{asset_id}",
        width=width,
        height=height,
        content_type=content_type,
        size_bytes=size_bytes,
    )

    variants: list[ImageVariant] = []
    for d in derivatives:
        label = label_for_max_edge(d.thumbnail_max_edge)
        if label is None or d.width is None or d.height is None:
            continue
        variants.append(
            ImageVariant(label=label, width=d.width, height=d.height, url=f"{prefix}/{d.id}")
        )
    variants.sort(key=lambda v: v.width)

    return MediaObject(media_type=media_type, original=original, variants=variants)
