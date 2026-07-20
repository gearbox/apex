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
    LibraryAssetDetail,
    LibraryAssetItem,
    LibraryAssetPatch,
    LibraryDescendants,
    LibraryGroupDetail,
    LibraryGroupLineage,
    LibraryLineage,
    LibraryOutputItem,
)
from src.api.schemas.media import ImageVariant, MediaObject, MediaOriginal
from src.api.schemas.pagination import CursorPage, decode_library_cursor, encode_library_cursor
from src.api.services.content_proxy import ContentNotFoundError
from src.api.services.library_capabilities import resolve_library_actions
from src.api.services.media import OUTPUT_PREFIX, UPLOAD_PREFIX
from src.core.enums import (
    GenerationType,
    LibraryBadge,
    LibraryGroupSourceType,
    LibrarySort,
    OutputMediaType,
)
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref, parse_asset_ref
from src.core.thumbnails import label_for_max_edge
from src.db.repositories.job import JobRepository
from src.db.repositories.library import LibraryRepository
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.content_proxy import ContentProxyService
    from src.core.product import ProductConfig
    from src.db.models.storage import GenerationJob, GenerationOutput, UserImage
    from src.db.repositories.library import LibraryAssetRow, OptionalUpdate

logger = structlog.get_logger(__name__)


class LibraryValidationError(Exception):
    """Raised when a library asset patch fails validation."""


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
            created_from: Optional lower bound on created_at.
            created_to: Optional upper bound on created_at.
            sort: newest (default) or oldest.

        Returns:
            CursorPage of LibraryAssetItem.

        Raises:
            ValueError: If ``cursor`` is malformed.
        """
        repo = LibraryRepository(session)

        decoded_cursor = decode_library_cursor(cursor) if cursor is not None else None

        rows = await repo.list_assets(
            user_id,
            product_id,
            limit=limit,
            cursor=decoded_cursor,
            source=source,
            media_type=media_type,
            model=model,
            favorite=favorite,
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
            )
            for row in page_rows
        ]

        next_cursor: str | None = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_library_cursor(last.created_at, last.source.value, last.id)

        logger.info(
            "library.list",
            user_id=str(user_id),
            product_id=product_id,
            count=len(items),
            has_more=has_more,
        )

        return CursorPage(items=items, limit=limit, has_more=has_more, next_cursor=next_cursor)

    def _build_item_from_row(
        self,
        row: LibraryAssetRow,
        *,
        derivatives: Sequence[GenerationOutput | UserImage],
        output_count: int | None,
        product_config: ProductConfig,
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
            source_output = await OutputRepository(session).get(image.source_output_id)
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
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            provider=job.provider,
            aspect_ratio=job.aspect_ratio,
            token_cost=job.token_cost,
            completed_at=job.completed_at,
            lineage=None,
            descendants=LibraryDescendants(job_count=job_count, frame_count=frame_count),
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

        Phase 1 supports ``display_title`` only (tri-state: absent = no-op,
        ``null`` = clear, string = set — validated to <=255 chars, stripped,
        empty string normalized to ``None``).

        Returns:
            Updated LibraryAssetDetail, or None if the asset doesn't exist /
            isn't owned by ``user_id``.

        Raises:
            LibraryValidationError: If ``display_title`` fails validation.
        """
        try:
            ref = parse_asset_ref(asset_ref)
        except ValueError:
            return None

        if not await self._asset_exists(ref, user_id, product_id, session):
            return None

        if patch.display_title is not msgspec.UNSET:
            display_title_update: OptionalUpdate[str | None] = self._normalize_display_title(
                patch.display_title
            )
            repo = LibraryRepository(session)
            await repo.upsert_metadata(
                user_id,
                product_id,
                ref.source,
                ref.asset_id,
                display_title=display_title_update,
            )
            logger.info("library.metadata_patched", asset_ref=asset_ref, user_id=str(user_id))

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
        """Delete a library asset via ContentProxyService, then purge its metadata.

        A typed pre-check (fetch by the ref's declared source) runs before
        delegating to ``ContentProxyService.delete_content`` — a malformed
        claim (ref says upload but the id belongs to an output) is treated
        as not-found rather than falling through to the other table.

        Returns:
            True if deleted; False if not found / not owned.
        """
        try:
            ref = parse_asset_ref(asset_ref)
        except ValueError:
            return False

        if not await self._asset_exists(ref, user_id, product_id, session):
            return False

        try:
            await content_proxy.delete_content(
                ref.asset_id,
                user_id=user_id,
                product_id=product_id,
                session=session,
            )
        except ContentNotFoundError:
            return False

        repo = LibraryRepository(session)
        await repo.delete_metadata_for_assets([(ref.source.value, ref.asset_id)])
        await session.commit()

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
