"""User content service - orchestrates R2 storage and database operations.

This is the service layer for ingesting user uploads (images, videos and
client-captured video frames) and reporting storage usage. It coordinates
between R2 storage for the file bytes and PostgreSQL for metadata tracking.

Reads of stored content go through ``ContentProxyService`` and ``LibraryService``,
not through this module.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.unified_generation import SourceMediaReference
from src.api.schemas.user_content import UploadedImage
from src.api.services.frame_lineage import FrameLineage, FrameLineageReason
from src.api.services.generation.source_media import (
    SourceMediaResolver,
    SourceMediaValidationError,
)
from src.api.services.image_normalization import (
    ImageNormalizationError,
    ImageTooLargeError,
    sniff_format,
)
from src.api.services.image_thumbnail import make_image_thumbnails
from src.api.services.media import build_upload_media
from src.api.services.media_hash_ledger import MediaHashLedger
from src.api.services.media_ingest import (
    ImageIngestPolicy,
    InvalidMediaError,
    MediaIngestor,
    MediaProcessingError,
)
from src.api.services.media_ingest.profile_log import VideoProfileOrigin, log_video_profile
from src.api.services.storage import (
    R2StorageService,
    StorageError,
    StorageType,
    StorageValidationError,
)
from src.api.services.storage.schemas import ALLOWED_VIDEO_CONTENT_TYPES
from src.api.services.thumbnail import extract_video_thumbnail
from src.core.enums import MediaKind
from src.core.library_ref import LibraryAssetSource, format_asset_ref
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models import UserImage

logger = structlog.get_logger(__name__)

_CONTROL_CHAR_RUNS = re.compile(r"[\x00-\x1f\x7f]+")
_DISPLAY_FILENAME_MAX_LEN = 255


def sanitize_display_filename(raw: str | None) -> str | None:
    """Normalize a client-supplied filename for display/search only.

    NEVER use the result for storage keys, HTTP headers, R2 metadata,
    ComfyUI inputs, or any filesystem path — ``UserImage.original_filename``
    is the canonical system identifier and is always ``{uuid}.{ext}``.

    Each maximal run of control characters (e.g. a header-injection
    ``\\r\\n``) collapses to a single space rather than being deleted
    outright — deleting would silently mash adjacent words together
    (``"photo\\r\\nX"`` -> ``"photoX"``).
    """
    if not raw:
        return None
    leaf = raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    cleaned = _CONTROL_CHAR_RUNS.sub(" ", unicodedata.normalize("NFC", leaf)).strip()
    return cleaned[:_DISPLAY_FILENAME_MAX_LEN] or None


class UserContentError(Exception):
    """Base exception for user content operations."""


class UserContentValidationError(UserContentError):
    """Raised when content validation fails."""


class UserContentTooLargeError(UserContentValidationError):
    """Raised when an uploaded image's pixel count exceeds the configured cap."""


class UserContentFrameLineageError(UserContentValidationError):
    """Raised when an uploaded frame's claimed source video fails validation.

    ``reason`` is for server-side logs only; the route maps every reason to one
    public error so the response is never an existence oracle.
    """

    def __init__(self, reason: FrameLineageReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


class UserContentStorageError(UserContentError):
    """Raised when the storage backend fails for reasons unrelated to client input."""


class UserContentUnavailableError(UserContentError):
    """Raised when media preparation is out of capacity or failed operationally."""


class UserContentService:
    """Service for managing user-uploaded and generated content.

    Coordinates between R2 storage (files) and PostgreSQL (metadata).
    Provides atomic operations that maintain consistency between both.
    """

    def __init__(
        self,
        storage: R2StorageService,
        session: AsyncSession,
        *,
        product_id: str,
        retention_days: int = 7,
        video_max_seconds: int = 300,
        media_ingestor: MediaIngestor,
    ) -> None:
        """Initialize user content service.

        Args:
            storage: R2 storage service for file operations.
            session: Database session for metadata operations.
            product_id: Product this service is operating on.
            retention_days: Days to retain content before cleanup.
            video_max_seconds: Preparation-time rejection cap for uploaded
                video duration.
        """
        self._storage = storage
        self._session = session
        self._output_repo = OutputRepository(session)
        self._image_repo = UserImageRepository(session)
        self._product_id = product_id
        self._retention_days = retention_days
        self._video_max_seconds = video_max_seconds
        self._media_ingestor = media_ingestor

    # -------------------------------------------------------------------------
    # Upload operations
    # -------------------------------------------------------------------------

    async def upload_image(
        self,
        *,
        user_id: UUID,
        data: bytes,
        filename: str,
        content_type: str,
        lineage: FrameLineage | None = None,
    ) -> UploadedImage:
        """Upload an image or video for use in generation, optionally as a captured frame.

        Despite the name, this also accepts uploaded videos (``content_type``
        in ``ALLOWED_VIDEO_CONTENT_TYPES``) — routed to ``_upload_video``.
        Video bytes are prepared from their actual container and selected
        streams (never the client MIME) before storage. A JPEG poster frame
        plus WEBP thumbnail derivatives are generated from those stored bytes.

        For images: uploads to R2 and creates database record atomically. The
        image bytes are normalized before storage (see ``image_normalization``):
        format is determined by sniffing the bytes, not the client-declared
        content type or filename. As a result, the returned
        ``content_type``/``size_bytes`` reflect the *stored* normalized
        object, which may differ from what the client originally sent (e.g. a
        mislabeled HEIC upload is stored as PNG).

        When ``lineage`` is given the upload is a frame the client captured from a
        source video it already has: the source (an owned, same-product video
        upload/output) is validated first, before any media work, and the frame
        is stored through the ordinary image path — client bytes, so the UPLOAD
        policy — with its lineage written atomically with the row. An upload
        source's retention window slides, as for any other active use.

        Args:
            user_id: Owner of the image.
            data: Raw image or video bytes.
            filename: Original filename.
            content_type: MIME type.
            lineage: Source video + capture timestamp for a client-captured frame.

        Returns:
            UploadedImage with storage details.

        Raises:
            UserContentFrameLineageError: If ``lineage`` is given and the source or
                timestamp is invalid, or the upload is itself a video.
            UserContentValidationError: If validation fails.
        """
        if lineage is not None:
            if content_type in ALLOWED_VIDEO_CONTENT_TYPES:
                raise UserContentFrameLineageError(FrameLineageReason.VIDEO_FILE)
            await self._validate_frame_lineage(user_id, lineage)

        if content_type in ALLOWED_VIDEO_CONTENT_TYPES:
            return await self._upload_video(
                user_id=user_id,
                data=data,
                filename=filename,
            )

        try:
            prepared = await self._media_ingestor.prepare_image(
                data, policy=ImageIngestPolicy.UPLOAD
            )
        except ImageTooLargeError as e:
            logger.warning(
                "user_content.upload_too_large",
                user_id=str(user_id),
                declared_content_type=content_type,
                sniffed=sniff_format(data).value,
                size_bytes=len(data),
                megapixels=e.megapixels,
                limit=e.limit,
            )
            raise UserContentTooLargeError(str(e)) from e
        except (ImageNormalizationError, InvalidMediaError) as e:
            logger.warning(
                "user_content.upload_normalization_failed",
                user_id=str(user_id),
                declared_content_type=content_type,
                sniffed=sniff_format(data).value,
                size_bytes=len(data),
                error=str(e),
            )
            raise UserContentValidationError("File is not a decodable image") from e
        except MediaProcessingError as e:
            logger.warning(
                "user_content.upload_image_preparation_unavailable", user_id=str(user_id)
            )
            raise UserContentUnavailableError("image preparation is temporarily unavailable") from e

        if prepared.converted:
            logger.info(
                "user_content.upload_normalized",
                sniffed=sniff_format(data).value,
                format=prepared.format.value,
                original_bytes=len(data),
                normalized_bytes=len(prepared.data),
                declared_content_type=content_type,
            )

        try:
            # Upload to R2 (validates size/format internally)
            result = await self._storage.upload(
                user_id=user_id,
                data=prepared.data,
                content_type=prepared.content_type,
                storage_type=StorageType.UPLOAD,
            )
            canonical_filename = f"{result.id}.{prepared.format.value}"

            now = datetime.now(UTC)
            expires_at = now + timedelta(days=self._retention_days)

            # Create database record
            db_image = await self._image_repo.create(
                id=result.id,
                user_id=user_id,
                storage_key=result.storage_key,
                original_filename=canonical_filename,
                display_filename=sanitize_display_filename(filename),
                content_type=prepared.content_type,
                size_bytes=len(prepared.data),
                format=prepared.format.value,
                expires_at=expires_at,
                product_id=self._product_id,
                width=prepared.width,
                height=prepared.height,
                source_upload_id=(
                    lineage.source.asset_id
                    if lineage and lineage.source.source is LibraryAssetSource.UPLOAD
                    else None
                ),
                source_output_id=(
                    lineage.source.asset_id
                    if lineage and lineage.source.source is LibraryAssetSource.OUTPUT
                    else None
                ),
                source_timestamp_ms=lineage.timestamp_ms if lineage else None,
            )
            await MediaHashLedger(self._session).register_upload(db_image, prepared.hash_set)
            await self._session.flush()

            logger.info(
                "user_content.uploaded",
                image_id=str(result.id),
                user_id=str(user_id),
                filename=filename,
                size_bytes=len(prepared.data),
            )
            if lineage is not None:
                await self._touch_frame_source_expiry(user_id, lineage, expires_at)

            created_derivatives: list[UserImage] = []
            # Generate sm + md WEBP thumbnails — non-fatal
            try:
                thumbnails = await make_image_thumbnails(prepared.data)
                for generated in thumbnails:
                    thumb_result = await self._storage.upload(
                        user_id=user_id,
                        data=generated.result.data,
                        content_type=generated.result.content_type,
                        storage_type=StorageType.UPLOAD,
                    )
                    async with self._session.begin_nested():
                        thumb_db = await self._image_repo.create(
                            id=thumb_result.id,
                            user_id=user_id,
                            storage_key=thumb_result.storage_key,
                            original_filename=f"{thumb_result.id}.{generated.result.format}",
                            content_type=generated.result.content_type,
                            size_bytes=len(generated.result.data),
                            format=generated.result.format,
                            expires_at=expires_at,
                            product_id=self._product_id,
                            is_thumbnail=True,
                            parent_image_id=db_image.id,
                            thumbnail_max_edge=generated.spec.max_edge,
                            width=generated.result.width,
                            height=generated.result.height,
                        )
                    created_derivatives.append(thumb_db)
            except Exception:
                logger.warning(
                    "user_content.thumbnail_generation_failed",
                    image_id=str(db_image.id),
                )

            media = build_upload_media(db_image, created_derivatives)

            return UploadedImage(
                id=db_image.id,
                storage_key=db_image.storage_key,
                filename=db_image.original_filename,
                content_type=db_image.content_type,
                size_bytes=db_image.size_bytes,
                created_at=db_image.created_at,
                expires_at=db_image.expires_at,
                media=media,
            )

        except StorageValidationError as e:
            raise UserContentValidationError(str(e)) from e
        except StorageError as e:
            logger.exception("user_content.storage_upload_failed", user_id=str(user_id))
            raise UserContentStorageError(
                f"Storage backend unavailable ({type(e).__name__})"
            ) from e

    async def _validate_frame_lineage(self, user_id: UUID, lineage: FrameLineage) -> None:
        """Require the lineage source to be an owned, same-product video containing the timestamp.

        Reuses ``SourceMediaResolver`` so ownership, product scoping and thumbnail
        rejection behave exactly as they do for generation inputs, with no existence
        oracle: every failure is a ``UserContentFrameLineageError`` whose ``reason``
        only reaches the logs.
        """
        try:
            (source,) = await SourceMediaResolver().resolve(
                [
                    SourceMediaReference(
                        asset_ref=format_asset_ref(lineage.source.source, lineage.source.asset_id)
                    )
                ],
                user_id=user_id,
                session=self._session,
                # Always pass it: ``None`` silently disables the product check.
                product_id=self._product_id,
            )
        except SourceMediaValidationError as e:
            raise UserContentFrameLineageError(FrameLineageReason.SOURCE_UNAVAILABLE) from e
        if source.media_kind is not MediaKind.VIDEO:
            raise UserContentFrameLineageError(FrameLineageReason.SOURCE_NOT_VIDEO)
        if source.duration_ms is None:
            raise UserContentFrameLineageError(FrameLineageReason.SOURCE_DURATION_UNKNOWN)
        if lineage.timestamp_ms > source.duration_ms:
            raise UserContentFrameLineageError(FrameLineageReason.TIMESTAMP_OUT_OF_RANGE)

    async def _touch_frame_source_expiry(
        self, user_id: UUID, lineage: FrameLineage, expires_at: datetime
    ) -> None:
        """Slide an upload source's retention window: capturing a frame is active use.

        Output sources are deliberately not touched.
        """
        if lineage.source.source is not LibraryAssetSource.UPLOAD:
            return
        touched = await self._image_repo.touch_expiry_many(
            [lineage.source.asset_id], user_id=user_id, expires_at=expires_at
        )
        if touched:
            logger.info(
                "user_content.frame_source_expiry_extended",
                image_id=str(lineage.source.asset_id),
                user_id=str(user_id),
                retention_days=self._retention_days,
            )

    async def _upload_video(
        self,
        *,
        user_id: UUID,
        data: bytes,
        filename: str,
    ) -> UploadedImage:
        """Upload a video for later frame extraction.

        The ingestor validates the actual container and remuxes approved
        streams without transcoding. Its metadata is authoritative; malformed
        media is a validation error while the poster-frame derivative remains
        best-effort.
        """
        try:
            prepared = await self._media_ingestor.prepare_video(
                data, max_duration_seconds=self._video_max_seconds
            )
        except InvalidMediaError as e:
            raise UserContentValidationError(str(e)) from e
        except MediaProcessingError as e:
            raise UserContentUnavailableError("video preparation is temporarily unavailable") from e

        try:
            result = await self._storage.upload(
                user_id=user_id,
                data=prepared.data,
                content_type=prepared.content_type,
                storage_type=StorageType.UPLOAD,
            )
        except StorageValidationError as e:
            raise UserContentValidationError(str(e)) from e
        except StorageError as e:
            logger.exception("user_content.storage_upload_failed", user_id=str(user_id))
            raise UserContentStorageError(
                f"Storage backend unavailable ({type(e).__name__})"
            ) from e

        now = datetime.now(UTC)
        expires_at = now + timedelta(days=self._retention_days)
        video_format = prepared.format

        db_image = await self._image_repo.create(
            id=result.id,
            user_id=user_id,
            storage_key=result.storage_key,
            original_filename=f"{result.id}.{video_format.value}",
            display_filename=sanitize_display_filename(filename),
            content_type=prepared.content_type,
            size_bytes=len(prepared.data),
            format=video_format.value,
            expires_at=expires_at,
            product_id=self._product_id,
            width=prepared.width,
            height=prepared.height,
            duration_ms=prepared.duration_ms,
        )
        await MediaHashLedger(self._session).register_upload(db_image, prepared.hash_set)
        await self._session.flush()

        logger.info(
            "user_content.video_uploaded",
            image_id=str(result.id),
            user_id=str(user_id),
            filename=filename,
            size_bytes=len(prepared.data),
            duration_ms=prepared.duration_ms,
        )
        log_video_profile(
            origin=VideoProfileOrigin.UPLOAD,
            asset_ref=format_asset_ref(LibraryAssetSource.UPLOAD, db_image.id),
            profile=prepared.stream_profile,
            width=prepared.width,
            height=prepared.height,
            duration_ms=prepared.duration_ms,
        )

        created_derivatives: list[UserImage] = []
        # Poster frame + sm/md WEBP thumbnails — non-fatal, mirrors the
        # prepared video bytes are the source for optional poster derivatives.
        try:
            poster = await extract_video_thumbnail(prepared.data)
            if poster is not None:
                thumbnails = await make_image_thumbnails(poster)
                for generated in thumbnails:
                    thumb_result = await self._storage.upload(
                        user_id=user_id,
                        data=generated.result.data,
                        content_type=generated.result.content_type,
                        storage_type=StorageType.UPLOAD,
                    )
                    async with self._session.begin_nested():
                        thumb_db = await self._image_repo.create(
                            id=thumb_result.id,
                            user_id=user_id,
                            storage_key=thumb_result.storage_key,
                            original_filename=f"{thumb_result.id}.{generated.result.format}",
                            content_type=generated.result.content_type,
                            size_bytes=len(generated.result.data),
                            format=generated.result.format,
                            expires_at=expires_at,
                            product_id=self._product_id,
                            is_thumbnail=True,
                            parent_image_id=db_image.id,
                            thumbnail_max_edge=generated.spec.max_edge,
                            width=generated.result.width,
                            height=generated.result.height,
                        )
                    created_derivatives.append(thumb_db)
            else:
                logger.warning("user_content.video_poster_skipped", image_id=str(db_image.id))
        except Exception:
            logger.warning(
                "user_content.thumbnail_generation_failed",
                image_id=str(db_image.id),
            )

        media = build_upload_media(db_image, created_derivatives)

        return UploadedImage(
            id=db_image.id,
            storage_key=db_image.storage_key,
            filename=db_image.original_filename,
            content_type=db_image.content_type,
            size_bytes=db_image.size_bytes,
            created_at=db_image.created_at,
            expires_at=db_image.expires_at,
            media=media,
        )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    async def get_user_stats(self, user_id: UUID) -> dict[str, int]:
        """Get storage statistics for a user.

        Aggregates upload and output counts from their respective
        repositories.

        Args:
            user_id: User to get stats for.

        Returns:
            Dict with upload_count, output_count, total_bytes.
        """
        upload_count, upload_bytes = await self._image_repo.count_and_sum_by_user(user_id)
        output_count, output_bytes = await self._output_repo.count_and_sum_by_user(user_id)

        return {
            "upload_count": upload_count,
            "output_count": output_count,
            "total_bytes": upload_bytes + output_bytes,
        }
