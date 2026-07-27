"""User content service - orchestrates R2 storage and database operations.

This is the main service layer for handling user content (uploads and outputs).
It coordinates between R2 storage for actual file storage and PostgreSQL
for metadata tracking and efficient queries.

All single-resource access methods require a user_id parameter and verify
ownership before returning data. This ensures defense-in-depth: even if
a route guard is misconfigured, the service layer will reject cross-user access.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.user_content import GeneratedImage, ImageAccess, UploadedImage
from src.api.services.frames.ffmpeg import FfprobeError
from src.api.services.frames.ffmpeg import probe as ffmpeg_probe
from src.api.services.image_normalization import (
    ImageNormalizationError,
    ImageTooLargeError,
    normalize_image,
    sniff_format,
)
from src.api.services.image_thumbnail import make_image_thumbnails, read_dimensions
from src.api.services.media import build_upload_media
from src.api.services.storage import (
    MediaFormat,
    R2StorageService,
    StorageError,
    StorageNotFoundError,
    StorageType,
    StorageValidationError,
)
from src.api.services.storage.schemas import ALLOWED_VIDEO_CONTENT_TYPES
from src.api.services.thumbnail import extract_video_thumbnail
from src.db.repositories.job import JobRepository
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.frames.ffmpeg import VideoProbe
    from src.db.models import GenerationOutput, UserImage

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


class UserContentNotFoundError(UserContentError):
    """Raised when requested content doesn't exist."""


class UserContentValidationError(UserContentError):
    """Raised when content validation fails."""


class UserContentTooLargeError(UserContentValidationError):
    """Raised when an uploaded image's pixel count exceeds the configured cap."""


class UserContentStorageError(UserContentError):
    """Raised when the storage backend fails for reasons unrelated to client input."""


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
        max_input_megapixels: float = 100.0,
        video_max_seconds: int = 300,
        ffmpeg_timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize user content service.

        Args:
            storage: R2 storage service for file operations.
            session: Database session for metadata operations.
            product_id: Product this service is operating on.
            retention_days: Days to retain content before cleanup.
            max_input_megapixels: Pixel-count cap enforced before decode
                (see ``image_normalization.py``); guards against
                decompression-bomb uploads.
            video_max_seconds: Upload-time ffprobe rejection cap for
                uploaded video duration.
            ffmpeg_timeout_seconds: Wall-clock timeout for the ffprobe
                subprocess invoked on video upload.
        """
        self._storage = storage
        self._job_repo = JobRepository(session)
        self._output_repo = OutputRepository(session)
        self._image_repo = UserImageRepository(session)
        self._product_id = product_id
        self._retention_days = retention_days
        self._max_input_megapixels = max_input_megapixels
        self._video_max_seconds = video_max_seconds
        self._ffmpeg_timeout_seconds = ffmpeg_timeout_seconds

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
    ) -> UploadedImage:
        """Upload an image or video for use in generation / frame extraction.

        Despite the name, this also accepts uploaded videos (``content_type``
        in ``ALLOWED_VIDEO_CONTENT_TYPES``) — routed to ``_upload_video``.
        Video bytes are probed with ffprobe (never trust the client MIME) and
        stored as-is, no normalization; a JPEG poster frame + WEBP thumbnail
        derivatives are generated the same way image thumbnails are.

        For images: uploads to R2 and creates database record atomically. The
        image bytes are normalized before storage (see ``image_normalization``):
        format is determined by sniffing the bytes, not the client-declared
        content type or filename. As a result, the returned
        ``content_type``/``size_bytes`` reflect the *stored* normalized
        object, which may differ from what the client originally sent (e.g. a
        mislabeled HEIC upload is stored as PNG).

        Args:
            user_id: Owner of the image.
            data: Raw image or video bytes.
            filename: Original filename.
            content_type: MIME type.

        Returns:
            UploadedImage with storage details.

        Raises:
            UserContentValidationError: If validation fails.
        """
        if content_type in ALLOWED_VIDEO_CONTENT_TYPES:
            return await self._upload_video(
                user_id=user_id,
                data=data,
                filename=filename,
                content_type=content_type,
            )

        try:
            normalized = await normalize_image(data, max_megapixels=self._max_input_megapixels)
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
        except ImageNormalizationError as e:
            logger.warning(
                "user_content.upload_normalization_failed",
                user_id=str(user_id),
                declared_content_type=content_type,
                sniffed=sniff_format(data).value,
                size_bytes=len(data),
                error=str(e),
            )
            raise UserContentValidationError("File is not a decodable image") from e

        if normalized.converted:
            logger.info(
                "user_content.upload_normalized",
                sniffed=normalized.sniffed.value,
                format=normalized.format.value,
                original_bytes=len(data),
                normalized_bytes=len(normalized.data),
                declared_content_type=content_type,
            )

        try:
            # Upload to R2 (validates size/format internally)
            result = await self._storage.upload(
                user_id=user_id,
                data=normalized.data,
                content_type=normalized.content_type,
                storage_type=StorageType.UPLOAD,
            )
            canonical_filename = f"{result.id}.{normalized.format.value}"

            now = datetime.now(UTC)
            expires_at = now + timedelta(days=self._retention_days)

            # Read original dimensions before creating the DB record (F4)
            dims = await read_dimensions(normalized.data)

            # Create database record
            db_image = await self._image_repo.create(
                id=result.id,
                user_id=user_id,
                storage_key=result.storage_key,
                original_filename=canonical_filename,
                display_filename=sanitize_display_filename(filename),
                content_type=normalized.content_type,
                size_bytes=len(normalized.data),
                format=normalized.format.value,
                expires_at=expires_at,
                product_id=self._product_id,
                width=dims.width if dims is not None else None,
                height=dims.height if dims is not None else None,
            )

            logger.info(
                "user_content.uploaded",
                image_id=str(result.id),
                user_id=str(user_id),
                filename=filename,
                size_bytes=len(normalized.data),
            )

            created_derivatives: list[UserImage] = []
            # Generate sm + md WEBP thumbnails — non-fatal
            try:
                thumbnails = await make_image_thumbnails(normalized.data)
                for generated in thumbnails:
                    thumb_result = await self._storage.upload(
                        user_id=user_id,
                        data=generated.result.data,
                        content_type=generated.result.content_type,
                        storage_type=StorageType.UPLOAD,
                    )
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

    async def _upload_video(
        self,
        *,
        user_id: UUID,
        data: bytes,
        filename: str,
        content_type: str,
    ) -> UploadedImage:
        """Upload a video for later frame extraction.

        Unlike images, video bytes are stored as-is — no re-encoding. The
        probe result (never the client-declared MIME) is authoritative:
        probe failure is fatal (validation error), unlike the poster-frame
        derivative below, which is best-effort.
        """
        probe = await self._probe_video_bytes(data)

        max_duration_ms = self._video_max_seconds * 1000
        if probe.duration_ms > max_duration_ms:
            raise UserContentValidationError(
                f"Video duration {probe.duration_ms / 1000:.1f}s exceeds "
                f"maximum {self._video_max_seconds}s"
            )

        try:
            result = await self._storage.upload(
                user_id=user_id,
                data=data,
                content_type=content_type,
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
        video_format = MediaFormat.from_content_type(content_type)

        db_image = await self._image_repo.create(
            id=result.id,
            user_id=user_id,
            storage_key=result.storage_key,
            original_filename=f"{result.id}.{video_format.value}",
            display_filename=sanitize_display_filename(filename),
            content_type=content_type,
            size_bytes=len(data),
            format=video_format.value,
            expires_at=expires_at,
            product_id=self._product_id,
            width=probe.width,
            height=probe.height,
            duration_ms=probe.duration_ms,
        )

        logger.info(
            "user_content.video_uploaded",
            image_id=str(result.id),
            user_id=str(user_id),
            filename=filename,
            size_bytes=len(data),
            duration_ms=probe.duration_ms,
        )

        created_derivatives: list[UserImage] = []
        # Poster frame + sm/md WEBP thumbnails — non-fatal, mirrors the
        # existing GrokJobService._store_video_result poster pipeline.
        try:
            poster = await extract_video_thumbnail(data)
            if poster is not None:
                thumbnails = await make_image_thumbnails(poster)
                for generated in thumbnails:
                    thumb_result = await self._storage.upload(
                        user_id=user_id,
                        data=generated.result.data,
                        content_type=generated.result.content_type,
                        storage_type=StorageType.UPLOAD,
                    )
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

    async def _probe_video_bytes(self, data: bytes) -> VideoProbe:
        """Write ``data`` to a temp file and ffprobe it.

        Raises:
            UserContentValidationError: If the bytes aren't a decodable video
                with a video stream (probe failure is fatal — never trust the
                client-declared content type).
        """

        def _write_temp() -> Path:
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
                f.write(data)
                return Path(f.name)

        temp_path = await asyncio.to_thread(_write_temp)
        try:
            return await ffmpeg_probe(temp_path, timeout_seconds=self._ffmpeg_timeout_seconds)
        except FfprobeError as e:
            logger.warning("user_content.video_probe_failed", error=str(e))
            raise UserContentValidationError("File is not a decodable video") from e
        finally:
            await asyncio.to_thread(temp_path.unlink, missing_ok=True)

    async def get_upload(self, image_id: UUID, *, user_id: UUID) -> UserImage | None:
        """Get upload metadata by ID.

        Args:
            image_id: Image ID to look up.
            user_id: Requesting user (must be owner).

        Returns:
            UserImage if found, None otherwise.
        """
        return await self._image_repo.get(image_id, user_id=user_id)

    async def get_upload_by_key(self, storage_key: str) -> UserImage | None:
        """Get upload metadata by storage key.

        Args:
            storage_key: R2 storage key.

        Returns:
            UserImage if found, None otherwise.
        """
        return await self._image_repo.get_by_key(storage_key)

    async def get_upload_access(
        self,
        image_id: UUID,
        *,
        user_id: UUID,
        expires_in: int = 3600,
    ) -> ImageAccess:
        """Get presigned URL for accessing an upload.

        Args:
            image_id: Image ID to access.
            user_id: Requesting user (must be owner).
            expires_in: URL validity in seconds.

        Returns:
            ImageAccess with presigned URL.

        Raises:
            UserContentNotFoundError: If image doesn't exist.
        """
        image = await self._image_repo.get(image_id, user_id=user_id)
        if image is None:
            raise UserContentNotFoundError(f"Image not found: {image_id}")

        result = await self._storage.get_presigned_url(
            image.storage_key,
            expires_in=expires_in,
        )

        return ImageAccess(
            storage_key=result.storage_key,
            presigned_url=result.presigned_url,
            content_type=result.content_type,
            size_bytes=result.size_bytes,
            expires_in_seconds=result.expires_in_seconds,
        )

    async def download_upload(self, image_id: UUID, *, user_id: UUID) -> bytes:
        """Download upload content.

        Args:
            image_id: Image ID to download.
            user_id: Requesting user (must be owner).

        Returns:
            Raw image bytes.

        Raises:
            UserContentNotFoundError: If image doesn't exist.
        """
        image = await self._image_repo.get(image_id, user_id=user_id)
        if image is None:
            raise UserContentNotFoundError(f"Image not found: {image_id}")

        try:
            return await self._storage.download(image.storage_key)
        except StorageNotFoundError as e:
            # DB record exists but R2 file missing - data inconsistency
            logger.exception(
                "r2.file_missing",
                image_id=str(image_id),
                storage_key=image.storage_key,
            )
            raise UserContentNotFoundError(f"Image file not found: {image_id}") from e

    async def list_upload_derivatives(self, image_id: UUID) -> list[UserImage]:
        """Return derivative (thumbnail) rows for a single upload.

        Args:
            image_id: Parent upload ID.

        Returns:
            List of derivative UserImage rows.
        """
        return list(await self._image_repo.list_derivatives(image_id))

    async def batch_output_derivatives(
        self, output_ids: list[UUID]
    ) -> dict[UUID, list[GenerationOutput]]:
        """Return derivative rows for a batch of outputs.

        Args:
            output_ids: Parent output IDs.

        Returns:
            Mapping from parent_output_id to list of derivative rows.
        """
        return await self._output_repo.batch_derivatives(output_ids)

    async def delete_upload(self, image_id: UUID, *, user_id: UUID) -> bool:
        """Delete an uploaded image.

        Removes from both R2 and database.

        Args:
            image_id: Image ID to delete.
            user_id: Requesting user (must be owner).

        Returns:
            True if deleted, False if not found.
        """
        image = await self._image_repo.get(image_id, user_id=user_id)
        if image is None:
            return False

        # Delete derivative (thumbnail) R2 objects first; DB cascade removes rows.
        derivatives = await self._image_repo.list_derivatives(image_id)
        for derivative in derivatives:
            await self._storage.delete(derivative.storage_key)

        await self._storage.delete(image.storage_key)
        await self._image_repo.delete(image_id, user_id=user_id)

        logger.info("user_content.deleted", image_id=str(image_id))
        return True

    # -------------------------------------------------------------------------
    # Output operations
    # -------------------------------------------------------------------------

    async def store_output(
        self,
        *,
        user_id: UUID,
        job_id: UUID,
        data: bytes,
        content_type: str,
        output_index: int,
        input_image_id: UUID | None = None,
    ) -> GeneratedImage:
        """Store a generated output image.

        Uploads to R2 and creates database record atomically.

        Args:
            user_id: Owner of the output.
            job_id: Associated generation job.
            data: Raw image bytes.
            content_type: MIME type.
            output_index: Index in batch (0-based).
            input_image_id: Associated input image (for i2i).

        Returns:
            GeneratedImage with storage details.
        """
        # Upload to R2
        result = await self._storage.upload(
            user_id=user_id,
            data=data,
            content_type=content_type,
            storage_type=StorageType.OUTPUT,
            job_id=job_id,
        )

        # Determine format
        image_format = MediaFormat.from_content_type(content_type)
        now = datetime.now(UTC)
        expires_at = now + timedelta(days=self._retention_days)

        # Create database record
        db_output = await self._output_repo.create(
            id=result.id,
            user_id=user_id,
            job_id=job_id,
            storage_key=result.storage_key,
            content_type=content_type,
            size_bytes=len(data),
            format=image_format.value,
            output_index=output_index,
            expires_at=expires_at,
            input_image_id=input_image_id,
            product_id=self._product_id,
        )

        logger.info(
            "user_content.output_stored",
            output_id=str(result.id),
            job_id=str(job_id),
            output_index=output_index,
            size_bytes=len(data),
        )

        return GeneratedImage(
            id=db_output.id,
            job_id=db_output.job_id,
            storage_key=db_output.storage_key,
            content_type=db_output.content_type,
            size_bytes=db_output.size_bytes,
            output_index=db_output.output_index,
            created_at=db_output.created_at,
            expires_at=db_output.expires_at,
        )

    async def get_output(self, output_id: UUID, *, user_id: UUID) -> GenerationOutput | None:
        """Get output metadata by ID.

        Args:
            output_id: Output ID to look up.
            user_id: Requesting user (must be owner).

        Returns:
            GenerationOutput if found, None otherwise.
        """
        return await self._output_repo.get(output_id, user_id=user_id)

    async def get_output_access(
        self,
        output_id: UUID,
        *,
        user_id: UUID,
        expires_in: int = 3600,
    ) -> ImageAccess:
        """Get presigned URL for accessing an output.

        Args:
            output_id: Output ID to access.
            user_id: Requesting user (must be owner).
            expires_in: URL validity in seconds.

        Returns:
            ImageAccess with presigned URL.

        Raises:
            UserContentNotFoundError: If output doesn't exist.
        """
        output = await self._output_repo.get(output_id, user_id=user_id)
        if output is None:
            raise UserContentNotFoundError(f"Output not found: {output_id}")

        result = await self._storage.get_presigned_url(
            output.storage_key,
            expires_in=expires_in,
        )

        return ImageAccess(
            storage_key=result.storage_key,
            presigned_url=result.presigned_url,
            content_type=result.content_type,
            size_bytes=result.size_bytes,
            expires_in_seconds=result.expires_in_seconds,
        )

    async def download_output(self, output_id: UUID, *, user_id: UUID) -> bytes:
        """Download output content.

        Args:
            output_id: Output ID to download.
            user_id: Requesting user (must be owner).
        Returns:
            Raw image bytes.

        Raises:
            UserContentNotFoundError: If output doesn't exist or is not owned by the user.
        """
        output = await self._output_repo.get(output_id, user_id=user_id)
        if output is None:
            raise UserContentNotFoundError(f"Output not found: {output_id}")

        try:
            return await self._storage.download(output.storage_key)
        except StorageNotFoundError as e:
            logger.exception(
                "r2.file_missing",
                output_id=str(output_id),
                storage_key=output.storage_key,
            )
            raise UserContentNotFoundError(f"Output file not found: {output_id}") from e

    async def list_job_outputs(
        self,
        job_id: UUID,
        *,
        user_id: UUID,
    ) -> list[GenerationOutput]:
        """List outputs for a job.

        Args:
            job_id: Job to list outputs for.
            user_id: Requesting user (must be owner of the outputs).

        Returns:
            List of GenerationOutput metadata ordered by index.
        """
        # Verify job ownership
        job = await self._job_repo.get(job_id, user_id=user_id)
        if job is None:
            raise UserContentNotFoundError(f"Job not found: {job_id}")

        outputs = await self._output_repo.list_by_job(job_id)
        return list(outputs)

    async def list_user_outputs(
        self,
        user_id: UUID,
        *,
        limit: int = 100,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> list[GenerationOutput]:
        """List outputs for a user.

        Uses limit+1 fetch pattern — caller checks ``len(result) > limit``
        to determine ``has_more``.

        Args:
            user_id: User to list outputs for.
            limit: Maximum results (fetch limit+1 for has_more).
            cursor_ts: ``created_at`` of the last item on the previous page.
            cursor_id: ``id`` of the last item on the previous page.

        Returns:
            List of GenerationOutput instances.
        """
        outputs = await self._output_repo.list_by_user(
            user_id,
            limit=limit,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
        )
        return list(outputs)

    # -------------------------------------------------------------------------
    # Storage key utilities (for ComfyUI integration)
    # -------------------------------------------------------------------------

    def get_upload_storage_key(self, image_id: UUID, user_id: UUID, format: MediaFormat) -> str:
        """Get the R2 storage key for an upload.

        Useful for passing to ComfyUI S3 nodes.

        Args:
            image_id: Image file ID.
            user_id: Owner of the image.
            format: Image format.

        Returns:
            Full R2 storage key.
        """
        return self._storage.build_storage_key(
            user_id=user_id,
            file_id=image_id,
            storage_type=StorageType.UPLOAD,
            format=format,
        )

    def get_output_storage_key(
        self,
        output_id: UUID,
        user_id: UUID,
        job_id: UUID,
        format: MediaFormat,
    ) -> str:
        """Get the R2 storage key for an output.

        Args:
            output_id: Output file ID.
            user_id: Owner of the output.
            job_id: Associated job.
            format: Image format.

        Returns:
            Full R2 storage key.
        """
        return self._storage.build_storage_key(
            user_id=user_id,
            file_id=output_id,
            storage_type=StorageType.OUTPUT,
            format=format,
            job_id=job_id,
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
