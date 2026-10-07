"""Storage API routes for user content management.

Provides endpoints for uploading images/videos (including client-captured
video frames) and reading storage statistics. Content is read through the
content proxy (``/v1/content``) and the library (``/v1/library``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import structlog
from litestar import Controller, Response, get, post
from litestar.datastructures import UploadFile  # noqa: TC002
from litestar.di import Provide
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.status_codes import (
    HTTP_201_CREATED,
    HTTP_400_BAD_REQUEST,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_502_BAD_GATEWAY,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.storage import (
    StorageStatsResponse,
    UploadResponse,
)
from src.api.security import auth_guard
from src.api.services.frame_lineage import (
    FrameLineageError,
    FrameLineageReason,
    parse_frame_lineage,
)
from src.api.services.storage.schemas import (
    ALLOWED_CLIENT_UPLOAD_CONTENT_TYPES,
    ALLOWED_VIDEO_CONTENT_TYPES,
)
from src.api.services.user_content import (
    UserContentError,
    UserContentFrameLineageError,
    UserContentService,
    UserContentStorageError,
    UserContentTooLargeError,
    UserContentUnavailableError,
    UserContentValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ALLOWED_CONTENT_TYPES = ALLOWED_CLIENT_UPLOAD_CONTENT_TYPES
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20MB


def _invalid_frame_lineage(
    reason: FrameLineageReason,
) -> Response[UploadResponse | ErrorEnvelope]:
    """The one public response for every frame-lineage failure (no existence oracle).

    The specific reason is logged, never returned.
    """
    logger.warning("storage.upload_frame_lineage_rejected", reason=reason.value)
    return Response(
        content=ErrorEnvelope(
            error="invalid_frame_lineage",
            message="Frame source is not available",
            status_code=HTTP_400_BAD_REQUEST,
        ),
        status_code=HTTP_400_BAD_REQUEST,
    )


@dataclass
class UploadForm:
    data: UploadFile
    # Frame lineage (client-captured video frame). Raw strings, parsed explicitly by
    # ``parse_frame_lineage`` — never framework-coerced — so "1e3", " 12", "-0", "12.0"
    # and "" are rejected uniformly. Both absent = ordinary upload; exactly one = error.
    source_asset_ref: str | None = None
    source_timestamp_ms: str | None = None


# -----------------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------------


class StorageController(Controller):
    """User content storage endpoints.

    Handles uploads and storage statistics only.
    All content is stored in Cloudflare R2 with metadata in PostgreSQL.
    """

    path = "/v1/storage"
    tags: Sequence[str] | None = ("Storage",)
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @post("/upload")
    async def upload_image(
        self,
        current_user_id: UUID,
        user_content: UserContentService,
        data: Annotated[UploadForm, Body(media_type=RequestEncodingType.MULTI_PART)],
    ) -> Response[UploadResponse | ErrorEnvelope]:
        """Upload an image or video for use in generation, or a captured video frame.

        Accepts PNG, JPEG, WebP, HEIC/HEIF, or AVIF images up to 20MB;
        non-PNG/JPEG/WebP inputs are converted to PNG. Also accepts MP4,
        WebM, or QuickTime (.mov) videos up to 20MB — videos are probed
        with ffprobe (rejected if undecodable, not a video, or longer than
        ``media_video_max_duration_seconds``) and stored as-is, with a JPEG
        poster frame derivative. Returns storage details and expiration time.

        Uploaded images and videos can be referenced by ``asset_ref`` in
        generation requests. Content is automatically deleted after the
        retention period.

        A frame the client captured from a video it already has is uploaded as an
        image with two extra multipart fields: ``source_asset_ref``
        (``upload:<uuid>`` / ``output:<uuid>``, the source video) and
        ``source_timestamp_ms`` (decimal integer, ``0 <= ts <= duration_ms`` of the
        source). Both or neither: the frame is stored with that lineage and shows up
        under the source's library lineage. Any problem with the lineage — partial,
        malformed, unknown/foreign/other-product source, not a video, timestamp out
        of range, or the upload itself being a video — returns the single error
        ``400 invalid_frame_lineage``.
        """
        # Validate content type
        content_type = data.data.content_type or "application/octet-stream"
        if content_type not in ALLOWED_CONTENT_TYPES:
            return Response(
                content=ErrorEnvelope(
                    error="invalid_file_type",
                    message=f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}",
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )
        logger.debug("storage.upload_started", content_type=content_type)

        # Frame lineage: parsed and shape-checked before any media work or DB access.
        # The client-declared type is only an early filter; image preparation stays
        # authoritative about the bytes.
        try:
            lineage = parse_frame_lineage(data.source_asset_ref, data.source_timestamp_ms)
        except FrameLineageError as e:
            return _invalid_frame_lineage(e.reason)
        if lineage is not None and content_type in ALLOWED_VIDEO_CONTENT_TYPES:
            return _invalid_frame_lineage(FrameLineageReason.VIDEO_FILE)

        # Read file data
        file_bytes = await data.data.read()

        # Validate size
        if len(file_bytes) > MAX_UPLOAD_SIZE:
            return Response(
                content=ErrorEnvelope(
                    error="file_too_large",
                    message=f"Maximum size: {MAX_UPLOAD_SIZE // (1024 * 1024)}MB",
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

        if len(file_bytes) == 0:
            return Response(
                content=ErrorEnvelope(
                    error="empty_file",
                    message="Uploaded file is empty",
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )
        logger.debug("storage.upload_size", bytes=len(file_bytes))

        try:
            logger.debug(
                "storage.uploading_image",
                user_id=str(current_user_id),
                filename=data.data.filename,
                bytes=len(file_bytes),
            )
            result = await user_content.upload_image(
                user_id=current_user_id,
                data=file_bytes,
                filename=data.data.filename or "data.png",
                content_type=content_type,
                lineage=lineage,
            )
            return Response(
                content=UploadResponse(
                    id=str(result.id),
                    filename=result.filename,
                    created_at=result.created_at,
                    expires_at=result.expires_at,
                    media=result.media,
                ),
                status_code=HTTP_201_CREATED,
            )

        except UserContentTooLargeError as e:
            logger.warning("storage.upload_too_large", error=str(e))
            return Response(
                content=ErrorEnvelope(
                    error="file_too_large",
                    message=str(e),
                    status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                ),
                status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
        except UserContentFrameLineageError as e:
            # Must precede UserContentValidationError (its base class).
            return _invalid_frame_lineage(e.reason)
        except UserContentValidationError as e:
            logger.warning("storage.upload_validation_failed", error=str(e))
            return Response(
                content=ErrorEnvelope(
                    error="validation_error",
                    message=str(e),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )
        except UserContentStorageError as e:
            logger.warning("storage.upload_upstream_error", error=str(e))
            return Response(
                content=ErrorEnvelope(
                    error="upstream_error",
                    message="Storage backend unavailable",
                    status_code=HTTP_502_BAD_GATEWAY,
                ),
                status_code=HTTP_502_BAD_GATEWAY,
            )
        except UserContentUnavailableError as e:
            logger.warning("storage.upload_media_unavailable", error=str(e))
            return Response(
                content=ErrorEnvelope(
                    error="service_unavailable",
                    message="Media processing is temporarily unavailable",
                    status_code=HTTP_503_SERVICE_UNAVAILABLE,
                ),
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
            )
        except UserContentError as e:
            logger.exception("storage.upload_failed", error=str(e))
            return Response(
                content=ErrorEnvelope(
                    error="upload_failed",
                    message=str(e),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    @get("/stats")
    async def get_storage_stats(
        self,
        current_user_id: UUID,
        user_content: UserContentService,
    ) -> StorageStatsResponse:
        """Get storage usage statistics for a user.

        Returns counts and total size of uploads and outputs.
        """
        stats = await user_content.get_user_stats(current_user_id)

        return StorageStatsResponse(
            upload_count=stats["upload_count"],
            output_count=stats["output_count"],
            total_bytes=stats["total_bytes"],
            total_mb=round(stats["total_bytes"] / (1024 * 1024), 2),
        )
