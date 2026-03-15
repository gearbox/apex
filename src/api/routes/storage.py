"""Storage API routes for user content management.

Provides endpoints for uploading images, retrieving content,
and managing user storage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated
from uuid import UUID

import structlog
from litestar import Controller, Response, delete, get, post
from litestar.datastructures import UploadFile
from litestar.di import Provide
from litestar.enums import RequestEncodingType
from litestar.exceptions import NotFoundException
from litestar.params import Body, Parameter
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_204_NO_CONTENT,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
)

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.pagination import PaginatedResponse, decode_cursor, encode_cursor
from src.api.schemas.storage import (
    ImageAccessResponse,
    ImageListItem,
    OutputListItem,
    StorageStatsResponse,
    UploadResponse,
)
from src.api.security import auth_guard
from src.api.services.user_content import (
    UserContentError,
    UserContentNotFoundError,
    UserContentService,
    UserContentValidationError,
)

logger = structlog.get_logger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20MB


# -----------------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------------


class StorageController(Controller):
    """User content storage endpoints.

    Handles image uploads, downloads, and storage management.
    All content is stored in Cloudflare R2 with metadata in PostgreSQL.
    """

    path = "/v1/storage"
    tags: Sequence[str] | None = ["Storage"]
    guards = [auth_guard]
    dependencies = {"current_user_id": Provide(get_current_user_id)}

    @post("/upload")
    async def upload_image(
        self,
        current_user_id: UUID,
        user_content: UserContentService,
        data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART, title="file")],
    ) -> Response[UploadResponse | ErrorEnvelope]:
        """Upload an image for use in generation.

        Accepts PNG, JPEG, or WebP images up to 20MB.
        Returns storage details and expiration time.

        The uploaded image can be referenced by ID in i2i generation requests.
        Images are automatically deleted after the retention period.
        """
        # Validate content type
        content_type = data.content_type or "application/octet-stream"
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

        # Read file data
        file_bytes = await data.read()

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
                filename=data.filename,
                bytes=len(file_bytes),
            )
            result = await user_content.upload_image(
                user_id=current_user_id,
                data=file_bytes,
                filename=data.filename or "upload.png",
                content_type=content_type,
            )

            return Response(
                content=UploadResponse(
                    id=str(result.id),
                    storage_key=result.storage_key,
                    filename=result.filename,
                    content_type=result.content_type,
                    size_bytes=result.size_bytes,
                    created_at=result.created_at,
                    expires_at=result.expires_at,
                ),
                status_code=HTTP_201_CREATED,
            )

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
        except UserContentError as e:
            logger.error("storage.upload_failed", error=str(e))
            return Response(
                content=ErrorEnvelope(
                    error="upload_failed",
                    message=str(e),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @get("/uploads/{image_id:uuid}")
    async def get_upload_access(
        self,
        current_user_id: UUID,
        user_content: UserContentService,
        image_id: UUID,
        expires_in: Annotated[
            int,
            Parameter(
                ge=60,
                le=86400,
                description="URL validity in seconds (1 min to 24 hours)",
            ),
        ] = 3600,
    ) -> Response[ImageAccessResponse | ErrorEnvelope]:
        """Get a presigned URL to access an uploaded image.

        Returns a temporary URL valid for the specified duration.
        Only returns URLs for images owned by the authenticated user.
        """
        try:
            access = await user_content.get_upload_access(
                image_id,
                user_id=current_user_id,
                expires_in=expires_in,
            )

            return Response(
                content=ImageAccessResponse(
                    id=str(image_id),
                    storage_key=access.storage_key,
                    presigned_url=access.presigned_url,
                    content_type=access.content_type,
                    size_bytes=access.size_bytes,
                    expires_in_seconds=access.expires_in_seconds,
                ),
                status_code=HTTP_200_OK,
            )

        except UserContentNotFoundError:
            return Response(
                content=ErrorEnvelope(
                    error="not_found", message="Image not found", status_code=HTTP_404_NOT_FOUND
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

    @get("/uploads/{image_id:uuid}/download")
    async def download_upload(
        self,
        current_user_id: UUID,
        user_content: UserContentService,
        image_id: UUID,
    ) -> Response[bytes | ErrorEnvelope]:
        """Download an uploaded image directly.

        Returns the raw image bytes with appropriate content type.
        For large files, prefer using the presigned URL from GET /uploads/{id}.
        """
        try:
            # Get metadata for content type
            image = await user_content.get_upload(image_id, user_id=current_user_id)
            if image is None:
                return Response(
                    content=ErrorEnvelope(
                        error="not_found", message="Image not found", status_code=HTTP_404_NOT_FOUND
                    ),
                    status_code=HTTP_404_NOT_FOUND,
                )

            data = await user_content.download_upload(image_id, user_id=current_user_id)

            return Response(
                content=data,
                status_code=HTTP_200_OK,
                headers={"Content-Type": image.content_type},
            )

        except UserContentNotFoundError:
            return Response(
                content=ErrorEnvelope(
                    error="not_found", message="Image not found", status_code=HTTP_404_NOT_FOUND
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

    @delete("/uploads/{image_id:uuid}", status_code=HTTP_204_NO_CONTENT)
    async def delete_upload(
        self,
        current_user_id: UUID,
        user_content: UserContentService,
        image_id: UUID,
    ) -> None:
        """Delete an uploaded image.

        Removes the image from storage immediately.
        This action cannot be undone.
        """
        deleted = await user_content.delete_upload(image_id, user_id=current_user_id)

        if not deleted:
            raise NotFoundException(detail="Image not found")

    @get("/uploads")
    async def list_uploads(
        self,
        current_user_id: UUID,
        user_content: UserContentService,
        limit: Annotated[int, Parameter(ge=1, le=100)] = 50,
        offset: Annotated[int, Parameter(ge=0)] = 0,
        cursor: str | None = None,
    ) -> PaginatedResponse[ImageListItem]:
        """List uploaded images for a user.

        Returns paginated list of uploads ordered by creation date (newest first).

        Query parameters:
          - ``limit``: Page size 1–100 (default 50)
          - ``offset``: Page offset (default 0, ignored when ``cursor`` is supplied)
          - ``cursor``: Opaque cursor from a previous response's ``next_cursor``
            field.  When supplied, enables efficient keyset pagination.
        """
        cursor_ts = None
        cursor_id = None
        effective_offset = offset
        if cursor is not None:
            cursor_ts, cursor_id = decode_cursor(cursor)
            effective_offset = 0

        images, total = await user_content.list_user_uploads(
            current_user_id,
            limit=limit,
            offset=effective_offset,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
        )

        items = [
            ImageListItem(
                id=str(img.id),
                filename=img.original_filename,
                content_type=img.content_type,
                size_bytes=img.size_bytes,
                created_at=img.created_at,
                expires_at=img.expires_at,
            )
            for img in images
        ]

        has_more = effective_offset + len(items) < total
        next_cursor: str | None = None
        if has_more and images:
            last = images[-1]
            next_cursor = encode_cursor(last.created_at, last.id)

        return PaginatedResponse(
            items=items,
            total=total,
            limit=limit,
            offset=effective_offset,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    # -------------------------------------------------------------------------
    # Output access endpoints
    # -------------------------------------------------------------------------

    @get("/outputs/{output_id:uuid}")
    async def get_output_access(
        self,
        current_user_id: UUID,
        user_content: UserContentService,
        output_id: UUID,
        expires_in: Annotated[
            int,
            Parameter(
                ge=60,
                le=86400,
                description="URL validity in seconds",
            ),
        ] = 3600,
    ) -> Response[ImageAccessResponse | ErrorEnvelope]:
        """Get a presigned URL to access a generated output.

        Returns a temporary URL valid for the specified duration.
        """
        try:
            access = await user_content.get_output_access(
                output_id,
                user_id=current_user_id,
                expires_in=expires_in,
            )

            return Response(
                content=ImageAccessResponse(
                    id=str(output_id),
                    storage_key=access.storage_key,
                    presigned_url=access.presigned_url,
                    content_type=access.content_type,
                    size_bytes=access.size_bytes,
                    expires_in_seconds=access.expires_in_seconds,
                ),
                status_code=HTTP_200_OK,
            )

        except UserContentNotFoundError:
            return Response(
                content=ErrorEnvelope(
                    error="not_found", message="Output not found", status_code=HTTP_404_NOT_FOUND
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

    @get("/outputs/{output_id:uuid}/download")
    async def download_output(
        self,
        current_user_id: UUID,
        user_content: UserContentService,
        output_id: UUID,
    ) -> Response[bytes | ErrorEnvelope]:
        """Download a generated output directly.

        Returns the raw image bytes with appropriate content type.
        """
        try:
            output = await user_content.get_output(output_id, user_id=current_user_id)
            if output is None:
                return Response(
                    content=ErrorEnvelope(
                        error="not_found",
                        message="Output not found",
                        status_code=HTTP_404_NOT_FOUND,
                    ),
                    status_code=HTTP_404_NOT_FOUND,
                )

            data = await user_content.download_output(output_id, user_id=current_user_id)

            return Response(
                content=data,
                status_code=HTTP_200_OK,
                headers={"Content-Type": output.content_type},
            )

        except UserContentNotFoundError:
            return Response(
                content=ErrorEnvelope(
                    error="not_found", message="Output not found", status_code=HTTP_404_NOT_FOUND
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

    @get("/outputs")
    async def list_outputs(
        self,
        current_user_id: UUID,
        user_content: UserContentService,
        limit: Annotated[int, Parameter(ge=1, le=100)] = 50,
        offset: Annotated[int, Parameter(ge=0)] = 0,
        cursor: str | None = None,
    ) -> PaginatedResponse[OutputListItem]:
        """List generated outputs for a user.

        Returns paginated list ordered by creation date (newest first).

        Query parameters:
          - ``limit``: Page size 1–100 (default 50)
          - ``offset``: Page offset (default 0, ignored when ``cursor`` is supplied)
          - ``cursor``: Opaque cursor from a previous response's ``next_cursor``
            field.  When supplied, enables efficient keyset pagination.
        """
        cursor_ts = None
        cursor_id = None
        effective_offset = offset
        if cursor is not None:
            cursor_ts, cursor_id = decode_cursor(cursor)
            effective_offset = 0

        outputs, total = await user_content.list_user_outputs(
            current_user_id,
            limit=limit,
            offset=effective_offset,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
        )

        items = [
            OutputListItem(
                id=str(out.id),
                job_id=str(out.job_id),
                content_type=out.content_type,
                size_bytes=out.size_bytes,
                output_index=out.output_index,
                created_at=out.created_at,
                expires_at=out.expires_at,
            )
            for out in outputs
        ]

        has_more = effective_offset + len(items) < total
        next_cursor: str | None = None
        if has_more and outputs:
            last = outputs[-1]
            next_cursor = encode_cursor(last.created_at, last.id)

        return PaginatedResponse(
            items=items,
            total=total,
            limit=limit,
            offset=effective_offset,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    @get("/jobs/{job_id:uuid}/outputs")
    async def list_job_outputs(
        self,
        current_user_id: UUID,
        user_content: UserContentService,
        job_id: UUID,
    ) -> Response[PaginatedResponse[OutputListItem] | ErrorEnvelope]:
        """List outputs for a specific job.

        Returns outputs ordered by output index (batch order).
        Only accessible by the job owner.
        """
        try:
            outputs = await user_content.list_job_outputs(job_id, user_id=current_user_id)
        except UserContentNotFoundError:
            return Response(
                content=ErrorEnvelope(
                    error="not_found",
                    message="Job outputs not found",
                    status_code=HTTP_404_NOT_FOUND,
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

        items = [
            OutputListItem(
                id=str(out.id),
                job_id=str(out.job_id),
                content_type=out.content_type,
                size_bytes=out.size_bytes,
                output_index=out.output_index,
                created_at=out.created_at,
                expires_at=out.expires_at,
            )
            for out in outputs
        ]

        return Response(
            content=PaginatedResponse(
                items=items,
                total=len(items),
                limit=len(items),
                offset=0,
                has_more=False,
                next_cursor=None,
            ),
            status_code=HTTP_200_OK,
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
