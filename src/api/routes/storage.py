"""Storage API routes for user content management.

Provides endpoints for uploading images, retrieving content,
and managing user storage.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Annotated
from uuid import UUID

import msgspec
from litestar import Controller, Response, delete, get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body, Parameter
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_204_NO_CONTENT,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
)

from src.api.services.user_content import (
    UserContentError,
    UserContentNotFoundError,
    UserContentService,
    UserContentValidationError,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Response schemas
# -----------------------------------------------------------------------------


class UploadResponse(msgspec.Struct, kw_only=True):
    """Response for successful image upload."""

    id: str
    storage_key: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: datetime
    expires_at: datetime


class ImageAccessResponse(msgspec.Struct, kw_only=True):
    """Response with presigned URL for image access."""

    id: str
    storage_key: str
    presigned_url: str
    content_type: str
    size_bytes: int
    expires_in_seconds: int


class ImageListItem(msgspec.Struct, kw_only=True):
    """Item in image list response."""

    id: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: datetime
    expires_at: datetime


class ImageListResponse(msgspec.Struct, kw_only=True):
    """Response for listing images."""

    items: list[ImageListItem]
    count: int


class OutputListItem(msgspec.Struct, kw_only=True):
    """Item in output list response."""

    id: str
    job_id: str
    content_type: str
    size_bytes: int
    output_index: int
    created_at: datetime
    expires_at: datetime


class OutputListResponse(msgspec.Struct, kw_only=True):
    """Response for listing outputs."""

    items: list[OutputListItem]
    count: int


class StorageStatsResponse(msgspec.Struct, kw_only=True):
    """Response for storage statistics."""

    upload_count: int
    output_count: int
    total_bytes: int
    total_mb: float


class ErrorResponse(msgspec.Struct, kw_only=True):
    """Error response."""

    error: str
    detail: str | None = None


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

    path = "/api/v1/storage"
    tags: Sequence[str] | None = ["Storage"]

    @post("/upload")
    async def upload_image(
        self,
        user_content: UserContentService,
        file: UploadFile,
        user_id: Annotated[
            UUID,
            Parameter(
                description="User ID (will come from auth in production)",
                required=True,
            ),
        ],
    ) -> Response[UploadResponse | ErrorResponse]:
        """Upload an image for use in generation.

        Accepts PNG, JPEG, or WebP images up to 20MB.
        Returns storage details and expiration time.

        The uploaded image can be referenced by ID in i2i generation requests.
        Images are automatically deleted after the retention period.
        """
        # Validate content type
        content_type = file.content_type or "application/octet-stream"
        if content_type not in ALLOWED_CONTENT_TYPES:
            return Response(
                content=ErrorResponse(
                    error="Invalid file type",
                    detail=f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}",
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

        # Read file data
        data = await file.read()

        # Validate size
        if len(data) > MAX_UPLOAD_SIZE:
            return Response(
                content=ErrorResponse(
                    error="File too large",
                    detail=f"Maximum size: {MAX_UPLOAD_SIZE // (1024 * 1024)}MB",
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

        if len(data) == 0:
            return Response(
                content=ErrorResponse(error="Empty file"),
                status_code=HTTP_400_BAD_REQUEST,
            )

        try:
            result = await user_content.upload_image(
                user_id=user_id,
                data=data,
                filename=file.filename or "upload.png",
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
            logger.warning(f"Upload validation failed: {e}")
            return Response(
                content=ErrorResponse(error="Validation failed", detail=str(e)),
                status_code=HTTP_400_BAD_REQUEST,
            )
        except UserContentError as e:
            logger.error(f"Upload failed: {e}")
            return Response(
                content=ErrorResponse(error="Upload failed", detail=str(e)),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @get("/uploads/{image_id:uuid}")
    async def get_upload_access(
        self,
        user_content: UserContentService,
        image_id: UUID,
        expires_in: Annotated[
            int,
            Parameter(
                ge=60,
                le=86400,
                default=3600,
                description="URL validity in seconds (1 min to 24 hours)",
            ),
        ] = 3600,
    ) -> Response[ImageAccessResponse | ErrorResponse]:
        """Get a presigned URL to access an uploaded image.

        Returns a temporary URL valid for the specified duration.
        Use this URL to download the image or pass to external services.
        """
        try:
            access = await user_content.get_upload_access(
                image_id,
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
                content=ErrorResponse(error="Image not found"),
                status_code=HTTP_404_NOT_FOUND,
            )

    @get("/uploads/{image_id:uuid}/download")
    async def download_upload(
        self,
        user_content: UserContentService,
        image_id: UUID,
    ) -> Response[bytes | ErrorResponse]:
        """Download an uploaded image directly.

        Returns the raw image bytes with appropriate content type.
        For large files, prefer using the presigned URL from GET /uploads/{id}.
        """
        try:
            # Get metadata for content type
            image = await user_content.get_upload(image_id)
            if image is None:
                return Response(
                    content=ErrorResponse(error="Image not found"),
                    status_code=HTTP_404_NOT_FOUND,
                )

            data = await user_content.download_upload(image_id)

            return Response(
                content=data,
                status_code=HTTP_200_OK,
                headers={"Content-Type": image.content_type},
            )

        except UserContentNotFoundError:
            return Response(
                content=ErrorResponse(error="Image not found"),
                status_code=HTTP_404_NOT_FOUND,
            )

    @delete("/uploads/{image_id:uuid}")
    async def delete_upload(
        self,
        user_content: UserContentService,
        image_id: UUID,
    ) -> Response[None | ErrorResponse]:
        """Delete an uploaded image.

        Removes the image from storage immediately.
        This action cannot be undone.
        """
        deleted = await user_content.delete_upload(image_id)

        if not deleted:
            return Response(
                content=ErrorResponse(error="Image not found"),
                status_code=HTTP_404_NOT_FOUND,
            )

        return Response(content=None, status_code=HTTP_204_NO_CONTENT)

    @get("/uploads")
    async def list_uploads(
        self,
        user_content: UserContentService,
        user_id: Annotated[
            UUID,
            Parameter(
                description="User ID (will come from auth in production)",
                required=True,
            ),
        ],
        limit: Annotated[int, Parameter(ge=1, le=100, default=50)] = 50,
        offset: Annotated[int, Parameter(ge=0, default=0)] = 0,
    ) -> ImageListResponse:
        """List uploaded images for a user.

        Returns paginated list of uploads ordered by creation date (newest first).
        """
        images = await user_content.list_user_uploads(
            user_id,
            limit=limit,
            offset=offset,
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

        return ImageListResponse(items=items, count=len(items))

    # -------------------------------------------------------------------------
    # Output access endpoints
    # -------------------------------------------------------------------------

    @get("/outputs/{output_id:uuid}")
    async def get_output_access(
        self,
        user_content: UserContentService,
        output_id: UUID,
        expires_in: Annotated[
            int,
            Parameter(
                ge=60,
                le=86400,
                default=3600,
                description="URL validity in seconds",
            ),
        ] = 3600,
    ) -> Response[ImageAccessResponse | ErrorResponse]:
        """Get a presigned URL to access a generated output.

        Returns a temporary URL valid for the specified duration.
        """
        try:
            access = await user_content.get_output_access(
                output_id,
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
                content=ErrorResponse(error="Output not found"),
                status_code=HTTP_404_NOT_FOUND,
            )

    @get("/outputs/{output_id:uuid}/download")
    async def download_output(
        self,
        user_content: UserContentService,
        output_id: UUID,
    ) -> Response[bytes | ErrorResponse]:
        """Download a generated output directly.

        Returns the raw image bytes with appropriate content type.
        """
        try:
            output = await user_content.get_output(output_id)
            if output is None:
                return Response(
                    content=ErrorResponse(error="Output not found"),
                    status_code=HTTP_404_NOT_FOUND,
                )

            data = await user_content.download_output(output_id)

            return Response(
                content=data,
                status_code=HTTP_200_OK,
                headers={"Content-Type": output.content_type},
            )

        except UserContentNotFoundError:
            return Response(
                content=ErrorResponse(error="Output not found"),
                status_code=HTTP_404_NOT_FOUND,
            )

    @get("/outputs")
    async def list_outputs(
        self,
        user_content: UserContentService,
        user_id: Annotated[
            UUID,
            Parameter(
                description="User ID",
                required=True,
            ),
        ],
        limit: Annotated[int, Parameter(ge=1, le=100, default=50)] = 50,
        offset: Annotated[int, Parameter(ge=0, default=0)] = 0,
    ) -> OutputListResponse:
        """List generated outputs for a user.

        Returns paginated list ordered by creation date (newest first).
        """
        outputs = await user_content.list_user_outputs(
            user_id,
            limit=limit,
            offset=offset,
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

        return OutputListResponse(items=items, count=len(items))

    @get("/jobs/{job_id:uuid}/outputs")
    async def list_job_outputs(
        self,
        user_content: UserContentService,
        job_id: UUID,
    ) -> OutputListResponse:
        """List outputs for a specific job.

        Returns outputs ordered by output index (batch order).
        """
        outputs = await user_content.list_job_outputs(job_id)

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

        return OutputListResponse(items=items, count=len(items))

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    @get("/stats")
    async def get_storage_stats(
        self,
        user_content: UserContentService,
        user_id: Annotated[
            UUID,
            Parameter(description="User ID", required=True),
        ],
    ) -> StorageStatsResponse:
        """Get storage usage statistics for a user.

        Returns counts and total size of uploads and outputs.
        """
        stats = await user_content.get_user_stats(user_id)

        return StorageStatsResponse(
            upload_count=stats["upload_count"],
            output_count=stats["output_count"],
            total_bytes=stats["total_bytes"],
            total_mb=round(stats["total_bytes"] / (1024 * 1024), 2),
        )
