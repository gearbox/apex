"""Content proxy — auth-gated streaming access to R2 objects.

Provides stable, non-expiring URLs for user content. The server
resolves ownership, then streams bytes from R2 with Cache-Control
headers. Presigned URLs are never exposed to the client.

Endpoints:
  GET /v1/content/outputs/{output_id}  — stream a generated output
  GET /v1/content/uploads/{image_id}   — stream an uploaded image
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from litestar import Controller, Response, delete, get
from litestar.di import Provide
from litestar.exceptions import NotFoundException
from litestar.response import Stream
from litestar.status_codes import HTTP_204_NO_CONTENT, HTTP_404_NOT_FOUND, HTTP_502_BAD_GATEWAY
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.security import auth_guard, content_auth_guard
from src.api.services.content_proxy import ContentNotFoundError, ContentProxyService
from src.api.services.storage.exceptions import StorageError
from src.api.services.storage.r2 import (
    ALLOWED_CONTENT_TYPES as _STORED_IMAGE_CONTENT_TYPES,
)
from src.api.services.storage.r2 import (
    R2StorageService,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

logger = structlog.get_logger(__name__)

# Types the proxy serves inline. Derived, not mirrored:
#   - stored images are exactly r2.ALLOWED_CONTENT_TYPES (upload normalization
#     guarantees png/jpeg/webp — see services/image_normalization.py),
#   - video outputs add mp4.
# NOTE: routes/storage.py::ALLOWED_CONTENT_TYPES is a DIFFERENT, wider set
# (the client-declared upload gate, incl. heic/heif/avif that are normalized
# to PNG before storage). Do NOT unify with that one.
_INLINE_SAFE_CONTENT_TYPES: frozenset[str] = frozenset(_STORED_IMAGE_CONTENT_TYPES) | {"video/mp4"}


class ContentProxyController(Controller):
    """Auth-gated streaming proxy for R2 content."""

    path = "/v1/content"
    tags: Sequence[str] | None = ("Content",)
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @get("/outputs/{output_id:uuid}", guards=[content_auth_guard])
    async def proxy_output(
        self,
        current_user_id: UUID,
        product_id: str,
        output_id: UUID,
        session: AsyncSession,
        content_proxy: ContentProxyService,
        r2_storage: R2StorageService,
    ) -> Stream | Response[ErrorEnvelope]:
        """Stream a generated output."""
        try:
            storage_key, etag = await content_proxy.resolve_output(
                output_id,
                user_id=current_user_id,
                product_id=product_id,
                session=session,
            )
        except ContentNotFoundError:
            return Response(
                content=ErrorEnvelope(
                    error="not_found",
                    message="Output not found",
                    status_code=HTTP_404_NOT_FOUND,
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

        return await self._stream_from_r2(r2_storage, storage_key, etag, content_proxy.ttl)

    @get("/uploads/{image_id:uuid}", guards=[content_auth_guard])
    async def proxy_upload(
        self,
        current_user_id: UUID,
        product_id: str,
        image_id: UUID,
        session: AsyncSession,
        content_proxy: ContentProxyService,
        r2_storage: R2StorageService,
    ) -> Stream | Response[ErrorEnvelope]:
        """Stream an uploaded image."""
        try:
            storage_key, etag = await content_proxy.resolve_upload(
                image_id,
                user_id=current_user_id,
                product_id=product_id,
                session=session,
            )
        except ContentNotFoundError:
            return Response(
                content=ErrorEnvelope(
                    error="not_found",
                    message="Upload not found",
                    status_code=HTTP_404_NOT_FOUND,
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

        return await self._stream_from_r2(r2_storage, storage_key, etag, content_proxy.ttl)

    @delete("/{content_id:uuid}", status_code=HTTP_204_NO_CONTENT, guards=[auth_guard])
    async def delete_content(
        self,
        current_user_id: UUID,
        product_id: str,
        content_id: UUID,
        session: AsyncSession,
        content_proxy: ContentProxyService,
    ) -> None:
        """Delete a content item (output or upload).

        Permanently removes the file from R2 storage and deletes
        the database record. This action cannot be undone.

        The endpoint accepts any content ID — it checks generation
        outputs first, then user uploads. Ownership and product
        scoping are enforced.

        Returns 404 if the content does not exist, is not owned
        by the caller, or belongs to a different product.
        """
        try:
            await content_proxy.delete_content(
                content_id,
                user_id=current_user_id,
                product_id=product_id,
                session=session,
            )
        except ContentNotFoundError as exc:
            raise NotFoundException(detail="Content not found") from exc

    @staticmethod
    async def _stream_from_r2(
        r2: R2StorageService,
        storage_key: str,
        etag: str,
        cache_ttl: int,
    ) -> Stream | Response[ErrorEnvelope]:
        """Open an R2 stream and return a Litestar Stream response.

        Uses a wrapper generator that keeps the R2 client context open
        while yielding chunks, plus a HEAD request for metadata.
        """
        try:

            async def _streaming_body() -> AsyncIterator[bytes]:
                async with r2.stream_object(storage_key) as (chunks, _, __):
                    async for chunk in chunks:
                        yield chunk

            # HEAD for metadata (content_type, size) — needed before returning Stream
            async with r2._get_client() as client:
                head = await client.head_object(
                    Bucket=r2._settings.bucket_name,
                    Key=storage_key,
                )
            stored_content_type = head.get("ContentType", "application/octet-stream")
            size_bytes = head.get("ContentLength", 0)

            is_inline_safe = stored_content_type in _INLINE_SAFE_CONTENT_TYPES
            media_type = stored_content_type if is_inline_safe else "application/octet-stream"
            disposition = "inline" if is_inline_safe else "attachment"

            return Stream(
                _streaming_body(),
                media_type=media_type,
                headers={
                    "Content-Length": str(size_bytes),
                    "Cache-Control": f"private, max-age={cache_ttl}, immutable",
                    "ETag": f'"{etag}"',
                    "X-Content-Id": etag,
                    "X-Content-Type-Options": "nosniff",
                    "Content-Disposition": disposition,
                },
            )

        except StorageError:
            logger.warning("content_proxy.r2_fetch_failed", storage_key=storage_key)
            return Response(
                content=ErrorEnvelope(
                    error="upstream_error",
                    message="Failed to fetch content",
                    status_code=HTTP_502_BAD_GATEWAY,
                ),
                status_code=HTTP_502_BAD_GATEWAY,
            )
