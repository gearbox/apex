"""Content proxy — auth-gated streaming access to R2 objects.

Provides stable, non-expiring URLs for user content. The server
resolves ownership, then streams bytes from R2 with Cache-Control
headers. Presigned URLs are never exposed to the client.

Endpoints:
  GET /v1/content/outputs/{output_id}  — stream a generated output
  GET /v1/content/uploads/{image_id}   — stream an uploaded image
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from uuid import UUID

import structlog
from litestar import Controller, Response, get
from litestar.di import Provide
from litestar.response import Stream
from litestar.status_codes import HTTP_404_NOT_FOUND, HTTP_502_BAD_GATEWAY
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.security import auth_guard
from src.api.services.content_proxy import ContentNotFoundError, ContentProxyService
from src.api.services.storage.exceptions import StorageError
from src.api.services.storage.r2 import R2StorageService

logger = structlog.get_logger(__name__)


class ContentProxyController(Controller):
    """Auth-gated streaming proxy for R2 content."""

    path = "/v1/content"
    tags: Sequence[str] | None = ["Content"]
    guards = [auth_guard]
    dependencies = {"current_user_id": Provide(get_current_user_id)}

    @get("/outputs/{output_id:uuid}")
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

    @get("/uploads/{image_id:uuid}")
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
            content_type = head.get("ContentType", "application/octet-stream")
            size_bytes = head.get("ContentLength", 0)

            return Stream(
                _streaming_body(),
                media_type=content_type,
                headers={
                    "Content-Length": str(size_bytes),
                    "Cache-Control": f"private, max-age={cache_ttl}, immutable",
                    "ETag": f'"{etag}"',
                    "X-Content-Id": etag,
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
