"""Auth-gated streaming proxy for R2 content.

Resolves DB record → ownership check → R2 stream.
Designed for future migration to CF Workers + Signed Cookies:
when that happens, this service returns a redirect URL to the Worker
instead of streaming bytes, with zero schema changes needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.storage.r2 import R2StorageService
    from src.core.config import Settings

logger = structlog.get_logger(__name__)


class ContentNotFoundError(Exception):
    """Content not found or not owned by the requesting user."""


class ContentFetchError(Exception):
    """Failed to fetch content from storage backend."""


class ContentProxyService:
    """Auth-gated streaming proxy for R2 content."""

    def __init__(self, storage: R2StorageService, settings: Settings) -> None:
        self._storage = storage
        self._ttl = settings.content_url_ttl

    async def resolve_output(
        self,
        output_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        session: AsyncSession,
    ) -> tuple[str, str]:
        """Resolve an output ID to its storage_key after ownership check.

        Args:
            output_id: ID of the generation output.
            user_id: Requesting user ID.
            product_id: Product slug for product-scoped check.
            session: Database session.

        Returns:
            (storage_key, etag)

        Raises:
            ContentNotFoundError: Output not found or not owned.
        """
        output_repo = OutputRepository(session)
        output = await output_repo.get(output_id, user_id=user_id)
        if output is None or output.product_id != product_id:
            raise ContentNotFoundError(f"Output not found: {output_id}")
        return output.storage_key, str(output.id)

    async def resolve_upload(
        self,
        image_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        session: AsyncSession,
    ) -> tuple[str, str]:
        """Resolve an upload ID to its storage_key after ownership check.

        Args:
            image_id: ID of the user image upload.
            user_id: Requesting user ID.
            product_id: Product slug for product-scoped check.
            session: Database session.

        Returns:
            (storage_key, etag)

        Raises:
            ContentNotFoundError: Upload not found or not owned.
        """
        image_repo = UserImageRepository(session)
        image = await image_repo.get(image_id, user_id=user_id)
        if image is None or image.product_id != product_id:
            raise ContentNotFoundError(f"Upload not found: {image_id}")
        return image.storage_key, str(image.id)

    async def delete_content(
        self,
        content_id: UUID,
        *,
        user_id: UUID,
        product_id: str,
        session: AsyncSession,
    ) -> bool:
        """Delete a content item (output or upload) by ID.

        Tries generation_outputs first, then user_images.
        Deletes R2 object first, then DB row.

        Args:
            content_id: UUID of the content to delete.
            user_id: Requesting user (must be owner).
            product_id: Product slug for scoping.
            session: Database session.

        Returns:
            True if content was found and deleted.

        Raises:
            ContentNotFoundError: Content not found or not owned.
        """
        output_repo = OutputRepository(session)
        image_repo = UserImageRepository(session)

        # Try output first (more common deletion target)
        output = await output_repo.get(content_id, user_id=user_id)
        if output is not None and output.product_id == product_id:
            # Delete derivative (thumbnail) R2 objects before removing the parent row.
            # The DB cascade (ON DELETE CASCADE on parent_output_id) removes derivative
            # rows, but we must explicitly clean up their R2 objects.
            derivatives = await output_repo.list_derivatives(content_id)
            for derivative in derivatives:
                await self._storage.delete(derivative.storage_key)
            await self._storage.delete(output.storage_key)
            await output_repo.delete(content_id, user_id=user_id)
            logger.info(
                "content.deleted",
                content_id=str(content_id),
                content_type="output",
                user_id=str(user_id),
                product_id=product_id,
            )
            return True

        # Try upload
        upload = await image_repo.get(content_id, user_id=user_id)
        if upload is not None and upload.product_id == product_id:
            await self._storage.delete(upload.storage_key)
            await image_repo.delete(content_id, user_id=user_id)
            logger.info(
                "content.deleted",
                content_id=str(content_id),
                content_type="upload",
                user_id=str(user_id),
                product_id=product_id,
            )
            return True

        raise ContentNotFoundError(f"Content not found: {content_id}")

    @property
    def ttl(self) -> int:
        """Cache-Control max-age value."""
        return self._ttl
