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

from src.db.repositories.storage import StorageRepository

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
        repo = StorageRepository(session)
        output = await repo.get_output(output_id, user_id=user_id)
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
        repo = StorageRepository(session)
        image = await repo.get_user_image(image_id, user_id=user_id)
        if image is None or image.product_id != product_id:
            raise ContentNotFoundError(f"Upload not found: {image_id}")
        return image.storage_key, str(image.id)

    @property
    def ttl(self) -> int:
        """Cache-Control max-age value."""
        return self._ttl
