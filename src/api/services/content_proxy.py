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

        Tries generation_outputs first, then user_images. Deletes the DB
        row(s) and commits *first* — the database is the source of truth for
        "does this content exist" — then best-effort deletes the R2 objects.
        The reverse order (R2 first) risks the DB still pointing at a
        destroyed object if the row delete or commit then fails; deleting
        DB-first means the worst case is an orphaned (harmless, GC-able) R2
        object, never a dangling reference in the gallery.

        Args:
            content_id: UUID of the content to delete.
            user_id: Requesting user (must be owner).
            product_id: Product slug for scoping.
            session: Database session. This method commits it directly (the
                delete must be durable before R2 objects are touched) —
                callers must not expect further work in the same transaction.

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
            # The DB cascade (ON DELETE CASCADE on parent_output_id) removes
            # derivative (thumbnail) rows, but their R2 objects still need
            # explicit cleanup — collect the keys before the row is gone.
            derivatives = await output_repo.list_derivatives(content_id)
            keys = [d.storage_key for d in derivatives]
            keys.append(output.storage_key)
            await output_repo.delete(content_id, user_id=user_id)
            await session.commit()
            await self._delete_storage_keys(keys, content_id=content_id)
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
            upload_derivatives = await image_repo.list_derivatives(content_id)
            keys = [d.storage_key for d in upload_derivatives]
            keys.append(upload.storage_key)
            await image_repo.delete(content_id, user_id=user_id)
            await session.commit()
            await self._delete_storage_keys(keys, content_id=content_id)
            logger.info(
                "content.deleted",
                content_id=str(content_id),
                content_type="upload",
                user_id=str(user_id),
                product_id=product_id,
            )
            return True

        raise ContentNotFoundError(f"Content not found: {content_id}")

    async def _delete_storage_keys(self, keys: list[str], *, content_id: UUID) -> None:
        """Best-effort R2 cleanup after the DB row is already committed gone.

        Each key is attempted independently — one failure logs loudly and
        moves on rather than blocking cleanup of the rest or raising back to
        the caller (the DB delete already succeeded and committed; a
        storage-side failure here must not surface as a failed request).
        """
        for key in keys:
            try:
                await self._storage.delete(key)
            except Exception:
                logger.exception(
                    "content.r2_delete_failed",
                    storage_key=key,
                    content_id=str(content_id),
                )

    @property
    def ttl(self) -> int:
        """Cache-Control max-age value."""
        return self._ttl
