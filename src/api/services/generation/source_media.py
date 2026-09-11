"""Resolution for ordered owned-library generation inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from src.core.enums import MediaKind, media_kind_from_content_type
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref, parse_asset_ref
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.unified_generation import SourceMediaReference
    from src.db.models.storage import GenerationOutput, UserImage

logger = structlog.get_logger(__name__)


class SourceMediaValidationError(ValueError):
    """A safe, client-actionable validation failure for ``source_media``."""


@dataclass(frozen=True, slots=True)
class ResolvedSourceMedia:
    """An ownership-checked source asset, retained in request order."""

    position: int
    ref: AssetRef
    asset_ref: str
    media_kind: MediaKind
    content_type: str
    storage_key: str
    size_bytes: int
    job_id: UUID | None


class SourceMediaResolver:
    """Resolve request references through the two owned-library tables."""

    async def resolve(
        self,
        refs: list[SourceMediaReference],
        *,
        user_id: UUID,
        session: AsyncSession,
        product_id: str | None = None,
    ) -> list[ResolvedSourceMedia]:
        """Return ownership-checked input assets in exactly request order.

        References are parsed before querying, then uploads and outputs are
        fetched in two batched ownership-scoped queries.  Missing, foreign,
        deleted-from-product, and thumbnail records intentionally map to the
        same public response, avoiding an existence oracle.
        """
        parsed: list[AssetRef] = []
        seen: set[AssetRef] = set()
        for position, item in enumerate(refs):
            try:
                ref = parse_asset_ref(item.asset_ref)
            except ValueError as exc:
                raise SourceMediaValidationError(
                    f"source_media position {position} has an invalid asset reference"
                ) from exc
            if ref in seen:
                raise SourceMediaValidationError(
                    f"source_media position {position} duplicates an earlier reference"
                )
            seen.add(ref)
            parsed.append(ref)

        upload_ids = [ref.asset_id for ref in parsed if ref.source is LibraryAssetSource.UPLOAD]
        output_ids = [ref.asset_id for ref in parsed if ref.source is LibraryAssetSource.OUTPUT]
        uploads = await UserImageRepository(session).get_many(upload_ids, user_id=user_id)
        outputs = await OutputRepository(session).get_many(output_ids, user_id=user_id)

        resolved: list[ResolvedSourceMedia] = []
        for position, ref in enumerate(parsed):
            row: UserImage | GenerationOutput | None
            if ref.source is LibraryAssetSource.UPLOAD:
                row = uploads.get(ref.asset_id)
                if (
                    row is None
                    or row.is_thumbnail
                    or (product_id is not None and row.product_id != product_id)
                ):
                    raise SourceMediaValidationError(
                        f"source_media position {position} does not name an available asset"
                    )
                job_id = None
            else:
                row = outputs.get(ref.asset_id)
                if (
                    row is None
                    or row.is_thumbnail
                    or (product_id is not None and row.product_id != product_id)
                ):
                    raise SourceMediaValidationError(
                        f"source_media position {position} does not name an available asset"
                    )
                job_id = row.job_id

            try:
                media_kind = media_kind_from_content_type(row.content_type)
            except ValueError as exc:
                raise SourceMediaValidationError(
                    f"source_media position {position} does not name a supported media asset"
                ) from exc

            resolved.append(
                ResolvedSourceMedia(
                    position=position,
                    ref=ref,
                    asset_ref=format_asset_ref(ref.source, ref.asset_id),
                    media_kind=media_kind,
                    content_type=row.content_type,
                    storage_key=row.storage_key,
                    size_bytes=row.size_bytes,
                    job_id=job_id,
                )
            )

        logger.info(
            "generation.source_media.resolved",
            user_id=str(user_id),
            count=len(resolved),
            positions=[source.position for source in resolved],
        )
        return resolved
