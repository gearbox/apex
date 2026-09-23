"""Narrow session-scoped registration of prepared-original PDQ hashes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.enums import MediaFormat, MediaHashMediaType, MediaHashSourceKind
from src.core.media_hash import HashSet  # noqa: TC001 - validated at runtime by callers
from src.core.uid import new_id
from src.db.models.media_hash import MediaHash
from src.db.repositories.media_hash import MediaHashRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.storage import GenerationOutput, UserImage


class MediaHashLedger:
    """Register one original row's immutable hash set in the caller's UoW."""

    def __init__(self, session: AsyncSession) -> None:
        self._repository = MediaHashRepository(session)

    async def register_output(self, output: GenerationOutput, hash_set: HashSet) -> None:
        if output.is_thumbnail is True:
            raise ValueError("thumbnails must never have ledger hashes")
        self._stage(
            hash_set=hash_set,
            product_id=output.product_id,
            user_id=output.user_id,
            job_id=output.job_id,
            source_kind=MediaHashSourceKind.OUTPUT,
            source_id=output.id,
            media_type=self._media_type(output.format),
        )

    async def register_upload(
        self, upload: UserImage, hash_set: HashSet, *, job_id: UUID | None = None
    ) -> None:
        if upload.is_thumbnail is True:
            raise ValueError("thumbnails must never have ledger hashes")
        self._stage(
            hash_set=hash_set,
            product_id=upload.product_id,
            user_id=upload.user_id,
            job_id=job_id,
            source_kind=MediaHashSourceKind.UPLOAD,
            source_id=upload.id,
            media_type=self._media_type(upload.format),
        )

    @staticmethod
    def _media_type(media_format: str) -> MediaHashMediaType:
        try:
            return (
                MediaHashMediaType.VIDEO
                if MediaFormat(media_format).is_video
                else MediaHashMediaType.IMAGE
            )
        except (TypeError, ValueError):
            # A row supplied by a narrow mock can omit the format, but real
            # repositories always persist one of MediaFormat's values.
            return MediaHashMediaType.IMAGE

    def _stage(
        self,
        *,
        hash_set: HashSet,
        product_id: str,
        user_id: UUID,
        job_id: UUID | None,
        source_kind: MediaHashSourceKind,
        source_id: UUID,
        media_type: MediaHashMediaType,
    ) -> None:
        rows = [
            MediaHash(
                id=new_id(),
                product_id=product_id,
                user_id=user_id,
                job_id=job_id,
                source_kind=source_kind,
                source_id=source_id,
                source_media_type=media_type,
                hash_profile=hash_set.profile_id,
                sampling_profile=hash_set.sampling_profile,
                sample_index=sample.sample_index,
                frame_timestamp_ms=sample.frame_timestamp_ms,
                pdq=sample.pdq.bits,
                pdq_quality=sample.pdq.quality,
            )
            for sample in hash_set.samples
        ]
        self._repository.add_all(rows)
