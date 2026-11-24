"""Data access for ``library_audio_profiles`` (S2a, library-owned).

Conventions of the other library repositories: Core ``insert … RETURNING``, no ``flush()`` and no
``commit()`` (the caller owns the transaction: the upload writes the asset row and its profile in
one), every read filters the asset's owner and product so a foreign or cross-product asset is
simply "not found".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core.enums import JobStatus
from src.core.library_ref import AssetRef, LibraryAssetSource
from src.core.uid import new_id
from src.db.models.library import LibraryAudioProfile
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["AudioPeaksRow", "LibraryAudioProfileRepository"]


@dataclass(frozen=True, slots=True)
class AudioPeaksRow:
    """One asset's content type and (when it is audio) its stored peaks."""

    content_type: str
    peaks: bytes | None
    peaks_version: int | None


class LibraryAudioProfileRepository:
    """Data access for ``library_audio_profiles``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert(
        self,
        *,
        upload_id: UUID | None = None,
        output_id: UUID | None = None,
        codec: str,
        sample_rate_hz: int,
        channels: int,
        bit_rate: int | None,
        peaks: bytes,
        peaks_version: int,
    ) -> UUID:
        """Insert the profile of exactly one asset (a CHECK refuses neither or both)."""
        result = await self._session.execute(
            pg_insert(LibraryAudioProfile)
            .values(
                id=new_id(),
                upload_id=upload_id,
                output_id=output_id,
                codec=codec,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                bit_rate=bit_rate,
                peaks=peaks,
                peaks_version=peaks_version,
            )
            .returning(LibraryAudioProfile.id)
        )
        return result.scalar_one()

    async def get_peaks(
        self, ref: AssetRef, *, user_id: UUID, product_id: str
    ) -> AudioPeaksRow | None:
        """The asset's content type and peaks in **one** SELECT; ``None`` = missing / not visible.

        Visible means what the library list shows: an owned, same-product, non-derivative asset
        (an output also needs a completed, not soft-deleted job). A non-audio asset comes back
        with ``peaks = None`` so the caller can answer ``422`` rather than ``404``.
        """
        if ref.source is LibraryAssetSource.UPLOAD:
            statement = (
                select(
                    UserImage.content_type,
                    LibraryAudioProfile.peaks,
                    LibraryAudioProfile.peaks_version,
                )
                .select_from(UserImage)
                .outerjoin(LibraryAudioProfile, LibraryAudioProfile.upload_id == UserImage.id)
                .where(
                    UserImage.id == ref.asset_id,
                    UserImage.user_id == user_id,
                    UserImage.product_id == product_id,
                    UserImage.is_thumbnail.is_(False),
                )
            )
        else:
            statement = (
                select(
                    GenerationOutput.content_type,
                    LibraryAudioProfile.peaks,
                    LibraryAudioProfile.peaks_version,
                )
                .select_from(GenerationOutput)
                .join(GenerationJob, GenerationJob.id == GenerationOutput.job_id)
                .outerjoin(
                    LibraryAudioProfile, LibraryAudioProfile.output_id == GenerationOutput.id
                )
                .where(
                    GenerationOutput.id == ref.asset_id,
                    GenerationOutput.user_id == user_id,
                    GenerationOutput.product_id == product_id,
                    GenerationOutput.is_thumbnail.is_(False),
                    GenerationJob.status == JobStatus.COMPLETED,
                    GenerationJob.is_deleted.is_(False),
                )
            )
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None
        return AudioPeaksRow(
            content_type=row.content_type, peaks=row.peaks, peaks_version=row.peaks_version
        )
