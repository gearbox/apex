"""Data access for ``library_audio_profiles`` (S2a, library-owned).

Conventions of the other library repositories: Core ``insert … RETURNING``, no ``flush()`` and no
``commit()`` (the caller owns the transaction: the upload writes the asset row and its profile in
one), every read filters the asset's owner and product so a foreign or cross-product asset is
simply "not found".

A detail-peaks window is sliced **in SQL** (``substring``), in the same statement that checks the
asset is visible: the column is ``STORAGE EXTERNAL``, so a window reads only its own chunks.

The two backfill methods (``list_missing_detail_peaks`` / ``fill_detail_peaks``) are the
exception to the owner filter: an operator's maintenance job over every user's profiles. No
request path calls them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import LargeBinary, func, literal, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core.enums import JobStatus
from src.core.library_ref import AssetRef, LibraryAssetSource
from src.core.uid import new_id
from src.db.models.library import LibraryAudioProfile
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from typing import Any
    from uuid import UUID

    from sqlalchemy import ColumnElement, Select
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import InstrumentedAttribute

__all__ = [
    "AudioDetailPeaksRow",
    "AudioPeaksRow",
    "LibraryAudioProfileRepository",
    "MissingDetailPeaksRow",
]


@dataclass(frozen=True, slots=True)
class AudioPeaksRow:
    """One asset's content type and (when it is audio) its stored peaks."""

    content_type: str
    peaks: bytes | None
    peaks_version: int | None


@dataclass(frozen=True, slots=True)
class AudioDetailPeaksRow:
    """One asset's content type and a window of its detail peaks."""

    content_type: str
    profile_id: UUID | None
    """``None`` when the asset has no audio profile."""
    detail_version: int | None
    """``None`` (like the two below) when the detail level has not been computed yet."""
    total_bytes: int | None
    window: bytes | None


@dataclass(frozen=True, slots=True)
class MissingDetailPeaksRow:
    """A profile without detail peaks, and the stored object to compute them from."""

    profile_id: UUID
    created_at: datetime
    storage_key: str


def _visible_asset(
    ref: AssetRef,
    *columns: InstrumentedAttribute[Any] | ColumnElement[Any],
    user_id: UUID,
    product_id: str,
) -> Select[*tuple[Any, ...]]:
    """The asset's content type and ``columns`` of its profile (outer join), if it is **visible**:
    what the library list shows - owned, same-product and not a derivative (an output also needs a
    completed, not soft-deleted job)."""
    if ref.source is LibraryAssetSource.UPLOAD:
        return (
            select(UserImage.content_type, *columns)
            .select_from(UserImage)
            .outerjoin(LibraryAudioProfile, LibraryAudioProfile.upload_id == UserImage.id)
            .where(
                UserImage.id == ref.asset_id,
                UserImage.user_id == user_id,
                UserImage.product_id == product_id,
                UserImage.is_thumbnail.is_(False),
            )
        )
    return (
        select(GenerationOutput.content_type, *columns)
        .select_from(GenerationOutput)
        .join(GenerationJob, GenerationJob.id == GenerationOutput.job_id)
        .outerjoin(LibraryAudioProfile, LibraryAudioProfile.output_id == GenerationOutput.id)
        .where(
            GenerationOutput.id == ref.asset_id,
            GenerationOutput.user_id == user_id,
            GenerationOutput.product_id == product_id,
            GenerationOutput.is_thumbnail.is_(False),
            GenerationJob.status == JobStatus.COMPLETED,
            GenerationJob.is_deleted.is_(False),
        )
    )


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
        detail_peaks: bytes,
        detail_peaks_version: int,
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
                detail_peaks=detail_peaks,
                detail_peaks_version=detail_peaks_version,
            )
            .returning(LibraryAudioProfile.id)
        )
        return result.scalar_one()

    async def get_peaks(
        self, ref: AssetRef, *, user_id: UUID, product_id: str
    ) -> AudioPeaksRow | None:
        """The asset's content type and peaks in **one** SELECT; ``None`` = missing / not visible.

        A non-audio asset comes back with ``peaks = None`` so the caller can answer ``422``
        rather than ``404``.
        """
        statement = _visible_asset(
            ref,
            LibraryAudioProfile.peaks,
            LibraryAudioProfile.peaks_version,
            user_id=user_id,
            product_id=product_id,
        )
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None
        return AudioPeaksRow(
            content_type=row.content_type, peaks=row.peaks, peaks_version=row.peaks_version
        )

    async def get_detail_peaks_window(
        self,
        ref: AssetRef,
        *,
        user_id: UUID,
        product_id: str,
        first_bucket: int,
        bucket_count: int,
    ) -> AudioDetailPeaksRow | None:
        """``bucket_count`` detail buckets from ``first_bucket``, and the total size, in **one**
        SELECT; ``None`` = missing / not visible.

        ``substring`` stops at the end of the value, so a window reaching past it comes back
        shorter (empty when it starts there). The caller keeps both numbers within the largest
        possible bucket count.
        """
        statement = _visible_asset(
            ref,
            LibraryAudioProfile.id.label("profile_id"),
            LibraryAudioProfile.detail_peaks_version,
            func.octet_length(LibraryAudioProfile.detail_peaks).label("total_bytes"),
            # Two bytes per bucket; SQL's substring counts from 1.
            func.substring(
                LibraryAudioProfile.detail_peaks,
                2 * first_bucket + 1,
                2 * bucket_count,
                type_=LargeBinary,
            ).label("window"),
            user_id=user_id,
            product_id=product_id,
        )
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None
        return AudioDetailPeaksRow(
            content_type=row.content_type,
            profile_id=row.profile_id,
            detail_version=row.detail_peaks_version,
            total_bytes=row.total_bytes,
            window=row.window,
        )

    # -- backfill (operator maintenance, every owner) ------------------------------------------

    async def list_missing_detail_peaks(
        self, *, limit: int, after: tuple[datetime, UUID] | None = None
    ) -> Sequence[MissingDetailPeaksRow]:
        """Up to ``limit`` profiles with no detail peaks, oldest first, after the keyset
        position ``(created_at, id)``. A profile has exactly one asset (CHECK), which names its
        storage key."""
        statement = (
            select(
                LibraryAudioProfile.id,
                LibraryAudioProfile.created_at,
                func.coalesce(UserImage.storage_key, GenerationOutput.storage_key).label(
                    "storage_key"
                ),
            )
            .outerjoin(UserImage, UserImage.id == LibraryAudioProfile.upload_id)
            .outerjoin(GenerationOutput, GenerationOutput.id == LibraryAudioProfile.output_id)
            .where(LibraryAudioProfile.detail_peaks.is_(None))
        )
        if after is not None:
            statement = statement.where(
                tuple_(LibraryAudioProfile.created_at, LibraryAudioProfile.id)
                > tuple_(literal(after[0]), literal(after[1]))
            )
        result = await self._session.execute(
            statement.order_by(LibraryAudioProfile.created_at, LibraryAudioProfile.id).limit(limit)
        )
        return [
            MissingDetailPeaksRow(
                profile_id=row.id, created_at=row.created_at, storage_key=row.storage_key
            )
            for row in result.all()
        ]

    async def fill_detail_peaks(
        self, profile_id: UUID, *, detail_peaks: bytes, detail_peaks_version: int
    ) -> bool:
        """Write the detail level **only if it is still missing**; the overview is not touched.

        Returns:
            ``False`` when another run filled it first (or the profile is gone).
        """
        result = await self._session.execute(
            update(LibraryAudioProfile)
            .where(
                LibraryAudioProfile.id == profile_id,
                LibraryAudioProfile.detail_peaks.is_(None),
            )
            .values(detail_peaks=detail_peaks, detail_peaks_version=detail_peaks_version)
            .returning(LibraryAudioProfile.id)
            .execution_options(synchronize_session=False)
        )
        return result.scalar_one_or_none() is not None
