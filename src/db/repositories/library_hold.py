"""Data access for ``library_asset_holds`` (the library's pin mechanism, S1-P3).

Conventions (the studio repositories' P1-9, applied to the library):

* Writes are Core ``insert/update/delete … RETURNING``; this repository never ``flush()``es or
  ``commit()``s. The caller owns the transaction (the studio calls it inside its own).
* Row locks are explicit and **named by what they are for**:

  ====================  ======================  ==============================================
  method                lock                    why
  ====================  ======================  ==============================================
  ``lock_upload`` /     ``FOR KEY SHARE``       a hold insert takes it anyway through its FK;
  ``lock_output``                               taking it first lets the eligibility checks run
                                                against the *locked* row, and makes the retention
                                                sweep (``FOR UPDATE SKIP LOCKED``) skip the asset
  ``lock_job_shared``   ``FOR SHARE``           pairs with the job soft-delete's ``FOR UPDATE``:
                                                whichever commits first wins
  ``lock_for_release``  ``FOR NO KEY UPDATE``   serializes two releases of one asset's last holds
                                                (compatible with ``KEY SHARE``, so a concurrent
                                                *hold* is not blocked)
  ====================  ======================  ==============================================

  Locking reads return the latest committed row version; a non-locking read would not.
* The two partial unique indexes ``(upload_id|output_id, holder_kind, holder_id)`` serve every
  ``EXISTS (hold)`` lookup by asset, so no further index is created for them.

The library knows a holder only as an opaque ``(holder_kind, holder_id)`` (D12).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from sqlalchemy import delete, exists, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core.uid import new_id
from src.db.models.library import LibraryAssetHold
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["LibraryHoldRepository", "LockedAsset", "LockedJob"]


@dataclass(frozen=True, slots=True)
class LockedAsset:
    """The columns of a locked asset row that the eligibility checks read."""

    is_derivative: bool
    """A thumbnail / poster frame, or a row with a parent: never a library asset of its own."""
    content_type: str
    job_id: UUID | None = None
    """Outputs only: the job the output belongs to (it never changes)."""


@dataclass(frozen=True, slots=True)
class LockedJob:
    """The columns of a locked job row that the eligibility checks read."""

    status: str
    is_deleted: bool


class LibraryHoldRepository:
    """Data access for ``library_asset_holds``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- locking reads (the hold path) ---------------------------------------------------------

    async def lock_upload(
        self, upload_id: UUID, *, user_id: UUID, product_id: str
    ) -> LockedAsset | None:
        """``SELECT … FOR KEY SHARE`` an upload of this owner; ``None`` when there is none.

        Waits behind a retention sweep that holds the row ``FOR UPDATE`` and then finds it gone.
        """
        row = (
            await self._session.execute(
                select(
                    UserImage.is_thumbnail,
                    UserImage.parent_image_id,
                    UserImage.content_type,
                )
                .where(
                    UserImage.id == upload_id,
                    UserImage.user_id == user_id,
                    UserImage.product_id == product_id,
                )
                .with_for_update(read=True, key_share=True)
            )
        ).one_or_none()
        if row is None:
            return None
        return LockedAsset(
            is_derivative=row.is_thumbnail or row.parent_image_id is not None,
            content_type=row.content_type,
        )

    async def lock_output(
        self, output_id: UUID, *, user_id: UUID, product_id: str
    ) -> LockedAsset | None:
        """``SELECT … FOR KEY SHARE`` an output of this owner; ``None`` when there is none."""
        row = (
            await self._session.execute(
                select(
                    GenerationOutput.is_thumbnail,
                    GenerationOutput.parent_output_id,
                    GenerationOutput.content_type,
                    GenerationOutput.job_id,
                )
                .where(
                    GenerationOutput.id == output_id,
                    GenerationOutput.user_id == user_id,
                    GenerationOutput.product_id == product_id,
                )
                .with_for_update(read=True, key_share=True)
            )
        ).one_or_none()
        if row is None:
            return None
        return LockedAsset(
            is_derivative=row.is_thumbnail or row.parent_output_id is not None,
            content_type=row.content_type,
            job_id=row.job_id,
        )

    async def lock_job_shared(
        self, job_id: UUID, *, user_id: UUID, product_id: str
    ) -> LockedJob | None:
        """``SELECT … FOR SHARE`` a job of this owner and return its status under that lock.

        Pairs with the job soft-delete's ``FOR UPDATE`` (``JobRepository.soft_delete``): a hold
        and a soft-delete of the same job cannot both pass their checks.
        """
        row = (
            await self._session.execute(
                select(GenerationJob.status, GenerationJob.is_deleted)
                .where(
                    GenerationJob.id == job_id,
                    GenerationJob.user_id == user_id,
                    GenerationJob.product_id == product_id,
                )
                .with_for_update(read=True)
            )
        ).one_or_none()
        if row is None:
            return None
        return LockedJob(status=str(row.status), is_deleted=row.is_deleted)

    # -- writes --------------------------------------------------------------------------------

    async def insert(
        self,
        *,
        user_id: UUID,
        product_id: str,
        upload_id: UUID | None,
        output_id: UUID | None,
        holder_kind: str,
        holder_id: UUID,
    ) -> UUID:
        """Insert a hold row and return its id.

        Raises:
            sqlalchemy.exc.IntegrityError: The asset is gone (``fk_library_asset_holds_*``), or
                this holder already holds it (a ``uq_library_asset_holds_*`` index); call inside
                ``session.begin_nested()``.
        """
        result = await self._session.execute(
            pg_insert(LibraryAssetHold)
            .values(
                id=new_id(),
                user_id=user_id,
                product_id=product_id,
                upload_id=upload_id,
                output_id=output_id,
                holder_kind=holder_kind,
                holder_id=holder_id,
            )
            .returning(LibraryAssetHold.id)
        )
        hold_id: UUID = result.scalar_one()
        return hold_id

    async def assets_of_holds(
        self, hold_ids: Collection[UUID]
    ) -> Sequence[tuple[UUID, UUID | None, UUID | None]]:
        """``(hold_id, upload_id, output_id)`` of the holds that exist (unknown ids are absent)."""
        if not hold_ids:
            return []
        result = await self._session.execute(
            select(LibraryAssetHold.id, LibraryAssetHold.upload_id, LibraryAssetHold.output_id)
            .where(LibraryAssetHold.id.in_(hold_ids))
            .order_by(LibraryAssetHold.id)
        )
        return [(row.id, row.upload_id, row.output_id) for row in result.all()]

    async def lock_for_release(
        self, *, upload_ids: Collection[UUID], output_ids: Collection[UUID]
    ) -> None:
        """``FOR NO KEY UPDATE`` the given assets, **uploads then outputs, each in ascending id**.

        One fixed order, so two releases that share assets cannot deadlock. It serializes two
        concurrent releases of the last two holds of one asset: the second one reads the first
        one's committed delete and re-arms the asset (without this neither would).
        """
        if upload_ids:
            await self._session.execute(
                select(UserImage.id)
                .where(UserImage.id.in_(upload_ids))
                .order_by(UserImage.id)
                .with_for_update(key_share=True)
            )
        if output_ids:
            await self._session.execute(
                select(GenerationOutput.id)
                .where(GenerationOutput.id.in_(output_ids))
                .order_by(GenerationOutput.id)
                .with_for_update(key_share=True)
            )

    async def delete(self, hold_ids: Collection[UUID]) -> int:
        """Delete the holds; return how many existed."""
        if not hold_ids:
            return 0
        result = await self._session.execute(
            delete(LibraryAssetHold)
            .where(LibraryAssetHold.id.in_(hold_ids))
            .returning(LibraryAssetHold.id)
        )
        return len(result.all())

    async def rearm_unheld(
        self,
        *,
        upload_ids: Collection[UUID],
        output_ids: Collection[UUID],
        retention_days: int,
    ) -> None:
        """Restart the retention clock of every given asset that no longer has a hold.

        ``expires_at = greatest(expires_at, now() + retention_days)``: never earlier than before
        (an asset that was fresh stays fresh), one statement per table.
        """
        horizon = func.now() + timedelta(days=retention_days)
        if upload_ids:
            await self._session.execute(
                update(UserImage)
                .where(
                    UserImage.id.in_(upload_ids),
                    ~exists().where(LibraryAssetHold.upload_id == UserImage.id),
                )
                .values(expires_at=func.greatest(UserImage.expires_at, horizon))
                .execution_options(synchronize_session=False)
            )
        if output_ids:
            await self._session.execute(
                update(GenerationOutput)
                .where(
                    GenerationOutput.id.in_(output_ids),
                    ~exists().where(LibraryAssetHold.output_id == GenerationOutput.id),
                )
                .values(expires_at=func.greatest(GenerationOutput.expires_at, horizon))
                .execution_options(synchronize_session=False)
            )

    # -- reads ---------------------------------------------------------------------------------

    async def held_upload_ids(
        self, upload_ids: Collection[UUID], *, user_id: UUID, product_id: str
    ) -> set[UUID]:
        """The uploads among ``upload_ids`` (of this owner) that have at least one hold."""
        if not upload_ids:
            return set()
        result = await self._session.execute(
            select(LibraryAssetHold.upload_id)
            .where(
                LibraryAssetHold.upload_id.in_(upload_ids),
                LibraryAssetHold.user_id == user_id,
                LibraryAssetHold.product_id == product_id,
            )
            .distinct()
        )
        return {upload for upload in result.scalars().all() if upload is not None}

    async def held_output_ids(
        self, output_ids: Collection[UUID], *, user_id: UUID, product_id: str
    ) -> set[UUID]:
        """The outputs among ``output_ids`` (of this owner) that have at least one hold."""
        if not output_ids:
            return set()
        result = await self._session.execute(
            select(LibraryAssetHold.output_id)
            .where(
                LibraryAssetHold.output_id.in_(output_ids),
                LibraryAssetHold.user_id == user_id,
                LibraryAssetHold.product_id == product_id,
            )
            .distinct()
        )
        return {output for output in result.scalars().all() if output is not None}

    async def holder_counts_for_upload(self, upload_id: UUID) -> Sequence[tuple[str, int]]:
        """``(holder_kind, count)`` of an upload's holds, ordered by kind."""
        result = await self._session.execute(
            select(LibraryAssetHold.holder_kind, func.count())
            .where(LibraryAssetHold.upload_id == upload_id)
            .group_by(LibraryAssetHold.holder_kind)
            .order_by(LibraryAssetHold.holder_kind)
        )
        return [(row[0], row[1]) for row in result.all()]

    async def holder_counts_for_output(self, output_id: UUID) -> Sequence[tuple[str, int]]:
        """``(holder_kind, count)`` of an output's holds, ordered by kind."""
        result = await self._session.execute(
            select(LibraryAssetHold.holder_kind, func.count())
            .where(LibraryAssetHold.output_id == output_id)
            .group_by(LibraryAssetHold.holder_kind)
            .order_by(LibraryAssetHold.holder_kind)
        )
        return [(row[0], row[1]) for row in result.all()]

    async def holder_counts_for_job(self, job_id: UUID) -> Sequence[tuple[str, int]]:
        """``(holder_kind, count)`` summed over the holds of every output of a job."""
        result = await self._session.execute(
            select(LibraryAssetHold.holder_kind, func.count())
            .join(GenerationOutput, GenerationOutput.id == LibraryAssetHold.output_id)
            .where(GenerationOutput.job_id == job_id)
            .group_by(LibraryAssetHold.holder_kind)
            .order_by(LibraryAssetHold.holder_kind)
        )
        return [(row[0], row[1]) for row in result.all()]

    async def job_has_hold(self, job_id: UUID) -> bool:
        """Whether any output of the job is held (one ``EXISTS``)."""
        result = await self._session.execute(
            select(
                exists().where(
                    LibraryAssetHold.output_id == GenerationOutput.id,
                    GenerationOutput.job_id == job_id,
                )
            )
        )
        return bool(result.scalar_one())
