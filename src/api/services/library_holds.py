"""Asset holds: the library's pin mechanism (S1-P3).

Any bounded context may *hold* a library asset. A held asset

* is never swept by the content-retention worker (``ContentRetentionService`` skips it);
* cannot be deleted by any path (the database refuses: ``ON DELETE NO ACTION`` foreign keys, see
  ``src/db/models/library.py::LibraryAssetHold``; the services turn that into a typed ``409``);
* gets its retention clock restarted when its **last** hold is released.

The library knows a holder only as an opaque :class:`AssetHolder` ``(kind, id)``: ``kind`` is a
namespaced string (``"studio.cast_reference"``) and this module never imports the context behind
it (D12). A context calls :class:`AssetHoldService` **inside its own transaction**: the service
never commits and never flushes, like the studio services.

Locks, in the order they are taken (every lock is also named at its call in
``LibraryHoldRepository``):

* ``hold`` of an **upload**: the upload ``FOR KEY SHARE``, the checks, the insert.
* ``hold`` of an **output**: the output ``FOR KEY SHARE``, then its job ``FOR SHARE`` (and the job's
  status is re-read **under that lock**), the checks, the insert. The job lock is what makes the job
  soft-delete safe (``JobRepository.soft_delete`` locks the job ``FOR UPDATE`` and only then looks
  for holds): whichever commits first wins and the loser gets a typed error.
  The output is locked **before** the job (the work order said job first): the retention sweep
  locks an output ``FOR UPDATE`` and then ``UPDATE``s its job, so job-then-output would deadlock
  with it on a single-output job (hold: job-shared, waiting for the output; sweep: output, waiting
  for the job). Output-first is the sweep's own order.
* ``release``: the released assets ``FOR NO KEY UPDATE`` (uploads, then outputs, ascending id), the
  delete, the re-arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy.exc import IntegrityError

from src.core.enums import JobStatus, MediaKind, UnknownMediaTypeError, media_kind_from_content_type
from src.core.library_ref import AssetRef, LibraryAssetSource
from src.db.integrity import violated_constraint
from src.db.models.library import HOLD_ASSET_FKS, HOLD_UNIQUES
from src.db.repositories.library_hold import LibraryHoldRepository

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "AssetHeldError",
    "AssetHoldService",
    "AssetHolder",
    "AssetIneligibleReason",
    "AssetNotEligibleError",
    "AssetNotFoundError",
    "HolderCount",
    "holder_counts_for_job",
    "holder_counts_for_ref",
]

_ALL_CONSTRAINTS = HOLD_ASSET_FKS | HOLD_UNIQUES


@dataclass(frozen=True, slots=True)
class AssetHolder:
    """Who holds an asset: an opaque, namespaced kind and the holder's own id."""

    kind: str
    """Namespaced, e.g. ``"studio.cast_reference"``; at most 48 characters."""
    id: UUID


@dataclass(frozen=True, slots=True)
class HolderCount:
    """How many holds of one kind an asset has (the ``409 asset_held`` detail)."""

    kind: str
    count: int


class AssetIneligibleReason(StrEnum):
    """Why an asset cannot be held (the ``reason`` of ``422 asset_not_eligible``)."""

    DERIVATIVE = "derivative"
    """A thumbnail or another derivative of a library asset."""
    MEDIA_KIND_NOT_ACCEPTED = "media_kind_not_accepted"
    """Its media kind is not one the holder declared (``accept``): a cast reference pins images
    only, a track's audio version pins audio only."""
    JOB_NOT_AVAILABLE = "job_not_available"
    """An output whose job is not ``COMPLETED``, or was soft-deleted."""


class AssetNotFoundError(Exception):
    """The asset is missing, or owned by another user or another product (all the same). → 404"""


class AssetNotEligibleError(Exception):
    """The asset exists but cannot be held. → 422 ``asset_not_eligible``"""

    def __init__(self, reason: AssetIneligibleReason) -> None:
        self.reason = reason
        super().__init__(f"The asset cannot be held: {reason.value}")


class AssetHeldError(Exception):
    """The asset (or one of a job's outputs) is held, so it cannot be deleted. → 409 ``asset_held``

    ``holders`` carries kinds and counts only: never names or ids of the holders (to show names
    the client asks the holding context, e.g. ``GET /v1/studio/cast?holding=<asset_ref>``).
    """

    def __init__(self, holders: Sequence[HolderCount]) -> None:
        self.holders = tuple(holders)
        super().__init__("The asset is held and cannot be deleted")


def _counts(rows: Sequence[tuple[str, int]]) -> tuple[HolderCount, ...]:
    return tuple(HolderCount(kind=kind, count=count) for kind, count in rows)


async def holder_counts_for_ref(session: AsyncSession, ref: AssetRef) -> tuple[HolderCount, ...]:
    """The holders of one asset grouped by kind, ordered by kind (empty: not held).

    Read-only, so a caller that never releases (``LibraryService.delete_asset``) needs no
    ``retention_days``. The asset's ownership is the caller's job (it has already checked it).
    """
    repo = LibraryHoldRepository(session)
    if ref.source == LibraryAssetSource.UPLOAD:
        return _counts(await repo.holder_counts_for_upload(ref.asset_id))
    return _counts(await repo.holder_counts_for_output(ref.asset_id))


async def holder_counts_for_job(session: AsyncSession, job_id: UUID) -> tuple[HolderCount, ...]:
    """The holders of every output of a job, summed by kind and ordered by kind."""
    return _counts(await LibraryHoldRepository(session).holder_counts_for_job(job_id))


class AssetHoldService:
    """Hold and release library assets on behalf of any bounded context."""

    def __init__(self, session: AsyncSession, *, retention_days: int) -> None:
        """Create the service.

        Args:
            session: The caller's session; the service never commits or flushes it.
            retention_days: ``Settings.retention_days``, wired by DI (the service reads no
                settings): the clock a released asset's expiry is restarted to.
        """
        self._session = session
        self._repo = LibraryHoldRepository(session)
        self._retention_days = retention_days

    async def hold(
        self,
        ref: AssetRef,
        *,
        holder: AssetHolder,
        user_id: UUID,
        product_id: str,
        accept: frozenset[MediaKind],
    ) -> UUID:
        """Pin ``ref`` for ``holder`` and return the new hold's id.

        Args:
            ref: The asset to hold.
            holder: Who holds it. One holder holds one asset at most once.
            user_id: The asset's owner (and the holder's).
            product_id: The product both belong to.
            accept: The media kinds this holder may pin. Required, with no default: every holder
                declares what it pins, so a new kind can never be held by a holder that did not
                say it wanted it.

        Raises:
            AssetNotFoundError: The asset is missing, owned by another user or in another
                product (indistinguishable), or was deleted concurrently.
            AssetNotEligibleError: A derivative, a kind outside ``accept``, or an output whose
                job is not available.
            sqlalchemy.exc.IntegrityError: ``holder`` already holds the asset - a programming
                error in the caller, deliberately not swallowed.
        """
        if ref.source == LibraryAssetSource.UPLOAD:
            await self._check_upload(
                ref.asset_id, user_id=user_id, product_id=product_id, accept=accept
            )
            return await self._insert(
                holder, user_id=user_id, product_id=product_id, upload_id=ref.asset_id
            )
        await self._check_output(
            ref.asset_id, user_id=user_id, product_id=product_id, accept=accept
        )
        return await self._insert(
            holder, user_id=user_id, product_id=product_id, output_id=ref.asset_id
        )

    async def _check_upload(
        self, upload_id: UUID, *, user_id: UUID, product_id: str, accept: frozenset[MediaKind]
    ) -> None:
        asset = await self._repo.lock_upload(upload_id, user_id=user_id, product_id=product_id)
        if asset is None:
            raise AssetNotFoundError
        if asset.is_derivative:
            raise AssetNotEligibleError(AssetIneligibleReason.DERIVATIVE)
        self._check_kind(asset.content_type, accept)

    @staticmethod
    def _check_kind(content_type: str, accept: frozenset[MediaKind]) -> None:
        """The asset's kind (from its stored content type) must be one the holder accepts."""
        try:
            kind = media_kind_from_content_type(content_type)
        except UnknownMediaTypeError:
            raise AssetNotEligibleError(AssetIneligibleReason.MEDIA_KIND_NOT_ACCEPTED) from None
        if kind not in accept:
            raise AssetNotEligibleError(AssetIneligibleReason.MEDIA_KIND_NOT_ACCEPTED)

    async def _check_output(
        self, output_id: UUID, *, user_id: UUID, product_id: str, accept: frozenset[MediaKind]
    ) -> None:
        # Output first, then its job (see the module docstring for why not the other way round).
        asset = await self._repo.lock_output(output_id, user_id=user_id, product_id=product_id)
        if asset is None or asset.job_id is None:
            raise AssetNotFoundError
        # The job's state is read UNDER its share lock: a soft-delete that committed before we got
        # the lock is visible here, one that comes later waits for us and then sees our hold.
        job = await self._repo.lock_job_shared(asset.job_id, user_id=user_id, product_id=product_id)
        if job is None:
            raise AssetNotFoundError
        if asset.is_derivative:
            raise AssetNotEligibleError(AssetIneligibleReason.DERIVATIVE)
        self._check_kind(asset.content_type, accept)
        if job.status != JobStatus.COMPLETED or job.is_deleted:
            raise AssetNotEligibleError(AssetIneligibleReason.JOB_NOT_AVAILABLE)

    async def _insert(
        self,
        holder: AssetHolder,
        *,
        user_id: UUID,
        product_id: str,
        upload_id: UUID | None = None,
        output_id: UUID | None = None,
    ) -> UUID:
        """Insert the hold in a savepoint; a vanished asset is ``AssetNotFoundError``.

        The asset is locked, so a concurrent delete cannot get in between; the constraint name is
        still matched (never a prior ``SELECT`` alone) so that an FK violation is *always* a 404
        and never a 500, whatever lock the caller took or skipped.
        """
        try:
            async with self._session.begin_nested():
                return await self._repo.insert(
                    user_id=user_id,
                    product_id=product_id,
                    upload_id=upload_id,
                    output_id=output_id,
                    holder_kind=holder.kind,
                    holder_id=holder.id,
                )
        except IntegrityError as exc:
            if violated_constraint(exc, _ALL_CONSTRAINTS) in HOLD_ASSET_FKS:
                raise AssetNotFoundError from exc
            raise  # the same holder holds the same asset twice: a programming error

    async def release(self, hold_ids: Sequence[UUID]) -> None:
        """Release holds; re-arm every asset that is left with none. Unknown ids are ignored.

        The released assets are locked ``FOR NO KEY UPDATE`` (uploads, then outputs, ascending
        id) **before** the holds are deleted. That serializes two concurrent releases of one
        asset's last two holds: the second sees the first's committed delete and re-arms the
        asset; without the lock each would still see the other's hold and neither would.
        """
        if not hold_ids:
            return
        rows = await self._repo.assets_of_holds(hold_ids)
        upload_ids = sorted({upload for _, upload, _ in rows if upload is not None})
        output_ids = sorted({output for _, _, output in rows if output is not None})
        await self._repo.lock_for_release(upload_ids=upload_ids, output_ids=output_ids)
        await self._repo.delete(hold_ids)
        await self._repo.rearm_unheld(
            upload_ids=upload_ids, output_ids=output_ids, retention_days=self._retention_days
        )

    async def held_refs(
        self, refs: Sequence[AssetRef], *, user_id: UUID, product_id: str
    ) -> frozenset[AssetRef]:
        """The refs among ``refs`` (of this owner) that have at least one hold."""
        uploads = [r.asset_id for r in refs if r.source == LibraryAssetSource.UPLOAD]
        outputs = [r.asset_id for r in refs if r.source == LibraryAssetSource.OUTPUT]
        held_uploads = await self._repo.held_upload_ids(
            uploads, user_id=user_id, product_id=product_id
        )
        held_outputs = await self._repo.held_output_ids(
            outputs, user_id=user_id, product_id=product_id
        )
        return frozenset(
            ref
            for ref in refs
            if ref.asset_id
            in (held_uploads if ref.source == LibraryAssetSource.UPLOAD else held_outputs)
        )

    async def holder_counts(self, ref: AssetRef) -> tuple[HolderCount, ...]:
        """The asset's holders grouped by kind, ordered by kind (empty: not held)."""
        return await holder_counts_for_ref(self._session, ref)

    async def hold_refs(self, hold_ids: Sequence[UUID]) -> Mapping[UUID, AssetRef]:
        """The asset each hold pins (unknown ids are absent)."""
        refs: dict[UUID, AssetRef] = {}
        for hold_id, upload_id, output_id in await self._repo.assets_of_holds(hold_ids):
            if upload_id is not None:
                refs[hold_id] = AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=upload_id)
            elif output_id is not None:
                refs[hold_id] = AssetRef(source=LibraryAssetSource.OUTPUT, asset_id=output_id)
        return refs
