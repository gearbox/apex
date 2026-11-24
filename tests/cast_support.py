"""Helpers shared by the cast tests (S1-P3): the reconciliation check (invariant 3) and builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from src.api.schemas.studio import CastMemberCreate, CastReferenceIn
from src.api.services.library_holds import AssetHoldService
from src.core.library_ref import format_asset_ref
from src.studio.services.cast import CastService
from tests.library_holds_support import RETENTION_DAYS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.library_ref import AssetRef
    from src.db.models.storage import GenerationOutput, UserImage

__all__ = ["assert_reconciled", "cast_service", "create_body", "ref_in", "wire"]

_ORPHAN_REFERENCES = text(
    """
    SELECT r.id FROM studio_cast_references r
    LEFT JOIN library_asset_holds h ON h.id = r.hold_id
    WHERE h.id IS NULL
       OR h.holder_kind <> 'studio.cast_reference'
       OR h.holder_id <> r.id
    """
)
_ORPHAN_HOLDS = text(
    """
    SELECT h.id FROM library_asset_holds h
    LEFT JOIN studio_cast_references r ON r.hold_id = h.id
    WHERE h.holder_kind = 'studio.cast_reference'
      AND (r.id IS NULL OR r.id <> h.holder_id)
    """
)
_BROKEN_POSITIONS = text(
    """
    SELECT cast_member_id FROM studio_cast_references
    GROUP BY cast_member_id
    HAVING count(*) <> max(position) + 1 OR min(position) <> 0
    """
)


async def assert_reconciled(session: AsyncSession) -> None:
    """Invariant 3: every reference has exactly one hold and every cast hold has its reference.

    Also: positions are dense (``0..n-1``) for every member, and the deferred unique on them holds
    (``SET CONSTRAINTS ALL IMMEDIATE`` runs the check that a commit would). Run after every cast
    test.
    """
    # The position key is DEFERRABLE INITIALLY DEFERRED: inside a transaction that is rolled back
    # (the savepoint-scoped ``db_session``) it would never be checked. Check it now.
    await session.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
    orphan_references = (await session.execute(_ORPHAN_REFERENCES)).scalars().all()
    assert not orphan_references, f"references without their hold: {orphan_references}"
    orphan_holds = (await session.execute(_ORPHAN_HOLDS)).scalars().all()
    assert not orphan_holds, f"cast holds without their reference: {orphan_holds}"
    broken = (await session.execute(_BROKEN_POSITIONS)).scalars().all()
    assert not broken, f"members whose reference positions are not 0..n-1: {broken}"


def cast_service(session: AsyncSession) -> CastService:
    return CastService(session, AssetHoldService(session, retention_days=RETENTION_DAYS))


def wire(asset: AssetRef | UserImage | GenerationOutput) -> str:
    """The wire asset ref (``upload:<uuid>`` / ``output:<uuid>``) of a ref or an ORM row."""
    from src.core.library_ref import AssetRef, LibraryAssetSource
    from src.db.models.storage import UserImage

    if isinstance(asset, AssetRef):
        return format_asset_ref(asset.source, asset.asset_id)
    source = (
        LibraryAssetSource.UPLOAD if isinstance(asset, UserImage) else LibraryAssetSource.OUTPUT
    )
    return format_asset_ref(source, asset.id)


def ref_in(label: str, asset: AssetRef | UserImage | GenerationOutput) -> CastReferenceIn:
    return CastReferenceIn(label=label, asset=wire(asset))


def create_body(
    name: str = "Mira",
    *,
    references: Sequence[CastReferenceIn] = (),
    description: str | None = None,
    appearance: str | None = None,
) -> CastMemberCreate:
    return CastMemberCreate(
        name=name,
        description=description,
        appearance=appearance,
        references=list(references),
    )


async def count(session: AsyncSession, table: str, **where: UUID | str) -> int:
    clause = " AND ".join(f"{column} = :{column}" for column in where) or "true"
    return (
        await session.execute(text(f"SELECT count(*) FROM {table} WHERE {clause}"), where)  # noqa: S608
    ).scalar_one()
