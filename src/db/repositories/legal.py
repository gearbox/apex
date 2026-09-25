"""Repository for the append-only ``legal_acceptances`` ledger."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from src.core.enums import LegalDocumentType
from src.db.models.legal import LegalAcceptance

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


class LegalAcceptanceRepository:
    """Insert-only access to legal acceptance events.

    Deliberately exposes no update or delete: the ledger is proof of what a
    user accepted and when. Never flushes or commits — the owning service and
    request transaction decide that.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add_many(self, rows: Sequence[LegalAcceptance]) -> None:
        """Stage new event rows in the session (no flush)."""
        self._session.add_all(rows)

    async def latest_per_type(
        self, *, user_id: UUID, product_id: str
    ) -> Mapping[LegalDocumentType, LegalAcceptance]:
        """Most recent event per document type for one user on one product.

        Args:
            user_id: The user.
            product_id: Product scope — rows from other products are never returned.

        Returns:
            ``{doc_type: latest event}``; types with no events are absent.
        """
        stmt = (
            select(LegalAcceptance)
            .where(
                LegalAcceptance.user_id == user_id,
                LegalAcceptance.product_id == product_id,
            )
            .distinct(LegalAcceptance.doc_type)
            .order_by(
                LegalAcceptance.doc_type,
                LegalAcceptance.created_at.desc(),
                LegalAcceptance.id.desc(),
            )
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return {LegalDocumentType(row.doc_type): row for row in rows}
