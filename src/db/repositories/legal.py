"""Repository for the append-only ``legal_acceptances`` ledger."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from sqlalchemy import select, text

from src.core.enums import LegalDocumentType
from src.db.models.legal import LegalAcceptance

# A dedicated int4 namespace for the two-key transaction-scoped legal-ledger
# lock. It is distinct from the GPU session advisory-lock key spaces.
LEGAL_LEDGER_LOCK_NAMESPACE: Final = 1_279_745_073  # ``LGL1`` in ASCII.

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

    async def lock_user_ledger(self, *, user_id: UUID) -> None:
        """Serialize ledger writes for one user until this transaction finishes.

        ``hashtext`` collisions only serialize unrelated users; they cannot
        make their ledger state incorrect. PostgreSQL releases this advisory
        transaction lock automatically at commit or rollback.
        """
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(:ns, hashtext(:uid))"),
            {"ns": LEGAL_LEDGER_LOCK_NAMESPACE, "uid": str(user_id)},
        )

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
