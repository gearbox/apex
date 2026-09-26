"""Repository for the append-only ``legal_acceptances`` ledger."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from sqlalchemy import select, text

from src.core.enums import LegalDocumentType
from src.db.models.legal import LegalAcceptance
from src.db.models.user import User

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

    async def is_user_active(self, *, user_id: UUID) -> bool:
        """Whether the user row exists and is active, read fresh from the database.

        A deliberate exception to this repository touching only
        ``legal_acceptances``: the check must run *inside* the ledger's
        critical section (after :meth:`lock_user_ledger`), so that an
        acceptance waiting behind a concurrent account closure sees the
        committed ``is_active = false``.

        - A scalar column select, never ``session.get``: the identity map may
          hold a ``User`` loaded earlier in the request with a stale
          ``is_active``; a column select always hits the database.
        - A plain ``SELECT``, never a row-locking read: it must not take a
          user-row lock inside the ledger lock, or it would interact with the
          revocation/logout lock ordering. The advisory lock already
          serializes it against closure, and under READ COMMITTED a statement
          issued after the lock is granted sees the closure's commit.

        Returns:
            ``False`` for an inactive or missing user.
        """
        stmt = select(User.is_active).where(User.id == user_id)
        return bool((await self._session.execute(stmt)).scalar_one_or_none())

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
            # seq is the serialized write order; timestamps are evidence only.
            .order_by(LegalAcceptance.doc_type, LegalAcceptance.seq.desc())
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return {LegalDocumentType(row.doc_type): row for row in rows}
