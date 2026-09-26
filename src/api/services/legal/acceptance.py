"""Legal acceptance: validate submissions, record events, compute the ``lgl`` digest.

The service is request-scoped (it wraps a session-bound repository) and never
commits. ``record_acceptances`` is intentionally caller-agnostic — signup,
re-acceptance, and a future OAuth callback all use it unchanged.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Annotated, Final

import msgspec
import structlog

from src.api.services.legal.errors import (
    LegalAccountInactiveError,
    LegalSubmissionIncompleteError,
    LegalVersionStaleError,
)
from src.core.enums import LegalAcceptanceSource, LegalAction, LegalDocumentType
from src.core.uid import new_id
from src.db.models.legal import LegalAcceptance

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.legal.registry import LegalDocument, LegalDocumentRegistry
    from src.core.product import ProductConfig
    from src.db.repositories.legal import LegalAcceptanceRepository

logger = structlog.get_logger(__name__)

_IP_MAX_LEN: Final = 64
_USER_AGENT_MAX_LEN: Final = 512


class AcceptedDocument(msgspec.Struct, kw_only=True, forbid_unknown_fields=True, frozen=True):
    """One document the client claims the user accepted (wire format)."""

    doc_type: LegalDocumentType
    version: date
    sha256: Annotated[str, msgspec.Meta(pattern=r"^[0-9a-f]{64}$")]


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Client metadata stored alongside an acceptance event as evidence."""

    ip_address: str | None
    user_agent: str | None


@dataclass(frozen=True, slots=True)
class LegalDocumentStatus:
    """Acceptance state of one required document type for one user."""

    doc_type: LegalDocumentType
    required_version: date
    current_version: date
    accepted_version: date | None
    accepted_at: datetime | None
    satisfied: bool


@dataclass(frozen=True, slots=True)
class LegalStatus:
    """Acceptance state across a product's required set."""

    documents: tuple[LegalDocumentStatus, ...]
    all_satisfied: bool


def _truncate(value: str | None, max_len: int) -> str | None:
    return None if value is None else value[:max_len]


class LegalAcceptanceService:
    """Business rules for legal acceptance events."""

    def __init__(
        self,
        *,
        registry: LegalDocumentRegistry,
        repository: LegalAcceptanceRepository,
        session: AsyncSession,
    ) -> None:
        """Initialize the service.

        Args:
            registry: The process-wide document registry.
            repository: Session-bound acceptance repository.
            session: The same request session the repository uses. Flushed
                after staging rows because sessions run with
                ``autoflush=False`` — without it, a ``latest_per_type`` read in
                the same transaction (e.g. ``satisfied_digest`` during signup)
                would not see the rows just added.
        """
        self._registry = registry
        self._repo = repository
        self._session = session

    @property
    def registry(self) -> LegalDocumentRegistry:
        """The document registry backing this service."""
        return self._registry

    # ------------------------------------------------------------------
    # Validation (pure — no DB access)
    # ------------------------------------------------------------------

    def validate_submission(
        self,
        product: ProductConfig,
        submitted: Sequence[AcceptedDocument],
        *,
        today: date,
    ) -> tuple[LegalDocument, ...]:
        """Check a submission covers exactly the required set at current versions.

        Args:
            product: The request's product.
            submitted: What the client says the user accepted.
            today: UTC date for "current" resolution.

        Returns:
            The current ``LegalDocument`` for each required type, ordered by type.

        Raises:
            LegalSubmissionIncompleteError: Missing, extra, or duplicated types (422).
            LegalVersionStaleError: A version or sha256 is not current (409).
        """
        required = product.required_legal_documents
        counts = Counter(d.doc_type for d in submitted)
        duplicated = [t for t, n in counts.items() if n > 1]
        missing = [t for t in required if t not in counts]
        unexpected = [t for t in counts if t not in required]
        if duplicated or missing or unexpected:
            raise LegalSubmissionIncompleteError(
                missing=missing, unexpected=unexpected, duplicated=duplicated
            )

        current = {
            t: self._registry.current(product.product, t, today=today) for t in sorted(required)
        }
        stale = any(
            d.version != current[d.doc_type].version or d.sha256 != current[d.doc_type].sha256
            for d in submitted
        )
        if stale:
            raise LegalVersionStaleError(
                [
                    {
                        "doc_type": doc.doc_type.value,
                        "version": doc.version.isoformat(),
                        "sha256": doc.sha256,
                    }
                    for doc in current.values()
                ]
            )
        return tuple(current.values())

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def record_acceptances(
        self,
        *,
        user_id: UUID,
        product: ProductConfig,
        documents: Sequence[LegalDocument],
        source: LegalAcceptanceSource,
        context: RequestContext,
    ) -> int:
        """Insert an ``ACCEPT`` event per document not already accepted at that version.

        Args:
            user_id: The accepting user.
            product: The user's product.
            documents: Validated documents (from :meth:`validate_submission`).
            source: Where the acceptance happened.
            context: IP / user agent evidence.

        Returns:
            Number of rows inserted (0 for an idempotent re-submission).

        Raises:
            LegalAccountInactiveError: The account is closed/deactivated — checked
                after taking the ledger lock, so an acceptance that waited behind
                a concurrent closure can never land after its ``WITHDRAW`` (401).
        """
        await self._repo.lock_user_ledger(user_id=user_id)
        if not await self._repo.is_user_active(user_id=user_id):
            raise LegalAccountInactiveError(user_id=user_id, product_id=product.slug)
        latest = await self._repo.latest_per_type(user_id=user_id, product_id=product.slug)
        rows = [
            LegalAcceptance(
                id=new_id(),
                user_id=user_id,
                product_id=product.slug,
                doc_type=doc.doc_type,
                action=LegalAction.ACCEPT,
                version=doc.version,
                content_sha256=doc.sha256,
                source=source,
                ip_address=_truncate(context.ip_address, _IP_MAX_LEN),
                user_agent=_truncate(context.user_agent, _USER_AGENT_MAX_LEN),
            )
            for doc in documents
            if not (
                (prev := latest.get(doc.doc_type)) is not None
                and prev.action == LegalAction.ACCEPT
                and prev.version == doc.version
            )
        ]
        if not rows:
            return 0
        self._repo.add_many(rows)
        await self._flush()
        # Never log IP or user agent — they are evidence in the DB row only.
        logger.info(
            "legal.acceptance.recorded",
            user_id=str(user_id),
            product_id=product.slug,
            doc_types=[row.doc_type for row in rows],
            source=source.value,
        )
        return len(rows)

    async def record_consent_withdrawal(
        self,
        *,
        user_id: UUID,
        product: ProductConfig,
        context: RequestContext,
        today: date,
    ) -> None:
        """Record withdrawal of sensitive-data consent (account closure only).

        No-op when the product doesn't require that consent. Deliberately no
        ``is_active`` check: closure deactivates the user in this same
        transaction, and the withdrawal must still be written.
        """
        await self._repo.lock_user_ledger(user_id=user_id)
        if LegalDocumentType.SENSITIVE_DATA_CONSENT not in product.required_legal_documents:
            return
        consent = self._registry.current(
            product.product, LegalDocumentType.SENSITIVE_DATA_CONSENT, today=today
        )
        self._repo.add_many(
            [
                LegalAcceptance(
                    id=new_id(),
                    user_id=user_id,
                    product_id=product.slug,
                    doc_type=LegalDocumentType.SENSITIVE_DATA_CONSENT,
                    action=LegalAction.WITHDRAW,
                    version=consent.version,
                    content_sha256=None,
                    source=LegalAcceptanceSource.ACCOUNT_CLOSURE,
                    ip_address=_truncate(context.ip_address, _IP_MAX_LEN),
                    user_agent=_truncate(context.user_agent, _USER_AGENT_MAX_LEN),
                )
            ]
        )
        await self._flush()
        logger.info("legal.consent.withdrawn", user_id=str(user_id), product_id=product.slug)

    async def _flush(self) -> None:
        # Sessions run with autoflush=False (src/db/session.py); flush here —
        # never in the repository — so later reads in this transaction see the rows.
        await self._session.flush()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def satisfied_digest(
        self, *, user_id: UUID, product: ProductConfig, today: date
    ) -> str | None:
        """The ``lgl`` digest this user has earned, or ``None``.

        Returns the registry's required digest only when every required
        type's latest event is an ``ACCEPT`` at or above the required version.
        """
        required = self._registry.required_versions(product, today=today)
        if not required:
            return None
        latest = await self._repo.latest_per_type(user_id=user_id, product_id=product.slug)
        for doc_type, required_version in required.items():
            event = latest.get(doc_type)
            if event is None or event.action != LegalAction.ACCEPT:
                return None
            if event.version < required_version:
                return None
        return self._registry.required_digest(product, today=today)

    async def status(self, *, user_id: UUID, product: ProductConfig, today: date) -> LegalStatus:
        """Per-type acceptance state for the product's required set."""
        required = self._registry.required_versions(product, today=today)
        latest = (
            await self._repo.latest_per_type(user_id=user_id, product_id=product.slug)
            if required
            else {}
        )
        documents: list[LegalDocumentStatus] = []
        for doc_type, required_version in required.items():
            event = latest.get(doc_type)
            accepted = event if event is not None and event.action == LegalAction.ACCEPT else None
            accepted_version = accepted.version if accepted is not None else None
            documents.append(
                LegalDocumentStatus(
                    doc_type=doc_type,
                    required_version=required_version,
                    current_version=self._registry.current(
                        product.product, doc_type, today=today
                    ).version,
                    accepted_version=accepted_version,
                    accepted_at=accepted.created_at if accepted is not None else None,
                    satisfied=accepted_version is not None and accepted_version >= required_version,
                )
            )
        return LegalStatus(
            documents=tuple(documents),
            all_satisfied=all(d.satisfied for d in documents),
        )
