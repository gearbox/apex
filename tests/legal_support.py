"""Shared builders for legal-document tests and for services that now require them."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from src.api.services.legal.acceptance import (
    AcceptedDocument,
    LegalAcceptanceService,
    RequestContext,
)
from src.api.services.legal.registry import (
    LegalDocument,
    LegalDocumentRegistry,
    content_sha256,
)
from src.core.enums import LegalDocumentType, Product
from src.db.repositories.legal import LegalAcceptanceRepository

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.db.models.legal import LegalAcceptance

FIXTURE_LEGAL_DIR = Path(__file__).parent / "fixtures" / "legal"
DEFAULT_LEGAL_VERSION = date(2020, 1, 1)
ALL_LEGAL_TYPES: tuple[LegalDocumentType, ...] = tuple(sorted(LegalDocumentType))


def make_legal_document(
    doc_type: LegalDocumentType,
    version: date = DEFAULT_LEGAL_VERSION,
    *,
    product: Product = Product.VEX,
    requires_reacceptance: bool = True,
    content: str | None = None,
) -> LegalDocument:
    body = content if content is not None else f"# {doc_type} {version.isoformat()}\n"
    return LegalDocument(
        product=product,
        doc_type=doc_type,
        version=version,
        requires_reacceptance=requires_reacceptance,
        content_md=body,
        sha256=content_sha256(body),
    )


def make_legal_registry(*documents: LegalDocument) -> LegalDocumentRegistry:
    """Registry of the given documents, or vex x all three types at 2020-01-01."""
    if not documents:
        documents = tuple(make_legal_document(t) for t in ALL_LEGAL_TYPES)
    return LegalDocumentRegistry(documents)


def accepted(doc: LegalDocument) -> AcceptedDocument:
    """The wire submission that accepts exactly ``doc``."""
    return AcceptedDocument(doc_type=doc.doc_type, version=doc.version, sha256=doc.sha256)


def make_legal_acceptance_service(
    *,
    registry: LegalDocumentRegistry | None = None,
    latest: Mapping[LegalDocumentType, LegalAcceptance] | None = None,
) -> LegalAcceptanceService:
    """A real service over a mocked repository.

    ``latest`` is what ``latest_per_type`` returns (default: nothing
    accepted, so ``satisfied_digest`` is None and no ``lgl`` is minted).
    """
    repo = MagicMock(spec=LegalAcceptanceRepository)
    repo.latest_per_type = AsyncMock(return_value=dict(latest or {}))
    return LegalAcceptanceService(
        registry=registry if registry is not None else make_legal_registry(),
        repository=repo,
    )


TEST_REQUEST_CONTEXT = RequestContext(ip_address="203.0.113.7", user_agent="pytest-legal/1.0")


def accept_all_current(
    registry: LegalDocumentRegistry | None = None,
    *,
    product: Product = Product.VEX,
    today: date | None = None,
) -> list[AcceptedDocument]:
    """Submission accepting every current document of ``product``."""
    reg = registry if registry is not None else make_legal_registry()
    when = today if today is not None else date.today()  # noqa: DTZ011 — test helper
    return [accepted(doc) for doc in reg.list_current(product, today=when)]
