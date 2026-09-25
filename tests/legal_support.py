"""Shared builders for legal-document tests and for services that now require them."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

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
from src.core.product_registry import VEX_CONFIG
from src.db.repositories.legal import LegalAcceptanceRepository

if TYPE_CHECKING:
    from collections.abc import Mapping

    from litestar.types import ASGIApp, Receive, Scope, Send

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
    session = AsyncMock(spec=AsyncSession)
    return LegalAcceptanceService(
        registry=registry if registry is not None else make_legal_registry(),
        repository=repo,
        session=session,
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


# --- HTTP harness wiring for non-legal route tests ------------------------
# ``auth_guard`` enforces legal acceptance on every non-safe method, which
# needs product scope in connection state and a registry in app.state. Route
# tests that build a bare Litestar app wire these three together: add
# ``vex_product_scope`` as middleware, put ``TEST_LEGAL_REGISTRY`` at
# ``app.state["legal_registry"]``, and mint tokens with
# ``product_id="vex", legal_digest=TEST_LEGAL_DIGEST``.

TEST_LEGAL_REGISTRY = make_legal_registry()
TEST_LEGAL_DIGEST = TEST_LEGAL_REGISTRY.required_digest(
    VEX_CONFIG,
    today=date.today(),  # noqa: DTZ011 — test helper
)


def vex_product_scope(app: ASGIApp) -> ASGIApp:
    """Stand-in for ProductMiddleware that pins every request to vex.

    Sets the same state keys ``ProductMiddleware`` does, without needing the
    test client's host or an ``X-Product-Id`` header to resolve.
    """

    async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            state = scope.setdefault("state", {})
            state["product_config"] = VEX_CONFIG
            state["product_id"] = VEX_CONFIG.slug
        await app(scope, receive, send)

    return middleware
