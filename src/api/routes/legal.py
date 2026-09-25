"""Legal document & acceptance endpoints.

Endpoints:
  GET  /v1/legal/documents/{doc_type}  — unauthenticated; one document (current, or ?version=YYYY-MM-DD)
  GET  /v1/legal/current               — unauthenticated; current versions of the required set
  GET  /v1/legal/status                — auth; the caller's acceptance state
  POST /v1/legal/acceptances           — auth, legal-exempt; record re-acceptance

After a successful ``POST /acceptances`` the client must call
``POST /v1/auth/refresh`` — only a freshly minted access token carries the
updated ``lgl`` digest that ``auth_guard`` checks. Contract:
``docs/contracts/legal-documents-contract.md``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

import structlog
from litestar import Controller, Request, Response, get, post
from litestar.di import Provide
from litestar.params import Body, Parameter
from litestar.status_codes import HTTP_200_OK, HTTP_304_NOT_MODIFIED

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.legal import (
    LegalAcceptanceRequest,
    LegalCurrentResponse,
    LegalDocumentMeta,
    LegalDocumentResponse,
    LegalDocumentStatusItem,
    LegalStatusResponse,
)
from src.api.security import auth_guard
from src.api.services.legal.acceptance import (
    LegalAcceptanceService,
    LegalStatus,
    RequestContext,
)
from src.api.services.legal.errors import LegalDocumentNotFoundError
from src.api.services.legal.registry import LegalDocumentRegistry
from src.core.enums import LegalAcceptanceSource, LegalDocumentType
from src.core.product import ProductConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

_DOCUMENT_EXACT_CACHE_CONTROL = "private, max-age=31536000, immutable"
_DOCUMENT_CURRENT_CACHE_CONTROL = "private, no-cache"
_DOCUMENT_VARY = "Origin, X-Product-Id"


def _today() -> date:
    return datetime.now(UTC).date()


def _if_none_match_matches(if_none_match: str | None, quoted_etag: str) -> bool:
    """Compare one validator, accepting the weak form browsers may send."""
    if if_none_match is None:
        return False
    candidate = if_none_match.strip()
    if candidate.startswith("W/"):
        candidate = candidate[2:].strip()
    return candidate == quoted_etag


def _parse_required_type(raw: str, product_config: ProductConfig) -> LegalDocumentType:
    """Parse a path segment into a type the product actually serves; 404 otherwise."""
    try:
        doc_type = LegalDocumentType(raw)
    except ValueError as exc:
        raise LegalDocumentNotFoundError(raw) from exc
    if doc_type not in product_config.required_legal_documents:
        raise LegalDocumentNotFoundError(raw)
    return doc_type


def _status_response(status: LegalStatus) -> LegalStatusResponse:
    return LegalStatusResponse(
        documents=[
            LegalDocumentStatusItem(
                doc_type=d.doc_type,
                required_version=d.required_version,
                current_version=d.current_version,
                accepted_version=d.accepted_version,
                accepted_at=d.accepted_at,
                satisfied=d.satisfied,
            )
            for d in status.documents
        ],
        all_satisfied=status.all_satisfied,
    )


class LegalController(Controller):
    """Versioned legal documents and the caller's acceptance of them."""

    path = "/v1/legal"
    tags: Sequence[str] | None = ("Legal",)

    @get("/documents/{doc_type:str}")
    async def get_document(
        self,
        doc_type: str,
        product_config: ProductConfig,
        legal_registry: LegalDocumentRegistry,
        request: Request[Any, Any, Any],
        version: Annotated[
            date | None,
            Parameter(
                query="version",
                description="Exact version (effective date, YYYY-MM-DD). Omit for current.",
            ),
        ] = None,
    ) -> Response[LegalDocumentResponse]:
        """Fetch one legal document — the current version, or an exact one.

        Unauthenticated. 404 ``legal_document_not_found`` for an unknown type, a type
        this product doesn't require, or an unknown version. Future-dated
        versions are fetchable by exact ``version`` (advance notice) but are
        never "current" before their date.
        """
        parsed = _parse_required_type(doc_type, product_config)
        doc = (
            legal_registry.current(product_config.product, parsed, today=_today())
            if version is None
            else legal_registry.get(product_config.product, parsed, version)
        )
        quoted_etag = f'"{doc.sha256}"'
        cache_control = (
            _DOCUMENT_CURRENT_CACHE_CONTROL if version is None else _DOCUMENT_EXACT_CACHE_CONTROL
        )
        headers = {
            "ETag": quoted_etag,
            "Cache-Control": cache_control,
            "Vary": _DOCUMENT_VARY,
        }
        if _if_none_match_matches(request.headers.get("if-none-match"), quoted_etag):
            return Response(
                content=None,  # type: ignore[arg-type]
                status_code=HTTP_304_NOT_MODIFIED,
                headers=headers,
            )
        return Response(
            content=LegalDocumentResponse(
                doc_type=doc.doc_type,
                version=doc.version,
                requires_reacceptance=doc.requires_reacceptance,
                sha256=doc.sha256,
                content_md=doc.content_md,
            ),
            status_code=HTTP_200_OK,
            headers=headers,
        )

    @get("/current")
    async def get_current(
        self,
        product_config: ProductConfig,
        legal_registry: LegalDocumentRegistry,
    ) -> LegalCurrentResponse:
        """Current versions of the product's required documents.

        Unauthenticated. The signup form renders these and echoes each
        ``{doc_type, version, sha256}`` back as ``accepted_documents``.
        """
        today = _today()
        return LegalCurrentResponse(
            documents=[
                LegalDocumentMeta(
                    doc_type=doc.doc_type,
                    version=doc.version,
                    sha256=doc.sha256,
                    requires_reacceptance=doc.requires_reacceptance,
                )
                for doc in legal_registry.list_current(product_config.product, today=today)
                if doc.doc_type in product_config.required_legal_documents
            ]
        )

    @get(
        "/status",
        guards=[auth_guard],
        dependencies={"current_user_id": Provide(get_current_user_id)},
    )
    async def get_status(
        self,
        current_user_id: UUID,
        product_config: ProductConfig,
        legal_acceptance_service: LegalAcceptanceService,
    ) -> LegalStatusResponse:
        """The caller's acceptance state for each required document."""
        status = await legal_acceptance_service.status(
            user_id=current_user_id, product=product_config, today=_today()
        )
        return _status_response(status)

    @post(
        "/acceptances",
        status_code=HTTP_200_OK,
        guards=[auth_guard],
        dependencies={"current_user_id": Provide(get_current_user_id)},
        opt={"legal_exempt": True},
    )
    async def accept(
        self,
        current_user_id: UUID,
        data: Annotated[LegalAcceptanceRequest, Body()],
        product_config: ProductConfig,
        legal_acceptance_service: LegalAcceptanceService,
        request_context: RequestContext,
    ) -> LegalStatusResponse:
        """Record (re-)acceptance of the current required documents.

        Must cover exactly the required set at current versions (422 / 409
        otherwise). Already-accepted current versions are skipped, so
        re-submitting is idempotent. The response is the new status; the
        client must then call ``POST /v1/auth/refresh`` to obtain a token
        whose ``lgl`` digest passes the mutation guard.
        """
        today = _today()
        documents = legal_acceptance_service.validate_submission(
            product_config, data.accepted_documents, today=today
        )
        await legal_acceptance_service.record_acceptances(
            user_id=current_user_id,
            product=product_config,
            documents=documents,
            source=LegalAcceptanceSource.REACCEPT,
            context=request_context,
        )
        status = await legal_acceptance_service.status(
            user_id=current_user_id, product=product_config, today=today
        )
        return _status_response(status)
