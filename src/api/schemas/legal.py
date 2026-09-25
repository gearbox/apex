"""Legal document & acceptance API schemas (wire contract).

Canonical frontend contract: ``docs/contracts/legal-documents-contract.md``.
"""

from __future__ import annotations

from datetime import date, datetime

import msgspec

# Runtime imports: msgspec resolves these annotations at runtime.
from src.api.services.legal.acceptance import AcceptedDocument
from src.core.enums import LegalDocumentType

__all__ = [
    "AcceptedDocument",
    "LegalAcceptanceRequest",
    "LegalCurrentResponse",
    "LegalDocumentMeta",
    "LegalDocumentResponse",
    "LegalDocumentStatusItem",
    "LegalStatusResponse",
]


class LegalDocumentResponse(msgspec.Struct, kw_only=True):
    """One version of a legal document, with its full markdown body."""

    doc_type: LegalDocumentType
    version: date
    """Effective date (UTC) — also the version identifier."""
    requires_reacceptance: bool
    sha256: str
    """Hex sha256 of ``content_md`` (UTF-8, LF line endings). Echo it back on acceptance."""
    content_md: str
    """Markdown source. Render with sanitisation; raw HTML is not expected."""


class LegalDocumentMeta(msgspec.Struct, kw_only=True):
    """Metadata of the current version of one required document."""

    doc_type: LegalDocumentType
    version: date
    sha256: str
    requires_reacceptance: bool


class LegalCurrentResponse(msgspec.Struct, kw_only=True):
    """Current versions of the product's required documents (empty if none required)."""

    documents: list[LegalDocumentMeta]


class LegalDocumentStatusItem(msgspec.Struct, kw_only=True):
    """The caller's acceptance state for one required document type."""

    doc_type: LegalDocumentType
    required_version: date
    """Minimum version that satisfies the requirement."""
    current_version: date
    """What the user would accept now (may be newer than ``required_version``)."""
    accepted_version: date | None
    accepted_at: datetime | None
    satisfied: bool


class LegalStatusResponse(msgspec.Struct, kw_only=True):
    """The caller's acceptance state across the product's required set."""

    documents: list[LegalDocumentStatusItem]
    all_satisfied: bool


class LegalAcceptanceRequest(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    """Re-acceptance submission: exactly the required set at current versions."""

    accepted_documents: list[AcceptedDocument]
