"""Legal-document domain exceptions.

Each HTTP-facing error is mapped to an ``ErrorEnvelope`` response by a handler
registered in ``src/api/app.py``:

| Exception | Status | ``error`` code |
|---|---|---|
| ``LegalDocumentNotFoundError`` | 404 | ``legal_document_not_found`` |
| ``LegalSubmissionIncompleteError`` | 422 | ``legal_acceptance_incomplete`` |
| ``LegalVersionStaleError`` | 409 | ``legal_version_stale`` |
| ``LegalAcceptanceRequiredError`` | 428 | ``legal_acceptance_required`` |
| ``LegalAccountInactiveError`` | 401 | ``account_inactive`` |

``LegalRegistryError`` is a startup failure, never an HTTP response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date
    from uuid import UUID

    from src.core.enums import LegalDocumentType


class LegalError(Exception):
    """Base class for legal-document errors."""


class LegalRegistryError(LegalError):
    """The legal manifest/files are inconsistent or unfit to publish (startup failure)."""


class LegalDocumentNotFoundError(LegalError):
    """No such document type or version for the product."""

    def __init__(self, doc_type: str, version: date | None = None) -> None:
        self.doc_type = doc_type
        self.version = version
        suffix = f" version {version.isoformat()}" if version is not None else ""
        super().__init__(f"Legal document {doc_type!r}{suffix} not found")


class LegalSubmissionIncompleteError(LegalError):
    """Submitted document types don't exactly match the product's required set."""

    def __init__(
        self,
        *,
        missing: Sequence[LegalDocumentType],
        unexpected: Sequence[LegalDocumentType],
        duplicated: Sequence[LegalDocumentType],
    ) -> None:
        self.missing = sorted(missing)
        self.unexpected = sorted(unexpected)
        self.duplicated = sorted(duplicated)
        super().__init__("Accepted documents must cover exactly the required legal documents")

    @property
    def detail(self) -> dict[str, Any]:
        """Wire-format detail for the 422 envelope."""
        return {
            "missing": [t.value for t in self.missing],
            "unexpected": [t.value for t in self.unexpected],
            "duplicated": [t.value for t in self.duplicated],
        }


class LegalVersionStaleError(LegalError):
    """A submitted version/sha256 is not the current one — the client must refetch."""

    def __init__(self, current: Sequence[dict[str, str]]) -> None:
        self.current = list(current)
        super().__init__("Submitted legal document versions are not current")

    @property
    def detail(self) -> dict[str, Any]:
        """Wire-format detail for the 409 envelope."""
        return {"current": self.current}


class LegalAcceptanceRequiredError(LegalError):
    """The caller's token doesn't carry the currently required acceptance digest."""

    def __init__(self) -> None:
        super().__init__(
            "You must accept the current legal documents before continuing. "
            "Accept them, then refresh your session."
        )


class LegalAccountInactiveError(LegalError):
    """The account was closed/deactivated before this acceptance could be recorded.

    Raised inside the ledger lock, so it also covers a closure that committed
    while the acceptance request (which already passed ``auth_guard``) waited.
    Maps to the same ``account_inactive`` 401 that login and refresh return.
    """

    def __init__(self, *, user_id: UUID, product_id: str) -> None:
        self.user_id = user_id
        self.product_id = product_id
        super().__init__("Account has been deactivated")
