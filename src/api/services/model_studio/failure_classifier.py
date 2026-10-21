"""Classification of Model Studio failures from structured signals only.

The decision uses the upstream ``code`` and the HTTP status. Free-text
messages are never inspected: a message that merely sounds like moderation is
not evidence of a content rejection, and a wrong guess here decides who pays.
"""

from __future__ import annotations

from typing import Final

from src.api.services.generation.provider_failures import ProviderFailure, ProviderFailureKind
from src.core.enums import Provider

_MODERATION_CODES: Final = frozenset(
    {"DataInspectionFailed", "data_inspection_failed", "IPInfringementSuspect", "CustomRoleBlocked"}
)
# Billing / account states. They win over the HTTP status: several are returned as 429.
_UNAVAILABLE_ACCOUNT_CODES: Final = frozenset(
    {
        "Arrearage",
        "AllocationQuota.FreeTierOnly",
        "PrepaidBillOverdue",
        "PostpaidBillOverdue",
        "BudgetLimitExceeded",
        "CommodityNotPurchased",
        "isv.OUT_OF_SERVICE",
        "ModelNotFound",
        "model_not_found",
    }
)
_UNAVAILABLE_PREFIXES: Final = ("InternalError",)
_AUTH_CODES: Final = frozenset(
    {
        "InvalidApiKey",
        "invalid_api_key",
        "Workspace.AccessDenied",
        "Model.AccessDenied",
        "App.AccessDenied",
        "Endpoint.AccessDenied",
        "NOT AUTHORIZED",
    }
)
_AUTH_PREFIXES: Final = ("AccessDenied",)
_INVALID_EXACT: Final = frozenset({"InvalidURL"})
_INVALID_PREFIXES: Final = ("InvalidParameter", "InvalidImage", "invalid_request_error")
_INVALID_DOTTED: Final = ("InvalidFile", "BadRequest")
_RATE_EXACT: Final = frozenset(
    {"LimitRequests", "limit_requests", "limit_burst_rate", "insufficient_quota"}
)
_RATE_PREFIXES: Final = ("Throttling",)

_MAX_CODE_LENGTH: Final = 100
_RETRYABLE: Final = frozenset(
    {
        ProviderFailureKind.RATE_LIMITED,
        ProviderFailureKind.PROVIDER_UNAVAILABLE,
        ProviderFailureKind.TIMEOUT,
    }
)


def _dotted_match(code: str, prefix: str) -> bool:
    return code == prefix or code.startswith(f"{prefix}.")


def _kind_from_code(code: str) -> ProviderFailureKind | None:
    """Map a documented code to a kind; ``None`` when the code is not recognized."""
    if code in _UNAVAILABLE_ACCOUNT_CODES or code.startswith(_UNAVAILABLE_PREFIXES):
        return ProviderFailureKind.PROVIDER_UNAVAILABLE
    if code in _MODERATION_CODES:
        return ProviderFailureKind.MODERATION_REJECTED
    if code in _AUTH_CODES or code.startswith(_AUTH_PREFIXES):
        return ProviderFailureKind.AUTHENTICATION_FAILED
    if (
        code in _INVALID_EXACT
        or code.startswith(_INVALID_PREFIXES)
        or any(_dotted_match(code, prefix) for prefix in _INVALID_DOTTED)
    ):
        return ProviderFailureKind.INVALID_REQUEST
    if code in _RATE_EXACT or code.startswith(_RATE_PREFIXES):
        return ProviderFailureKind.RATE_LIMITED
    return None


def _kind_from_status(status: int | None) -> ProviderFailureKind | None:
    if status in (401, 403):
        return ProviderFailureKind.AUTHENTICATION_FAILED
    if status == 429:
        return ProviderFailureKind.RATE_LIMITED
    if status is not None and status >= 500:
        return ProviderFailureKind.PROVIDER_UNAVAILABLE
    return None


class ModelStudioFailureClassifier:
    """Normalize Model Studio error codes and statuses into ``ProviderFailure``."""

    def classify(
        self,
        *,
        code: str | None = None,
        http_status: int | None = None,
        provider_request_accepted: bool | None = None,
        provider_request_id: str | None = None,
    ) -> ProviderFailure:
        """Classify one failure.

        Args:
            code: Upstream ``code`` (create error body, or a FAILED task's code).
            http_status: HTTP status of the response, when there was one.
            provider_request_accepted: ``True`` once a task exists (any query),
                ``False`` for a 4xx on create, ``None`` when ambiguous.
            provider_request_id: Upstream request or task id, for diagnostics.
        """
        # A CANCELED/UNKNOWN task carries no code and no error status, so it
        # falls through to UNKNOWN like any other unrecognized signal.
        kind = (
            (_kind_from_code(code) if code else None)
            or _kind_from_status(http_status)
            or ProviderFailureKind.UNKNOWN
        )
        return self._build(
            kind,
            code=code,
            http_status=http_status,
            provider_request_accepted=provider_request_accepted,
            provider_request_id=provider_request_id,
        )

    def timeout(
        self,
        *,
        provider_request_accepted: bool | None,
        provider_request_id: str | None = None,
    ) -> ProviderFailure:
        """A client-side timeout (``httpx.TimeoutException``)."""
        return self._build(
            ProviderFailureKind.TIMEOUT,
            provider_request_accepted=provider_request_accepted,
            provider_request_id=provider_request_id,
        )

    def unavailable(
        self,
        *,
        provider_request_accepted: bool | None,
        provider_request_id: str | None = None,
    ) -> ProviderFailure:
        """A connection-level failure (DNS, refused, reset)."""
        return self._build(
            ProviderFailureKind.PROVIDER_UNAVAILABLE,
            provider_request_accepted=provider_request_accepted,
            provider_request_id=provider_request_id,
        )

    def malformed(
        self,
        *,
        provider_request_accepted: bool | None,
        provider_request_id: str | None = None,
        http_status: int | None = None,
    ) -> ProviderFailure:
        """An unparseable 2xx body, or ``SUCCEEDED`` without a ``video_url``."""
        return self._build(
            ProviderFailureKind.MALFORMED_RESPONSE,
            http_status=http_status,
            provider_request_accepted=provider_request_accepted,
            provider_request_id=provider_request_id,
        )

    @staticmethod
    def _build(
        kind: ProviderFailureKind,
        *,
        code: str | None = None,
        http_status: int | None = None,
        provider_request_accepted: bool | None,
        provider_request_id: str | None,
    ) -> ProviderFailure:
        return ProviderFailure(
            kind=kind,
            provider=Provider.ALIBABA,
            sanitized_message=ProviderFailure.safe_message_for_kind(kind),
            provider_status_code=http_status,
            provider_error_code=code[:_MAX_CODE_LENGTH] if code else None,
            retryable=kind in _RETRYABLE,
            provider_request_accepted=provider_request_accepted,
            provider_request_id=provider_request_id,
        )
