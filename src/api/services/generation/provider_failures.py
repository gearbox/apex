"""Provider failure taxonomy shared by generation integrations.

Provider clients keep raw upstream diagnostic data out of this module.  The
``ProviderFailure`` value contains only normalized, safe-to-persist fields
that can be used consistently by job state, billing policy, and API code.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from src.api.services.billing import BalanceEvent
    from src.core.enums import Provider


class ProviderFailureKind(StrEnum):
    """Normalized failure categories returned by generation providers."""

    MODERATION_REJECTED = "moderation_rejected"
    INVALID_REQUEST = "invalid_request"
    RATE_LIMITED = "rate_limited"
    AUTHENTICATION_FAILED = "authentication_failed"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"
    UNKNOWN = "unknown"


_PUBLIC_MESSAGES: dict[ProviderFailureKind, str] = {
    ProviderFailureKind.MODERATION_REJECTED: (
        "The requested content was rejected by the AI provider's safety system. "
        "Modify the prompt or input and try again."
    ),
    ProviderFailureKind.INVALID_REQUEST: "The AI provider could not process this request.",
    ProviderFailureKind.RATE_LIMITED: "The AI provider is temporarily rate limited.",
    ProviderFailureKind.AUTHENTICATION_FAILED: "The AI provider is unavailable.",
    ProviderFailureKind.PROVIDER_UNAVAILABLE: "The AI provider is temporarily unavailable.",
    ProviderFailureKind.TIMEOUT: "The AI provider timed out while processing the request.",
    ProviderFailureKind.MALFORMED_RESPONSE: "The AI provider returned an invalid response.",
    ProviderFailureKind.UNKNOWN: "The AI provider could not complete the request.",
}

_PUBLIC_CODES: dict[ProviderFailureKind, str] = {
    ProviderFailureKind.MODERATION_REJECTED: "provider_moderation_rejected",
    ProviderFailureKind.INVALID_REQUEST: "provider_invalid_request",
    ProviderFailureKind.RATE_LIMITED: "provider_rate_limited",
    ProviderFailureKind.AUTHENTICATION_FAILED: "provider_authentication_failed",
    ProviderFailureKind.PROVIDER_UNAVAILABLE: "provider_unavailable",
    ProviderFailureKind.TIMEOUT: "provider_timeout",
    ProviderFailureKind.MALFORMED_RESPONSE: "provider_malformed_response",
    ProviderFailureKind.UNKNOWN: "provider_unknown",
}


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderFailure:
    """A provider failure normalized for safe internal handling.

    ``provider_request_accepted`` is deliberately three-valued. ``None``
    means that the integration cannot establish whether the provider accepted
    the request, so billing policy must preserve its existing behavior rather
    than treating it as either affirmative or negative.
    """

    kind: ProviderFailureKind
    provider: Provider
    sanitized_message: str
    provider_status_code: int | None = None
    provider_error_code: str | None = None
    retryable: bool = False
    provider_request_accepted: bool | None = None
    billable: bool = False
    provider_request_id: str | None = None

    @property
    def public_code(self) -> str:
        """Stable client-visible error code for this failure."""
        return _PUBLIC_CODES[self.kind]

    @classmethod
    def safe_message_for_kind(cls, kind: ProviderFailureKind) -> str:
        """Return the client-safe message for a normalized failure kind."""
        return _PUBLIC_MESSAGES[kind]


class ProviderSubmissionFailedError(Exception):
    """A normalized provider submission failure with durable job context.

    The provider adapter raises this after it has created the job and reserved
    its debit, but before it mutates either.  The generation orchestrator owns
    the single settlement decision (failed state, charge/refund, commit, and
    post-commit events), so it cannot accidentally roll back the normalized
    terminal record.
    """

    def __init__(
        self,
        *,
        failure: ProviderFailure,
        job_id: UUID,
        previous_status: str = "unknown",
        balance_event: BalanceEvent | None,
    ) -> None:
        self.failure = failure
        self.job_id = job_id
        self.previous_status = previous_status
        self.balance_event = balance_event
        self.public_code = failure.public_code
        # ``sanitized_message`` is retained for compatibility with existing
        # normalized provider adapters, but callers must never be able to
        # make a free-form string public by constructing ProviderFailure.
        self.public_message = ProviderFailure.safe_message_for_kind(failure.kind)
        super().__init__(self.public_message)


class ProviderModerationRejectedError(ProviderSubmissionFailedError):
    """A billable provider moderation rejection with submission context.

    This exception intentionally has a fixed public message. The original
    provider message remains diagnostic-only and is never made part of the
    exception text, persisted job fields, or API response.
    """

    public_code = "provider_moderation_rejected"
    public_message = _PUBLIC_MESSAGES[ProviderFailureKind.MODERATION_REJECTED]

    def __init__(
        self,
        *,
        failure: ProviderFailure,
        job_id: UUID,
        previous_status: str = "unknown",
        balance_event: BalanceEvent | None,
    ) -> None:
        if failure.kind != ProviderFailureKind.MODERATION_REJECTED:
            raise ValueError("ProviderModerationRejectedError requires a moderation failure")
        super().__init__(
            failure=failure,
            job_id=job_id,
            previous_status=previous_status,
            balance_event=balance_event,
        )
