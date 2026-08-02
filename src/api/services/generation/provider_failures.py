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
    OUTPUT_NOT_DELIVERED = "output_not_delivered"
    UNKNOWN = "unknown"


_MODERATION_REJECTION_MESSAGE = (
    "The requested content was rejected by the AI provider's safety system."
)
_MODIFICATION_MESSAGE = "Modify the prompt or input and try again."
_BILLED_NOTICE = "This generation was charged because the provider processed the request."
_OUTPUT_NOT_DELIVERED_MESSAGE = (
    "The AI provider generated the content but could not deliver it. "
    "Contact support with this job ID if the charge is unexpected."
)
_PUBLIC_MESSAGES: dict[ProviderFailureKind, str] = {
    ProviderFailureKind.MODERATION_REJECTED: (
        f"{_MODERATION_REJECTION_MESSAGE} {_MODIFICATION_MESSAGE}"
    ),
    ProviderFailureKind.INVALID_REQUEST: "The AI provider could not process this request.",
    ProviderFailureKind.RATE_LIMITED: "The AI provider is temporarily rate limited.",
    ProviderFailureKind.AUTHENTICATION_FAILED: "The AI provider is unavailable.",
    ProviderFailureKind.PROVIDER_UNAVAILABLE: "The AI provider is temporarily unavailable.",
    ProviderFailureKind.TIMEOUT: "The AI provider timed out while processing the request.",
    ProviderFailureKind.MALFORMED_RESPONSE: "The AI provider returned an invalid response.",
    ProviderFailureKind.OUTPUT_NOT_DELIVERED: _OUTPUT_NOT_DELIVERED_MESSAGE,
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
    ProviderFailureKind.OUTPUT_NOT_DELIVERED: "provider_output_not_delivered",
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
        return self.public_code_for_kind(self.kind)

    @classmethod
    def public_code_for_kind(cls, kind: ProviderFailureKind) -> str:
        """Return the stable client-visible code for a normalized kind."""
        return _PUBLIC_CODES[kind]

    @classmethod
    def safe_message_for_kind(cls, kind: ProviderFailureKind) -> str:
        """Return the client-safe message for a normalized failure kind."""
        return _PUBLIC_MESSAGES[kind]

    @classmethod
    def public_message_for_failure(cls, failure: ProviderFailure) -> str:
        """Return the safe message after the resolved billing policy is known.

        Composed explicitly from the same literals ``_PUBLIC_MESSAGES`` draws
        from, rather than by substring surgery on the base message: a
        ``str.replace`` against a needle that later drifts out of sync with
        the catalog entry fails silently, dropping the billing disclosure
        with no error. See invariant 12 — every billable kind must disclose.
        """
        if not failure.billable:
            return cls.safe_message_for_kind(failure.kind)
        if failure.kind is ProviderFailureKind.MODERATION_REJECTED:
            return f"{_MODERATION_REJECTION_MESSAGE} {_BILLED_NOTICE} {_MODIFICATION_MESSAGE}"
        if failure.kind is ProviderFailureKind.OUTPUT_NOT_DELIVERED:
            return f"{cls.safe_message_for_kind(failure.kind)} {_BILLED_NOTICE}"
        return cls.safe_message_for_kind(failure.kind)


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
        self.public_message = ProviderFailure.public_message_for_failure(failure)
        super().__init__(self.public_message)


class ProviderModerationRejectedError(ProviderSubmissionFailedError):
    """A normalized provider moderation rejection with submission context.

    The original provider message remains diagnostic-only and is never made
    part of the exception text, persisted job fields, or API response.
    """

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
