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
        if self.kind == ProviderFailureKind.MODERATION_REJECTED:
            return "provider_moderation_rejected"
        return f"provider_{self.kind.value}"

    @classmethod
    def safe_message_for_kind(cls, kind: ProviderFailureKind) -> str:
        """Return the client-safe message for a normalized failure kind."""
        return _PUBLIC_MESSAGES[kind]


class ProviderModerationRejectedError(Exception):
    """A billable provider moderation rejection with committed job context.

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
        balance_event: BalanceEvent | None,
    ) -> None:
        if failure.kind != ProviderFailureKind.MODERATION_REJECTED:
            raise ValueError("ProviderModerationRejectedError requires a moderation failure")
        self.failure = failure
        self.job_id = job_id
        self.balance_event = balance_event
        super().__init__(self.public_message)
