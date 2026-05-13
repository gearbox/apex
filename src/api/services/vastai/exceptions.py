"""Vast.ai API exception hierarchy."""

from __future__ import annotations


class VastAIError(Exception):
    """Base exception for Vast.ai API errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class NoCapacityError(VastAIError):
    """No GPU offers match the requested hardware requirements."""


class VastAIPaymentError(VastAIError):
    """Vast.ai account has insufficient credits or billing issue."""


class InstanceNotFoundError(VastAIError):
    """Vast.ai instance does not exist."""


class OfferTakenError(VastAIError):
    """The selected offer was rented by someone else between search and create."""


class VastAIRateLimitError(VastAIError):
    """Raised when Vast.ai returns 429 and our retry budget is exhausted.

    The retry handler in VastAIClient honors Retry-After up to
    settings.vastai_max_429_retries. If that's exceeded, this error
    surfaces to callers so they can decide their own policy.
    """

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after_seconds = retry_after_seconds
