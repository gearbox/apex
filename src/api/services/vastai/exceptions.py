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
