"""Vast.ai API client package."""

from .client import VastAIClient
from .exceptions import (
    InstanceNotFoundError,
    NoCapacityError,
    OfferTakenError,
    VastAIError,
    VastAIPaymentError,
    VastAIRateLimitError,
)

__all__ = [
    "InstanceNotFoundError",
    "NoCapacityError",
    "OfferTakenError",
    "VastAIClient",
    "VastAIError",
    "VastAIPaymentError",
    "VastAIRateLimitError",
]
