"""Extensible payment gateway package."""

from src.api.services.payments.contracts import (
    ChargeContext,
    ChargeResult,
    CreatedCharge,
    PaymentLookup,
    WebhookEnvelope,
    WebhookOutcome,
)
from src.api.services.payments.protocol import PaymentGateway
from src.api.services.payments.registry import GatewayRegistry
from src.api.services.payments.service import PaymentService

__all__ = [
    "ChargeContext",
    "ChargeResult",
    "CreatedCharge",
    "GatewayRegistry",
    "PaymentGateway",
    "PaymentLookup",
    "PaymentService",
    "WebhookEnvelope",
    "WebhookOutcome",
]
