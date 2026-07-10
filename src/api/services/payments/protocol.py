"""Structural contract implemented by every payment gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.api.services.payments.contracts import (
        ChargeContext,
        ChargeResult,
        WebhookEnvelope,
        WebhookOutcome,
    )
    from src.core.product import PaymentProvider


@runtime_checkable
class PaymentGateway(Protocol):
    """Pure provider translator; gateways never access persistence or billing."""

    provider: PaymentProvider

    async def create_charge(self, ctx: ChargeContext) -> ChargeResult: ...

    async def verify_webhook(self, envelope: WebhookEnvelope) -> WebhookOutcome: ...
