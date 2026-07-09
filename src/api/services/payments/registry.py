"""Payment gateway registry keyed by the canonical provider enum."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.services.billing_errors import UnsupportedProviderError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.api.services.payments.protocol import PaymentGateway
    from src.core.product import PaymentProvider


class GatewayRegistry:
    """Resolve gateway implementations and reject ambiguous registrations."""

    def __init__(self, gateways: Iterable[PaymentGateway]) -> None:
        self._gateways: dict[PaymentProvider, PaymentGateway] = {}
        for gateway in gateways:
            if gateway.provider in self._gateways:
                raise ValueError(f"Duplicate payment gateway: {gateway.provider.value}")
            self._gateways[gateway.provider] = gateway

    def get(self, provider: PaymentProvider) -> PaymentGateway:
        try:
            return self._gateways[provider]
        except KeyError as exc:
            raise UnsupportedProviderError(provider) from exc

    @property
    def providers(self) -> frozenset[PaymentProvider]:
        """Registered providers, exposed for startup completeness checks."""

        return frozenset(self._gateways)
