"""Structural contract for provider-side currency catalog discovery.

Kept separate from ``protocol.py``'s minimal ``PaymentGateway`` (D4): catalog
support is optional per-provider, and a second narrow protocol lets the sync
service discover capable gateways via ``isinstance`` without forcing every
gateway to stub out unsupported methods or branching on provider identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import msgspec

if TYPE_CHECKING:
    from src.core.product import PaymentProvider


class CurrencyDetails(msgspec.Struct, frozen=True, kw_only=True):
    """Provider-reported display metadata for one currency ticker."""

    ticker: str
    """Uppercased provider ticker code."""
    name: str | None
    network: str | None
    """Uppercased provider network code, or None when absent."""
    logo_url: str | None
    """Absolute URL, resolved by the gateway (see D11 in the currency-catalog design)."""


@runtime_checkable
class SupportsCurrencyCatalog(Protocol):
    """Optional capability: a gateway that can enumerate its currency catalog.

    Gateways implementing this never touch the DB or R2 (standing gateway
    invariant) — they only translate the provider's own discovery endpoints.
    """

    provider: PaymentProvider

    async def list_merchant_currencies(self, product_id: str) -> list[str]:
        """Return the uppercased, deduplicated tickers enabled in the dashboard."""
        ...

    async def list_full_currencies(self, product_id: str) -> dict[str, CurrencyDetails]:
        """Return the provider's full currency universe, keyed by uppercased ticker."""
        ...
