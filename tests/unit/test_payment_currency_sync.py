"""Unit tests for PaymentCurrencySyncService."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.services.billing_errors import LogoCacheError
from src.api.services.payment_currency_sync import PaymentCurrencySyncService
from src.api.services.payments.catalog import CurrencyDetails
from src.api.services.payments.registry import GatewayRegistry
from src.core.product import PaymentProvider
from src.core.product_registry import SYNTHARA_CONFIG, VEX_CONFIG

pytestmark = pytest.mark.unit


@dataclass
class _Row:
    ticker: str
    logo_key: str | None = None
    logo_source_url: str | None = None


class _FakeCatalogGateway:
    """Structurally satisfies SupportsCurrencyCatalog for the sync service."""

    provider = PaymentProvider.NOWPAYMENTS

    def __init__(
        self,
        *,
        selected: list[str],
        details: dict[str, CurrencyDetails],
    ) -> None:
        self.list_merchant_currencies = AsyncMock(return_value=selected)
        self.list_full_currencies = AsyncMock(return_value=details)


def _make_repo(existing_rows: list[_Row] | None = None) -> AsyncMock:
    repo = AsyncMock()
    repo.list_currencies = AsyncMock(return_value=existing_rows or [])
    repo.sync_catalog = AsyncMock(return_value=(1, 0))
    return repo


async def test_skips_products_without_a_capable_provider() -> None:
    """SYNTHARA_CONFIG's capability set has no NowPayments — nothing to sync."""
    gateway = _FakeCatalogGateway(selected=["BTC"], details={})
    registry = GatewayRegistry([gateway])  # type: ignore[list-item]
    service = PaymentCurrencySyncService(registry=registry, logo_cache=AsyncMock())

    results = await service.refresh(SYNTHARA_CONFIG, session=MagicMock())

    assert results == []
    gateway.list_merchant_currencies.assert_not_awaited()


async def test_unchanged_logo_triggers_zero_downloads() -> None:
    details = {
        "BTC": CurrencyDetails(
            ticker="BTC", name="Bitcoin", network="BTC", logo_url="https://nowpayments.io/btc.svg"
        )
    }
    gateway = _FakeCatalogGateway(selected=["BTC"], details=details)
    registry = GatewayRegistry([gateway])  # type: ignore[list-item]
    logo_cache = AsyncMock()
    service = PaymentCurrencySyncService(registry=registry, logo_cache=logo_cache)

    existing = [
        _Row(ticker="BTC", logo_key="abc123.svg", logo_source_url="https://nowpayments.io/btc.svg")
    ]
    repo = _make_repo(existing)
    with patch(
        "src.api.services.payment_currency_sync.PaymentCurrencyRepository", return_value=repo
    ):
        await service.refresh(VEX_CONFIG, session=MagicMock())

    logo_cache.ensure_cached.assert_not_awaited()
    entries = repo.sync_catalog.call_args.args[2]
    assert entries[0].logo_key is None  # repo preserves the existing key untouched


async def test_changed_logo_url_triggers_refetch() -> None:
    details = {
        "BTC": CurrencyDetails(
            ticker="BTC", name="Bitcoin", network="BTC", logo_url="https://nowpayments.io/new.svg"
        )
    }
    gateway = _FakeCatalogGateway(selected=["BTC"], details=details)
    registry = GatewayRegistry([gateway])  # type: ignore[list-item]
    logo_cache = AsyncMock()
    logo_cache.ensure_cached = AsyncMock(return_value="newkey.svg")
    service = PaymentCurrencySyncService(registry=registry, logo_cache=logo_cache)

    existing = [
        _Row(ticker="BTC", logo_key="oldkey.svg", logo_source_url="https://nowpayments.io/old.svg")
    ]
    repo = _make_repo(existing)
    with patch(
        "src.api.services.payment_currency_sync.PaymentCurrencyRepository", return_value=repo
    ):
        await service.refresh(VEX_CONFIG, session=MagicMock())

    logo_cache.ensure_cached.assert_awaited_once_with("https://nowpayments.io/new.svg")
    entries = repo.sync_catalog.call_args.args[2]
    assert entries[0].logo_key == "newkey.svg"
    assert entries[0].logo_source_url == "https://nowpayments.io/new.svg"


async def test_logo_failure_leaves_entry_without_logo_fields() -> None:
    details = {
        "BTC": CurrencyDetails(
            ticker="BTC", name="Bitcoin", network="BTC", logo_url="https://nowpayments.io/btc.svg"
        )
    }
    gateway = _FakeCatalogGateway(selected=["BTC"], details=details)
    registry = GatewayRegistry([gateway])  # type: ignore[list-item]
    logo_cache = AsyncMock()
    logo_cache.ensure_cached = AsyncMock(side_effect=LogoCacheError("boom"))
    service = PaymentCurrencySyncService(registry=registry, logo_cache=logo_cache)

    repo = _make_repo()
    with patch(
        "src.api.services.payment_currency_sync.PaymentCurrencyRepository", return_value=repo
    ):
        results = await service.refresh(VEX_CONFIG, session=MagicMock())

    entries = repo.sync_catalog.call_args.args[2]
    assert entries[0].ticker == "BTC"
    assert entries[0].logo_key is None
    assert entries[0].logo_source_url is None
    assert results[0].provider == PaymentProvider.NOWPAYMENTS


async def test_metadata_missing_ticker_stays_available_with_null_metadata() -> None:
    gateway = _FakeCatalogGateway(selected=["DOGE"], details={})
    registry = GatewayRegistry([gateway])  # type: ignore[list-item]
    service = PaymentCurrencySyncService(registry=registry, logo_cache=AsyncMock())

    repo = _make_repo()
    with patch(
        "src.api.services.payment_currency_sync.PaymentCurrencyRepository", return_value=repo
    ):
        await service.refresh(VEX_CONFIG, session=MagicMock())

    entries = repo.sync_catalog.call_args.args[2]
    assert entries[0].ticker == "DOGE"
    assert entries[0].name is None
    assert entries[0].network is None
    repo.sync_catalog.assert_awaited_once()


async def test_full_currencies_failure_aborts_before_sync_catalog() -> None:
    gateway = _FakeCatalogGateway(selected=["BTC"], details={})
    gateway.list_full_currencies = AsyncMock(side_effect=RuntimeError("provider down"))
    registry = GatewayRegistry([gateway])  # type: ignore[list-item]
    service = PaymentCurrencySyncService(registry=registry, logo_cache=AsyncMock())

    repo = _make_repo()
    with (
        patch(
            "src.api.services.payment_currency_sync.PaymentCurrencyRepository", return_value=repo
        ),
        pytest.raises(RuntimeError, match="provider down"),
    ):
        await service.refresh(VEX_CONFIG, session=MagicMock())

    repo.sync_catalog.assert_not_awaited()


async def test_merchant_currencies_failure_aborts_before_sync_catalog() -> None:
    gateway = _FakeCatalogGateway(selected=[], details={})
    gateway.list_merchant_currencies = AsyncMock(side_effect=RuntimeError("provider down"))
    registry = GatewayRegistry([gateway])  # type: ignore[list-item]
    service = PaymentCurrencySyncService(registry=registry, logo_cache=AsyncMock())

    repo = _make_repo()
    with (
        patch(
            "src.api.services.payment_currency_sync.PaymentCurrencyRepository", return_value=repo
        ),
        pytest.raises(RuntimeError, match="provider down"),
    ):
        await service.refresh(VEX_CONFIG, session=MagicMock())

    repo.sync_catalog.assert_not_awaited()
