"""Unit tests for PaymentCurrencySyncService."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.services.billing_errors import LogoCacheError, LogoStorageError
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
    assert entries[0].has_metadata is False
    repo.sync_catalog.assert_awaited_once()


async def test_ticker_present_in_details_marks_has_metadata_true() -> None:
    details = {"BTC": CurrencyDetails(ticker="BTC", name="Bitcoin", network="BTC", logo_url=None)}
    gateway = _FakeCatalogGateway(selected=["BTC"], details=details)
    registry = GatewayRegistry([gateway])  # type: ignore[list-item]
    service = PaymentCurrencySyncService(registry=registry, logo_cache=AsyncMock())

    repo = _make_repo()
    with patch(
        "src.api.services.payment_currency_sync.PaymentCurrencyRepository", return_value=repo
    ):
        await service.refresh(VEX_CONFIG, session=MagicMock())

    entries = repo.sync_catalog.call_args.args[2]
    assert entries[0].has_metadata is True


async def test_logo_cache_none_syncs_without_any_logo_calls() -> None:
    details = {
        "BTC": CurrencyDetails(
            ticker="BTC", name="Bitcoin", network="BTC", logo_url="https://nowpayments.io/btc.svg"
        )
    }
    gateway = _FakeCatalogGateway(selected=["BTC"], details=details)
    registry = GatewayRegistry([gateway])  # type: ignore[list-item]
    service = PaymentCurrencySyncService(registry=registry, logo_cache=None)

    repo = _make_repo()
    with patch(
        "src.api.services.payment_currency_sync.PaymentCurrencyRepository", return_value=repo
    ):
        results = await service.refresh(VEX_CONFIG, session=MagicMock())

    entries = repo.sync_catalog.call_args.args[2]
    assert entries[0].logo_key is None
    assert entries[0].logo_source_url is None
    assert entries[0].has_metadata is True
    assert results[0].provider == PaymentProvider.NOWPAYMENTS
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


def _three_ticker_details() -> dict[str, CurrencyDetails]:
    return {
        ticker: CurrencyDetails(
            ticker=ticker,
            name=ticker.title(),
            network=ticker,
            logo_url=f"https://nowpayments.io/{ticker.lower()}.svg",
        )
        for ticker in ("BTC", "ETH", "LTC")
    }


async def test_storage_failure_short_circuits_remaining_tickers() -> None:
    """P1-2: once a LogoStorageError is seen, later tickers skip ensure_cached entirely."""
    details = _three_ticker_details()
    gateway = _FakeCatalogGateway(selected=["BTC", "ETH", "LTC"], details=details)
    registry = GatewayRegistry([gateway])  # type: ignore[list-item]
    logo_cache = AsyncMock()
    logo_cache.ensure_cached = AsyncMock(side_effect=LogoStorageError("R2 storage failure"))
    service = PaymentCurrencySyncService(registry=registry, logo_cache=logo_cache)

    repo = _make_repo()
    with (
        patch(
            "src.api.services.payment_currency_sync.PaymentCurrencyRepository", return_value=repo
        ),
        patch("src.api.services.payment_currency_sync.logger") as mock_logger,
    ):
        results = await service.refresh(VEX_CONFIG, session=MagicMock())

    logo_cache.ensure_cached.assert_awaited_once()
    entries = repo.sync_catalog.call_args.args[2]
    assert [entry.ticker for entry in entries] == ["BTC", "ETH", "LTC"]
    assert all(entry.logo_key is None for entry in entries)
    assert all(entry.has_metadata is True for entry in entries)
    repo.sync_catalog.assert_awaited_once()
    assert results[0].provider == PaymentProvider.NOWPAYMENTS

    error_calls = [
        call
        for call in mock_logger.exception.call_args_list
        if call.args and call.args[0] == "payment_currency.logo_storage_unavailable"
    ]
    assert len(error_calls) == 1

    sync_ok_call = next(
        call
        for call in mock_logger.info.call_args_list
        if call.args[0] == "payment_currency.sync_ok"
    )
    assert sync_ok_call.kwargs["logos_skipped_storage"] == 3


async def test_download_caused_failures_keep_attempting_per_ticker() -> None:
    """Regression check: plain (non-storage) LogoCacheError does NOT short-circuit."""
    details = _three_ticker_details()
    gateway = _FakeCatalogGateway(selected=["BTC", "ETH", "LTC"], details=details)
    registry = GatewayRegistry([gateway])  # type: ignore[list-item]
    logo_cache = AsyncMock()
    logo_cache.ensure_cached = AsyncMock(side_effect=LogoCacheError("bad content-type"))
    service = PaymentCurrencySyncService(registry=registry, logo_cache=logo_cache)

    repo = _make_repo()
    with patch(
        "src.api.services.payment_currency_sync.PaymentCurrencyRepository", return_value=repo
    ):
        await service.refresh(VEX_CONFIG, session=MagicMock())

    assert logo_cache.ensure_cached.await_count == 3
