"""NowPayments currency-catalog (merchant/coins + full-currencies) parsing tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.services.billing_errors import PaymentCatalogError
from src.api.services.payments.nowpayments_gateway import NowPaymentsGateway
from src.core.config import Settings

pytestmark = pytest.mark.unit


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "jwt_secret_key": "test-secret-key-that-is-definitely-long-enough-32bytes",
        "nowpayments_api_key_vex": "np_key_vex_123",
    } | overrides
    return Settings(**defaults)  # type: ignore[arg-type]


def _client_returning(payload: Any, *, status_ok: bool = True) -> AsyncMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    if not status_ok:
        import httpx

        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=MagicMock()
        )
    response.json = MagicMock(return_value=payload)

    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class TestListMerchantCurrencies:
    async def test_uppercases_strips_and_dedupes(self) -> None:
        client = _client_returning({"selectedCurrencies": ["btc", " eth ", "BTC", "usdcmatic"]})
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        result = await gateway.list_merchant_currencies("vex")

        assert result == ["BTC", "ETH", "USDCMATIC"]

    async def test_non_2xx_propagates(self) -> None:
        import httpx

        client = _client_returning({}, status_ok=False)
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        with pytest.raises(httpx.HTTPStatusError):
            await gateway.list_merchant_currencies("vex")

    async def test_missing_key_raises_catalog_error(self) -> None:
        client = _client_returning({"unexpected": []})
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        with pytest.raises(PaymentCatalogError, match="selectedCurrencies"):
            await gateway.list_merchant_currencies("vex")

    async def test_non_list_shape_raises_catalog_error(self) -> None:
        client = _client_returning({"selectedCurrencies": "btc"})
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        with pytest.raises(PaymentCatalogError, match="selectedCurrencies"):
            await gateway.list_merchant_currencies("vex")

    async def test_non_string_items_raise_catalog_error(self) -> None:
        client = _client_returning({"selectedCurrencies": ["btc", 123]})
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        with pytest.raises(PaymentCatalogError, match="selectedCurrencies"):
            await gateway.list_merchant_currencies("vex")

    async def test_non_object_payload_raises_catalog_error(self) -> None:
        client = _client_returning(["not", "an", "object"])
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        with pytest.raises(PaymentCatalogError, match="JSON object"):
            await gateway.list_merchant_currencies("vex")


class TestListFullCurrencies:
    async def test_builds_dict_keyed_by_uppercased_ticker(self) -> None:
        client = _client_returning(
            {
                "currencies": [
                    {
                        "code": "btc",
                        "name": "Bitcoin",
                        "network": "btc",
                        "logo_url": "https://nowpayments.io/images/coins/btc.svg",
                    }
                ]
            }
        )
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        details = await gateway.list_full_currencies("vex")

        assert set(details) == {"BTC"}
        entry = details["BTC"]
        assert entry.ticker == "BTC"
        assert entry.name == "Bitcoin"
        assert entry.network == "BTC"
        assert entry.logo_url == "https://nowpayments.io/images/coins/btc.svg"

    async def test_missing_top_level_key_raises(self) -> None:
        client = _client_returning({"nope": []})
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        with pytest.raises(PaymentCatalogError, match="currencies"):
            await gateway.list_full_currencies("vex")

    async def test_non_list_currencies_raises(self) -> None:
        client = _client_returning({"currencies": "btc"})
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        with pytest.raises(PaymentCatalogError, match="currencies"):
            await gateway.list_full_currencies("vex")

    async def test_absent_per_item_fields_tolerated(self) -> None:
        client = _client_returning({"currencies": [{"code": "btc"}]})
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        details = await gateway.list_full_currencies("vex")

        entry = details["BTC"]
        assert entry.name is None
        assert entry.network is None
        assert entry.logo_url is None

    async def test_entry_missing_code_is_skipped(self) -> None:
        client = _client_returning({"currencies": [{"name": "No code"}, {"code": "eth"}]})
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        details = await gateway.list_full_currencies("vex")

        assert set(details) == {"ETH"}

    async def test_relative_logo_url_resolved_against_nowpayments_site(self) -> None:
        client = _client_returning(
            {"currencies": [{"code": "btc", "logo_url": "/images/coins/btc.svg"}]}
        )
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        details = await gateway.list_full_currencies("vex")

        assert details["BTC"].logo_url == "https://nowpayments.io/images/coins/btc.svg"

    async def test_disallowed_absolute_host_raises(self) -> None:
        client = _client_returning(
            {"currencies": [{"code": "btc", "logo_url": "https://evil.example.com/btc.svg"}]}
        )
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        with pytest.raises(PaymentCatalogError, match="Disallowed"):
            await gateway.list_full_currencies("vex")

    async def test_allowed_www_subdomain_host(self) -> None:
        client = _client_returning(
            {
                "currencies": [
                    {"code": "btc", "logo_url": "https://www.nowpayments.io/images/coins/btc.svg"}
                ]
            }
        )
        gateway = NowPaymentsGateway(_settings(), client_factory=lambda: client)

        details = await gateway.list_full_currencies("vex")

        assert details["BTC"].logo_url == "https://www.nowpayments.io/images/coins/btc.svg"
