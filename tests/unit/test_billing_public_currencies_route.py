"""Unit tests for GET /v1/billing/currencies."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.billing_public import BillingPublicController
from src.core.config import Settings
from src.core.product_registry import VEX_CONFIG

pytestmark = pytest.mark.unit


@dataclass
class _Row:
    ticker: str
    name: str | None = None
    network: str | None = None
    logo_key: str | None = None


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "jwt_secret_key": "test-secret-key-that-is-definitely-long-enough-32bytes",
    } | overrides
    return Settings(**defaults)  # type: ignore[arg-type]


async def test_empty_catalog_returns_empty_list() -> None:
    repo = AsyncMock()
    repo.list_currencies = AsyncMock(return_value=[])
    with patch("src.api.routes.billing_public.PaymentCurrencyRepository", return_value=repo):
        result = await BillingPublicController.list_currencies.fn(
            MagicMock(),
            session=AsyncMock(),
            product_config=VEX_CONFIG,
            settings=_settings(r2_public_url_base="https://assets.vex.pics"),
        )
    assert result == []


async def test_logo_url_built_from_assets_base_and_key() -> None:
    repo = AsyncMock()
    repo.list_currencies = AsyncMock(
        return_value=[_Row(ticker="BTC", name="Bitcoin", network="BTC", logo_key="abc123.svg")]
    )
    with patch("src.api.routes.billing_public.PaymentCurrencyRepository", return_value=repo):
        result = await BillingPublicController.list_currencies.fn(
            MagicMock(),
            session=AsyncMock(),
            product_config=VEX_CONFIG,
            settings=_settings(r2_public_url_base="https://assets.vex.pics/"),
        )
    assert result[0].logo_url == "https://assets.vex.pics/abc123.svg"
    assert "nowpayments.io" not in (result[0].logo_url or "")


async def test_null_logo_key_yields_null_logo_url() -> None:
    repo = AsyncMock()
    repo.list_currencies = AsyncMock(return_value=[_Row(ticker="BTC", logo_key=None)])
    with patch("src.api.routes.billing_public.PaymentCurrencyRepository", return_value=repo):
        result = await BillingPublicController.list_currencies.fn(
            MagicMock(),
            session=AsyncMock(),
            product_config=VEX_CONFIG,
            settings=_settings(r2_public_url_base="https://assets.vex.pics"),
        )
    assert result[0].logo_url is None


async def test_unconfigured_assets_base_yields_null_logo_url() -> None:
    """Degrades to null rather than crashing when R2_PUBLIC_URL_BASE is unset (dev)."""
    repo = AsyncMock()
    repo.list_currencies = AsyncMock(return_value=[_Row(ticker="BTC", logo_key="abc123.svg")])
    with patch("src.api.routes.billing_public.PaymentCurrencyRepository", return_value=repo):
        result = await BillingPublicController.list_currencies.fn(
            MagicMock(),
            session=AsyncMock(),
            product_config=VEX_CONFIG,
            settings=_settings(r2_public_url_base=None),
        )
    assert result[0].logo_url is None


async def test_queries_only_available_rows() -> None:
    repo = AsyncMock()
    repo.list_currencies = AsyncMock(return_value=[])
    with patch("src.api.routes.billing_public.PaymentCurrencyRepository", return_value=repo):
        await BillingPublicController.list_currencies.fn(
            MagicMock(),
            session=AsyncMock(),
            product_config=VEX_CONFIG,
            settings=_settings(),
        )
    repo.list_currencies.assert_awaited_once_with(VEX_CONFIG.slug, only_available=True)
