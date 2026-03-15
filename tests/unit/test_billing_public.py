"""Tests for public billing endpoints (no auth required)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from src.api.routes.billing_public import (
    PublicBillingController,
    PublicPricingResponse,
    TokenPackageResponse,
    _build_packages,
)
from src.core.config import TOKEN_PACKAGES

# ---------------------------------------------------------------------------
# _build_packages (pure function)
# ---------------------------------------------------------------------------


class TestBuildPackages:
    def test_returns_all_configured_packages(self) -> None:
        packages = _build_packages()
        assert len(packages) == len(TOKEN_PACKAGES)

    def test_sorted_by_price_ascending(self) -> None:
        packages = _build_packages()
        prices = [Decimal(p.price_usd) for p in packages]
        assert prices == sorted(prices)

    def test_price_usd_is_decimal_string(self) -> None:
        packages = _build_packages()
        for p in packages:
            assert isinstance(p.price_usd, str)
            Decimal(p.price_usd)  # must not raise

    def test_starter_package_fields(self) -> None:
        packages = {p.id: p for p in _build_packages()}
        starter = packages["starter"]
        assert starter.name == "Starter"
        assert starter.tokens == 1_000
        assert starter.price_usd == "9.99"
        assert starter.bonus_pct == 0

    def test_bonus_pct_pro(self) -> None:
        # Pro: 1_500 bonus / 15_000 tokens = 10 %
        packages = {p.id: p for p in _build_packages()}
        assert packages["pro"].bonus_pct == 10

    def test_bonus_pct_enterprise(self) -> None:
        # Enterprise: 10_000 bonus / 50_000 tokens = 20 %
        packages = {p.id: p for p in _build_packages()}
        assert packages["enterprise"].bonus_pct == 20

    def test_no_bonus_for_basic(self) -> None:
        packages = {p.id: p for p in _build_packages()}
        assert packages["basic"].bonus_pct == 0

    def test_returns_token_package_response_instances(self) -> None:
        for p in _build_packages():
            assert isinstance(p, TokenPackageResponse)

    def test_package_ids_match_config(self) -> None:
        packages = {p.id: p for p in _build_packages()}
        assert set(packages.keys()) == set(TOKEN_PACKAGES.keys())


# ---------------------------------------------------------------------------
# PublicBillingController.list_packages
# ---------------------------------------------------------------------------


class TestListPackages:
    async def test_returns_list_of_packages(self) -> None:
        result = await PublicBillingController.list_packages.fn(MagicMock())
        assert len(result) == len(TOKEN_PACKAGES)

    async def test_all_items_are_token_package_responses(self) -> None:
        result = await PublicBillingController.list_packages.fn(MagicMock())
        assert all(isinstance(p, TokenPackageResponse) for p in result)

    async def test_packages_ordered_by_price(self) -> None:
        result = await PublicBillingController.list_packages.fn(MagicMock())
        prices = [Decimal(p.price_usd) for p in result]
        assert prices == sorted(prices)


# ---------------------------------------------------------------------------
# PublicBillingController.get_pricing
# ---------------------------------------------------------------------------


class TestGetPricing:
    def _make_session(self, rules: list) -> AsyncMock:
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = rules
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        return session

    async def test_returns_public_pricing_response(self) -> None:
        session = self._make_session([])
        result = await PublicBillingController.get_pricing.fn(MagicMock(), session)
        assert isinstance(result, PublicPricingResponse)

    async def test_packages_included_in_response(self) -> None:
        session = self._make_session([])
        result = await PublicBillingController.get_pricing.fn(MagicMock(), session)
        assert len(result.packages) == len(TOKEN_PACKAGES)

    async def test_empty_pricing_rules(self) -> None:
        session = self._make_session([])
        result = await PublicBillingController.get_pricing.fn(MagicMock(), session)
        assert result.prices == []

    async def test_single_pricing_rule_mapped(self) -> None:
        rule = MagicMock()
        rule.provider = "grok"
        rule.generation_type = "t2i"
        rule.model = "grok-imagine-image"
        rule.token_cost = 50

        session = self._make_session([rule])
        result = await PublicBillingController.get_pricing.fn(MagicMock(), session)

        assert len(result.prices) == 1
        price = result.prices[0]
        assert price.provider == "grok"
        assert price.generation_type == "t2i"
        assert price.model == "grok-imagine-image"
        assert price.token_cost == 50

    async def test_multiple_pricing_rules(self) -> None:
        rules = [
            MagicMock(provider="grok", generation_type="t2i", model="m1", token_cost=50),
            MagicMock(provider="aisha", generation_type="i2i", model=None, token_cost=20),
        ]
        session = self._make_session(rules)
        result = await PublicBillingController.get_pricing.fn(MagicMock(), session)

        assert len(result.prices) == 2
        providers = {p.provider for p in result.prices}
        assert providers == {"grok", "aisha"}

    async def test_rule_with_no_model(self) -> None:
        rule = MagicMock()
        rule.provider = "aisha"
        rule.generation_type = "t2i"
        rule.model = None
        rule.token_cost = 10

        session = self._make_session([rule])
        result = await PublicBillingController.get_pricing.fn(MagicMock(), session)

        assert result.prices[0].model is None

    async def test_session_execute_called_once(self) -> None:
        session = self._make_session([])
        await PublicBillingController.get_pricing.fn(MagicMock(), session)
        session.execute.assert_awaited_once()
