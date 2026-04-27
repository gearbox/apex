"""Unit tests for PricingService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.services.billing_errors import PriceNotFoundError
from src.api.services.pricing import PricingService

pytestmark = pytest.mark.unit


def _make_rule(**kwargs: object) -> MagicMock:
    rule = MagicMock()
    rule.id = kwargs.get("id", uuid4())
    rule.provider = kwargs.get("provider", "aisha")
    rule.generation_type = kwargs.get("generation_type", "t2i")
    rule.model = kwargs.get("model")
    rule.token_cost = kwargs.get("token_cost", 100)
    rule.is_active = kwargs.get("is_active", True)
    rule.notes = kwargs.get("notes")
    return rule


def _make_repo(**kwargs: object) -> AsyncMock:
    repo = AsyncMock()
    repo.get_active_price = AsyncMock(return_value=kwargs.get("price_rule"))
    repo.list_pricing_rules = AsyncMock(return_value=kwargs.get("rules", []))
    repo.create_pricing_rule = AsyncMock(return_value=_make_rule())
    repo.get_pricing_rule = AsyncMock(return_value=kwargs.get("rule"))
    return repo


class TestGetPrice:
    async def test_returns_token_cost_when_rule_found(self) -> None:
        rule = _make_rule(token_cost=200)
        repo = _make_repo(price_rule=rule)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.pricing.BillingRepository", lambda _: repo)
            svc = PricingService()
            cost = await svc.get_price("aisha", "t2i", None, session=session)

        assert cost == 200

    async def test_raises_when_no_rule_found(self) -> None:
        repo = _make_repo(price_rule=None)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.pricing.BillingRepository", lambda _: repo)
            svc = PricingService()
            with pytest.raises(PriceNotFoundError):
                await svc.get_price("aisha", "t2i", None, session=session)


class TestListCatalog:
    async def test_returns_all_active_rules_by_default(self) -> None:
        rules = [_make_rule(), _make_rule()]
        repo = _make_repo(rules=rules)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.pricing.BillingRepository", lambda _: repo)
            svc = PricingService()
            result = await svc.list_catalog(session=session)

        assert list(result) == rules
        repo.list_pricing_rules.assert_awaited_once_with(active_only=True)

    async def test_passes_active_only_false(self) -> None:
        repo = _make_repo(rules=[])
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.pricing.BillingRepository", lambda _: repo)
            svc = PricingService()
            await svc.list_catalog(active_only=False, session=session)

        repo.list_pricing_rules.assert_awaited_once_with(active_only=False)


class TestCreateRule:
    async def test_creates_and_returns_rule(self) -> None:
        new_rule = _make_rule(token_cost=50)
        repo = _make_repo()
        repo.create_pricing_rule = AsyncMock(return_value=new_rule)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.pricing.BillingRepository", lambda _: repo)
            svc = PricingService()
            result = await svc.create_rule(
                provider="aisha",
                generation_type="t2i",
                model="wan",
                token_cost=50,
                notes="test",
                admin_id=uuid4(),
                session=session,
            )

        assert result is new_rule
        repo.create_pricing_rule.assert_awaited_once()


class TestDeactivateRule:
    async def test_sets_is_active_false_and_flushes(self) -> None:
        rule = _make_rule(is_active=True)
        repo = _make_repo(rule=rule)
        session = AsyncMock()
        session.flush = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.pricing.BillingRepository", lambda _: repo)
            svc = PricingService()
            await svc.deactivate_rule(rule.id, session=session)

        assert rule.is_active is False
        session.flush.assert_awaited_once()

    async def test_raises_when_rule_not_found(self) -> None:
        repo = _make_repo(rule=None)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.pricing.BillingRepository", lambda _: repo)
            svc = PricingService()
            with pytest.raises(PriceNotFoundError):
                await svc.deactivate_rule(uuid4(), session=session)


class TestUpdateRule:
    async def test_updates_token_cost(self) -> None:
        rule = _make_rule(token_cost=100)
        repo = _make_repo(rule=rule)
        session = AsyncMock()
        session.flush = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.pricing.BillingRepository", lambda _: repo)
            svc = PricingService()
            result = await svc.update_rule(rule.id, token_cost=200, session=session)

        assert rule.token_cost == 200
        assert result is rule

    async def test_updates_is_active(self) -> None:
        rule = _make_rule(is_active=True)
        repo = _make_repo(rule=rule)
        session = AsyncMock()
        session.flush = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.pricing.BillingRepository", lambda _: repo)
            svc = PricingService()
            await svc.update_rule(rule.id, is_active=False, session=session)

        assert rule.is_active is False

    async def test_updates_notes(self) -> None:
        rule = _make_rule(notes=None)
        repo = _make_repo(rule=rule)
        session = AsyncMock()
        session.flush = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.pricing.BillingRepository", lambda _: repo)
            svc = PricingService()
            await svc.update_rule(rule.id, notes="new note", session=session)

        assert rule.notes == "new note"

    async def test_raises_when_rule_not_found(self) -> None:
        repo = _make_repo(rule=None)
        session = AsyncMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.pricing.BillingRepository", lambda _: repo)
            svc = PricingService()
            with pytest.raises(PriceNotFoundError):
                await svc.update_rule(uuid4(), token_cost=999, session=session)

    async def test_updates_effective_until(self) -> None:
        from datetime import UTC, datetime

        rule = _make_rule()
        rule.effective_until = None
        repo = _make_repo(rule=rule)
        session = AsyncMock()
        session.flush = AsyncMock()
        deadline = datetime.now(UTC)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.api.services.pricing.BillingRepository", lambda _: repo)
            svc = PricingService()
            await svc.update_rule(rule.id, effective_until=deadline, session=session)

        assert rule.effective_until is deadline
