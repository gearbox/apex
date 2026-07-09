"""Unit tests for the AdminController balance-adjustment route (R6)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.routes.admin import AdminController
from src.api.schemas.billing import (
    AdminAdjustRequest,
    CreatePricingRuleRequest,
    PatchPricingRuleRequest,
)
from src.core.enums import TransactionType
from src.db.models.billing import TokenTransaction

pytestmark = pytest.mark.unit


class TestAdminAdjustBalanceRoute:
    async def test_admin_adjust_balance_from_txn_no_extra_ledger_scan(self) -> None:
        """``new_balance`` in the response must come from
        ``billing_result.txn.balance_after`` — no extra ``get_balance()``
        ledger scan after ``admin_adjust()`` (R6)."""
        account_id = uuid4()
        txn = TokenTransaction(
            id=uuid4(),
            account_id=account_id,
            transaction_type=TransactionType.ADMIN_ADJUSTMENT.value,
            amount=100,
            balance_after=600,
            description="bonus",
            product_id="vex",
            metadata_={},
            created_at=datetime.now(UTC),
            created_by=None,
            job_id=None,
            payment_id=None,
        )

        billing_service = AsyncMock()
        billing_service.admin_adjust = AsyncMock(return_value=MagicMock(txn=txn, event=None))
        # If the route regressed to re-querying the ledger, this stray value
        # would leak into new_balance instead of the txn's own balance_after.
        billing_service.get_balance = AsyncMock(return_value=999)

        event_bus = AsyncMock()
        idempotency_service = AsyncMock()
        idempotency_service.check = AsyncMock(return_value=uuid4())
        idempotency_service.complete = AsyncMock()

        admin_user = MagicMock(id=uuid4())

        response = await AdminController.adjust_account.fn(  # type: ignore[attr-defined]
            MagicMock(),
            admin_user=admin_user,
            account_id=account_id,
            data=AdminAdjustRequest(amount=100, description="bonus"),
            session=AsyncMock(),
            billing_service=billing_service,
            event_bus=event_bus,
            product_id="vex",
            idempotency_service=idempotency_service,
            idempotency_key_header="test-key-1",
        )

        assert response.content.new_balance == 600
        billing_service.get_balance.assert_not_called()


def _make_pricing_rule(*, input_token_cost: int = 0) -> MagicMock:
    rule = MagicMock()
    rule.id = uuid4()
    rule.provider = "grok"
    rule.generation_type = "i2i"
    rule.model = "grok-imagine-image"
    rule.token_cost = 20
    rule.input_token_cost = input_token_cost
    rule.is_active = True
    rule.effective_from = datetime.now(UTC)
    rule.effective_until = None
    rule.notes = None
    return rule


class TestAdminPricingRoutes:
    async def test_create_pricing_rule_persists_and_echoes_input_token_cost(self) -> None:
        rule = _make_pricing_rule(input_token_cost=3)
        pricing_service = AsyncMock()
        pricing_service.create_rule = AsyncMock(return_value=rule)

        response = await AdminController.create_pricing_rule.fn(  # type: ignore[attr-defined]
            MagicMock(),
            admin_user=MagicMock(id=uuid4()),
            data=CreatePricingRuleRequest(
                provider="grok",
                generation_type="i2i",
                model="grok-imagine-image",
                token_cost=20,
                input_token_cost=3,
            ),
            session=AsyncMock(),
            pricing_service=pricing_service,
        )

        assert response.content.input_token_cost == 3
        assert pricing_service.create_rule.await_args.kwargs["input_token_cost"] == 3

    async def test_create_pricing_rule_defaults_input_token_cost_to_zero(self) -> None:
        rule = _make_pricing_rule(input_token_cost=0)
        pricing_service = AsyncMock()
        pricing_service.create_rule = AsyncMock(return_value=rule)

        response = await AdminController.create_pricing_rule.fn(  # type: ignore[attr-defined]
            MagicMock(),
            admin_user=MagicMock(id=uuid4()),
            data=CreatePricingRuleRequest(
                provider="grok",
                generation_type="t2i",
                model="grok-imagine-image",
                token_cost=20,
            ),
            session=AsyncMock(),
            pricing_service=pricing_service,
        )

        assert response.content.input_token_cost == 0
        assert pricing_service.create_rule.await_args.kwargs["input_token_cost"] == 0

    async def test_patch_pricing_rule_updates_input_token_cost_only(self) -> None:
        rule = _make_pricing_rule(input_token_cost=3)
        pricing_service = AsyncMock()
        pricing_service.update_rule = AsyncMock(return_value=rule)

        response = await AdminController.update_pricing_rule.fn(  # type: ignore[attr-defined]
            MagicMock(),
            admin_user=MagicMock(id=uuid4()),
            rule_id=rule.id,
            data=PatchPricingRuleRequest(input_token_cost=3),
            session=AsyncMock(),
            pricing_service=pricing_service,
        )

        assert response.token_cost == 20
        assert response.input_token_cost == 3
        update_kwargs = pricing_service.update_rule.await_args.kwargs
        assert update_kwargs["token_cost"] is None
        assert update_kwargs["input_token_cost"] == 3
        assert update_kwargs["is_active"] is None
        assert "effective_until" not in update_kwargs

    async def test_patch_pricing_rule_can_clear_effective_until(self) -> None:
        rule = _make_pricing_rule(input_token_cost=0)
        rule.effective_until = None
        pricing_service = AsyncMock()
        pricing_service.update_rule = AsyncMock(return_value=rule)

        response = await AdminController.update_pricing_rule.fn(  # type: ignore[attr-defined]
            MagicMock(),
            admin_user=MagicMock(id=uuid4()),
            rule_id=rule.id,
            data=PatchPricingRuleRequest(effective_until=None),
            session=AsyncMock(),
            pricing_service=pricing_service,
        )

        assert response.effective_until is None
        assert pricing_service.update_rule.await_args.kwargs["effective_until"] is None
