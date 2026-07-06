"""Unit tests for the AdminController balance-adjustment route (R6)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.routes.admin import AdminController
from src.api.schemas.billing import AdminAdjustRequest
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
