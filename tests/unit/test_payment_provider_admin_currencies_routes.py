"""Unit tests for the superadmin payment currency catalog routes."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from litestar.exceptions import HTTPException

from src.api.routes.payment_provider_admin import PaymentProviderAdminController
from src.api.services.payment_currency_sync import SyncResult
from src.core.product import PaymentProvider
from src.core.product_registry import VEX_CONFIG

pytestmark = pytest.mark.unit


@dataclass
class _Row:
    ticker: str
    provider: str = "nowpayments"
    is_available: bool = True
    name: str | None = None
    network: str | None = None
    logo_key: str | None = None
    logo_source_url: str | None = None
    logo_synced_at: object = None
    last_seen_at: object = None


async def test_list_currencies_returns_full_catalog_incl_unavailable() -> None:
    repo = AsyncMock()
    repo.list_currencies = AsyncMock(
        return_value=[
            _Row(ticker="BTC", is_available=True),
            _Row(ticker="OLD", is_available=False),
        ]
    )
    with patch(
        "src.api.routes.payment_provider_admin.PaymentCurrencyRepository", return_value=repo
    ):
        result = await PaymentProviderAdminController.list_currencies.fn(
            MagicMock(),
            superadmin=MagicMock(id=uuid4()),
            session=AsyncMock(),
            product_config=VEX_CONFIG,
        )
    assert {row.ticker for row in result} == {"BTC", "OLD"}
    repo.list_currencies.assert_awaited_once_with(VEX_CONFIG.slug, only_available=False)


async def test_refresh_happy_path_commits_and_writes_audit() -> None:
    results = [SyncResult(provider=PaymentProvider.NOWPAYMENTS, upserted=5, deactivated=1)]
    sync_service = AsyncMock()
    sync_service.refresh = AsyncMock(return_value=results)
    session = AsyncMock()
    actor_id = uuid4()

    audit_repo = MagicMock()
    audit_repo.write_audit = AsyncMock()
    with patch("src.api.routes.payment_provider_admin.AdminRepository", return_value=audit_repo):
        result = await PaymentProviderAdminController.refresh_currencies.fn(
            MagicMock(),
            superadmin=MagicMock(id=actor_id),
            session=session,
            product_config=VEX_CONFIG,
            payment_currency_sync_service=sync_service,
        )

    assert result == results
    audit_repo.write_audit.assert_awaited_once()
    audit_entry = audit_repo.write_audit.call_args.args[0]
    assert audit_entry.action == "payment_currencies.refresh"
    assert audit_entry.target_user_id is None
    assert audit_entry.actor_id == actor_id
    session.commit.assert_awaited_once()


async def test_refresh_provider_failure_returns_502_and_does_not_commit() -> None:
    sync_service = AsyncMock()
    sync_service.refresh = AsyncMock(side_effect=RuntimeError("nowpayments unreachable"))
    session = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await PaymentProviderAdminController.refresh_currencies.fn(
            MagicMock(),
            superadmin=MagicMock(id=uuid4()),
            session=session,
            product_config=VEX_CONFIG,
            payment_currency_sync_service=sync_service,
        )

    assert exc_info.value.status_code == 502
    assert "nowpayments unreachable" in str(exc_info.value.detail)
    session.commit.assert_not_awaited()
