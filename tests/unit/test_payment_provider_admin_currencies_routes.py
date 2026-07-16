"""Unit tests for the superadmin payment currency catalog routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
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


async def test_refresh_catalog_error_returns_502() -> None:
    from src.api.services.billing_errors import PaymentCatalogError

    sync_service = AsyncMock()
    sync_service.refresh = AsyncMock(side_effect=PaymentCatalogError("bad shape"))
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
    session.commit.assert_not_awaited()


async def test_refresh_http_error_returns_502() -> None:
    import httpx

    sync_service = AsyncMock()
    sync_service.refresh = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
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
    session.commit.assert_not_awaited()


async def test_refresh_with_storage_dead_logo_cache_returns_200_not_500() -> None:
    """End-to-end proof of the D10/P1-2 degrade: a dead assets bucket must not 500/502."""
    from src.api.services.payment_currency_logos import LogoCacheService
    from src.api.services.payment_currency_sync import PaymentCurrencySyncService
    from src.api.services.payments.catalog import CurrencyDetails
    from src.api.services.payments.registry import GatewayRegistry
    from src.api.services.storage.exceptions import StorageConnectionError

    class _FakeCatalogGateway:
        provider = PaymentProvider.NOWPAYMENTS

        def __init__(self) -> None:
            self.list_merchant_currencies = AsyncMock(return_value=["BTC"])
            self.list_full_currencies = AsyncMock(
                return_value={
                    "BTC": CurrencyDetails(
                        ticker="BTC",
                        name="Bitcoin",
                        network="BTC",
                        logo_url="https://nowpayments.io/btc.svg",
                    )
                }
            )

    r2 = AsyncMock()
    r2.exists = AsyncMock(side_effect=StorageConnectionError("403 Forbidden on HeadObject"))
    logo_cache = LogoCacheService(
        r2_client=r2,
        http_client_factory=lambda: _http_client_returning(b"<svg></svg>", "image/svg+xml"),
    )
    gateway = _FakeCatalogGateway()
    registry = GatewayRegistry([gateway])  # type: ignore[list-item]
    sync_service = PaymentCurrencySyncService(registry=registry, logo_cache=logo_cache)

    session = AsyncMock()
    repo = AsyncMock()
    repo.list_currencies = AsyncMock(return_value=[])
    repo.sync_catalog = AsyncMock(return_value=(1, 0))
    audit_repo = MagicMock()
    audit_repo.write_audit = AsyncMock()

    with (
        patch(
            "src.api.services.payment_currency_sync.PaymentCurrencyRepository", return_value=repo
        ),
        patch("src.api.routes.payment_provider_admin.AdminRepository", return_value=audit_repo),
    ):
        result = await PaymentProviderAdminController.refresh_currencies.fn(
            MagicMock(),
            superadmin=MagicMock(id=uuid4()),
            session=session,
            product_config=VEX_CONFIG,
            payment_currency_sync_service=sync_service,
        )

    assert result == [SyncResult(provider=PaymentProvider.NOWPAYMENTS, upserted=1, deactivated=0)]
    session.commit.assert_awaited_once()
    entries = repo.sync_catalog.call_args.args[2]
    assert entries[0].logo_key is None


class _FakeLogoResponse:
    def __init__(self, *, content_type: str, content: bytes) -> None:
        self.headers = {"content-type": content_type}
        self._content = content

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self) -> Any:
        yield self._content


class _FakeLogoStreamCM:
    def __init__(self, response: _FakeLogoResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeLogoResponse:
        return self._response

    async def __aexit__(self, *args: object) -> bool:
        return False


def _http_client_returning(content: bytes, content_type: str) -> AsyncMock:
    response = _FakeLogoResponse(content_type=content_type, content=content)
    client = AsyncMock()
    client.stream = MagicMock(return_value=_FakeLogoStreamCM(response))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


async def test_refresh_unexpected_exception_propagates_as_500() -> None:
    """A genuine programming error must not be misreported as 'bad gateway'."""
    sync_service = AsyncMock()
    sync_service.refresh = AsyncMock(side_effect=AttributeError("'NoneType' has no attribute 'x'"))
    session = AsyncMock()

    with pytest.raises(AttributeError):
        await PaymentProviderAdminController.refresh_currencies.fn(
            MagicMock(),
            superadmin=MagicMock(id=uuid4()),
            session=session,
            product_config=VEX_CONFIG,
            payment_currency_sync_service=sync_service,
        )

    session.commit.assert_not_awaited()
