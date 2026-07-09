"""Superadmin payment provider route handler tests."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from litestar.exceptions import HTTPException, NotFoundException

from src.api.routes.payment_provider_admin import PaymentProviderAdminController
from src.api.schemas.admin import PaymentProviderPatchRequest
from src.api.services.billing_errors import UnknownProviderError
from src.api.services.payment_provider_state import ProviderInfo
from src.core.product import PaymentProvider
from src.core.product_registry import VEX_CONFIG

pytestmark = pytest.mark.unit


async def test_empty_patch_returns_400() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await PaymentProviderAdminController.update_provider.fn(
            MagicMock(),
            superadmin=MagicMock(id=uuid4()),
            provider="stripe",
            data=PaymentProviderPatchRequest(),
            session=AsyncMock(),
            product_config=VEX_CONFIG,
            payment_provider_state_service=AsyncMock(),
        )
    assert exc_info.value.status_code == 400


async def test_unknown_provider_returns_404() -> None:
    with pytest.raises(NotFoundException):
        await PaymentProviderAdminController.update_provider.fn(
            MagicMock(),
            superadmin=MagicMock(id=uuid4()),
            provider="unknown",
            data=PaymentProviderPatchRequest(is_enabled=False),
            session=AsyncMock(),
            product_config=VEX_CONFIG,
            payment_provider_state_service=AsyncMock(),
        )


async def test_non_capability_provider_returns_404() -> None:
    service = AsyncMock()
    service.set_state = AsyncMock(side_effect=UnknownProviderError(PaymentProvider.STRIPE))
    with pytest.raises(NotFoundException):
        await PaymentProviderAdminController.update_provider.fn(
            MagicMock(),
            superadmin=MagicMock(id=uuid4()),
            provider="stripe",
            data=PaymentProviderPatchRequest(is_enabled=False),
            session=AsyncMock(),
            product_config=VEX_CONFIG,
            payment_provider_state_service=service,
        )


async def test_happy_path_commits_and_returns_info() -> None:
    info = ProviderInfo(
        provider=PaymentProvider.STRIPE,
        is_enabled=False,
        display_order=2,
        credentials_configured=True,
    )
    service = AsyncMock()
    service.set_state = AsyncMock(return_value=info)
    session = AsyncMock()
    result = await PaymentProviderAdminController.update_provider.fn(
        MagicMock(),
        superadmin=MagicMock(id=uuid4()),
        provider="stripe",
        data=PaymentProviderPatchRequest(is_enabled=False, display_order=2),
        session=session,
        product_config=VEX_CONFIG,
        payment_provider_state_service=service,
    )
    assert result == info
    session.commit.assert_awaited_once()
