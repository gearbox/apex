"""Payment provider capability/runtime-state intersection tests."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.billing_errors import UnknownProviderError
from src.api.services.payment_provider_state import PaymentProviderStateService
from src.core.config import Settings
from src.core.product import PaymentProvider
from src.core.product_registry import SYNTHARA_CONFIG, VEX_CONFIG

pytestmark = pytest.mark.unit


def _service() -> PaymentProviderStateService:
    return PaymentProviderStateService(
        Settings(
            jwt_secret_key="test-secret-key-that-is-definitely-long-enough-32bytes",
            stripe_secret_key_vex="sk_test",
            stripe_webhook_secret_vex="whsec_test",
            nowpayments_api_key_vex="np_key",
            nowpayments_ipn_secret_vex="np_secret",
        )
    )


async def test_absent_rows_enable_full_capability() -> None:
    repository = AsyncMock()
    repository.get_states = AsyncMock(return_value=[])
    with patch(
        "src.api.services.payment_provider_state.PaymentProviderStateRepository",
        return_value=repository,
    ):
        infos = await _service().effective_providers(VEX_CONFIG, session=AsyncMock())
    assert {info.provider for info in infos} == VEX_CONFIG.payment_providers


async def test_disabled_row_is_excluded_and_order_is_applied() -> None:
    disabled = MagicMock(provider="stripe", is_enabled=False, display_order=0)
    enabled = MagicMock(provider="nowpayments", is_enabled=True, display_order=5)
    repository = AsyncMock()
    repository.get_states = AsyncMock(return_value=[enabled, disabled])
    with patch(
        "src.api.services.payment_provider_state.PaymentProviderStateRepository",
        return_value=repository,
    ):
        infos = await _service().effective_providers(VEX_CONFIG, session=AsyncMock())
    assert [info.provider for info in infos] == [PaymentProvider.NOWPAYMENTS]
    assert infos[0].display_order == 5


async def test_provider_outside_capability_is_rejected() -> None:
    with pytest.raises(UnknownProviderError):
        await _service().set_state(
            SYNTHARA_CONFIG,
            PaymentProvider.NOWPAYMENTS,
            is_enabled=True,
            display_order=None,
            actor_id=uuid4(),
            session=AsyncMock(),
        )


async def test_disable_writes_null_target_audit() -> None:
    state = MagicMock(provider="stripe", is_enabled=False, display_order=0)
    repository = AsyncMock()
    repository.get_states = AsyncMock(return_value=[])
    repository.upsert_state = AsyncMock(return_value=state)
    admin_repository = AsyncMock()
    with (
        patch(
            "src.api.services.payment_provider_state.PaymentProviderStateRepository",
            return_value=repository,
        ),
        patch(
            "src.api.services.payment_provider_state.AdminRepository",
            return_value=admin_repository,
        ),
    ):
        await _service().set_state(
            VEX_CONFIG,
            PaymentProvider.STRIPE,
            is_enabled=False,
            display_order=None,
            actor_id=uuid4(),
            session=AsyncMock(),
        )
    audit = admin_repository.write_audit.await_args.args[0]
    assert audit.target_user_id is None
    assert audit.action == "payment_provider.disable"
