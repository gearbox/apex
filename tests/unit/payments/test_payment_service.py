"""Provider-neutral payment orchestrator tests."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.billing_errors import PaymentProviderDisabledError
from src.api.services.payments import (
    ChargeResult,
    GatewayRegistry,
    PaymentLookup,
    PaymentService,
    WebhookEnvelope,
    WebhookOutcome,
)
from src.core.config import Settings
from src.core.enums import PaymentStatus
from src.core.product import PaymentProvider
from src.core.product_registry import VEX_CONFIG

pytestmark = pytest.mark.unit


def _settings() -> Settings:
    return Settings(jwt_secret_key="test-secret-key-that-is-definitely-long-enough-32bytes")


def _service(
    gateway: AsyncMock, state: AsyncMock, billing: AsyncMock | None = None
) -> PaymentService:
    gateway.provider = PaymentProvider.STRIPE
    return PaymentService(
        billing_service=billing or AsyncMock(),
        settings=_settings(),
        registry=GatewayRegistry([gateway]),
        provider_state_service=state,
    )


async def test_create_charge_persists_normalized_fields() -> None:
    gateway = AsyncMock()
    gateway.create_charge = AsyncMock(
        return_value=ChargeResult(
            external_id="cs_1",
            redirect_url="https://checkout/1",
            currency="USD",
            provider_metadata={"provider": "data"},
        )
    )
    state = AsyncMock()
    state.is_effective = AsyncMock(return_value=True)
    repo = AsyncMock()
    repo.get_account = AsyncMock(return_value=MagicMock())

    with patch("src.api.services.payments.service.BillingRepository", return_value=repo):
        result = await _service(gateway, state).create_charge(
            PaymentProvider.STRIPE,
            uuid4(),
            100,
            uuid4(),
            session=AsyncMock(),
            product_config=VEX_CONFIG,
        )

    assert result.external_id == "cs_1"
    assert repo.create_payment.await_args.kwargs["payment_provider"] == "stripe"
    assert repo.create_payment.await_args.kwargs["amount_usd"] == Decimal("95.00")
    assert repo.create_payment.await_args.kwargs["tokens_granted"] == 10_000


async def test_disabled_provider_stops_before_gateway_call() -> None:
    gateway = AsyncMock()
    state = AsyncMock()
    state.is_effective = AsyncMock(return_value=False)
    repo = AsyncMock()
    repo.get_account = AsyncMock(return_value=MagicMock())
    with (
        patch("src.api.services.payments.service.BillingRepository", return_value=repo),
        pytest.raises(PaymentProviderDisabledError),
    ):
        await _service(gateway, state).create_charge(
            PaymentProvider.STRIPE,
            uuid4(),
            10,
            uuid4(),
            session=AsyncMock(),
            product_config=VEX_CONFIG,
        )
    gateway.create_charge.assert_not_awaited()


async def test_completed_webhook_credits_once_and_replay_noops() -> None:
    payment = MagicMock(
        id=uuid4(),
        account_id=uuid4(),
        status=PaymentStatus.PENDING.value,
        amount_usd=Decimal("10.00"),
        tokens_granted=1000,
        product_id="vex",
        provider_metadata={},
    )
    gateway = AsyncMock()
    gateway.verify_webhook = AsyncMock(
        return_value=WebhookOutcome(
            lookup=PaymentLookup(by="external_id", value="cs_1"),
            status=PaymentStatus.COMPLETED,
        )
    )
    state = AsyncMock()
    billing = AsyncMock()
    billing.credit = AsyncMock(return_value=MagicMock(event=None))
    repo = AsyncMock()
    repo.get_payment_by_external_id_for_update = AsyncMock(return_value=payment)
    repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)
    session = AsyncMock()

    with patch("src.api.services.payments.service.BillingRepository", return_value=repo):
        service = _service(gateway, state, billing)
        envelope = WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex")
        await service.handle_webhook(PaymentProvider.STRIPE, envelope, session=session)
        await service.handle_webhook(PaymentProvider.STRIPE, envelope, session=session)

    billing.credit.assert_awaited_once()
    assert payment.status == PaymentStatus.COMPLETED.value


async def test_status_none_does_not_touch_repository() -> None:
    gateway = AsyncMock()
    gateway.verify_webhook = AsyncMock(
        return_value=WebhookOutcome(
            lookup=PaymentLookup(by="external_id", value="ignored"), status=None
        )
    )
    with patch("src.api.services.payments.service.BillingRepository") as repository:
        await _service(gateway, AsyncMock()).handle_webhook(
            PaymentProvider.STRIPE,
            WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
            session=AsyncMock(),
        )
    repository.assert_not_called()
