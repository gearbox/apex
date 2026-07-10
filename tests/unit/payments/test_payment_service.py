"""Provider-neutral payment orchestrator tests."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.billing import BillingService
from src.api.services.billing_errors import PaymentProviderDisabledError, PaymentVerificationError
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
    gateway: AsyncMock, state: AsyncMock, billing: AsyncMock | BillingService | None = None
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


# ---------------------------------------------------------------------------
# P1-1 / P1-2 — unit-consistent proportional settlement ratio, fail-loud
# on bad IPN fields, and the restored ratio/tolerance/delta/mismatch/event
# matrix (ported from the pre-refactor master test_payment_service.py onto
# the provider-neutral WebhookOutcome contract).
# ---------------------------------------------------------------------------


def _make_payment(
    *,
    tokens_granted: int = 1000,
    amount_usd: Decimal = Decimal("10.00"),
    status: str = PaymentStatus.PENDING.value,
    product_id: str = "vex",
) -> MagicMock:
    return MagicMock(
        id=uuid4(),
        account_id=uuid4(),
        status=status,
        amount_usd=amount_usd,
        tokens_granted=tokens_granted,
        product_id=product_id,
        provider_metadata={},
    )


def _ratio_outcome(
    *,
    payment_id: object,
    status: PaymentStatus,
    amount_paid: str,
    amount_due: str,
) -> WebhookOutcome:
    return WebhookOutcome(
        lookup=PaymentLookup(by="payment_id", value=str(payment_id)),
        status=status,
        amount_paid=Decimal(amount_paid),
        amount_due=Decimal(amount_due),
    )


class TestProportionalSettlement:
    """D2/P1-1: proportional crediting keyed on amount_paid/amount_due (both
    in the provider's pay-currency unit) — never mixed with fiat amount_usd."""

    async def test_product_mismatch_raises(self) -> None:
        payment = _make_payment(product_id="synthara")
        gateway = AsyncMock()
        gateway.verify_webhook = AsyncMock(
            return_value=_ratio_outcome(
                payment_id=payment.id,
                status=PaymentStatus.COMPLETED,
                amount_paid="10.00",
                amount_due="10.00",
            )
        )
        repo = AsyncMock()
        repo.get_payment_for_update = AsyncMock(return_value=payment)

        with (
            patch("src.api.services.payments.service.BillingRepository", return_value=repo),
            pytest.raises(PaymentVerificationError, match="product mismatch"),
        ):
            await _service(gateway, AsyncMock()).handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )

    async def test_zero_amount_due_raises(self) -> None:
        payment = _make_payment()
        gateway = AsyncMock()
        gateway.verify_webhook = AsyncMock(
            return_value=WebhookOutcome(
                lookup=PaymentLookup(by="payment_id", value=str(payment.id)),
                status=PaymentStatus.COMPLETED,
                amount_paid=Decimal("10.00"),
                amount_due=Decimal("0"),
            )
        )
        repo = AsyncMock()
        repo.get_payment_for_update = AsyncMock(return_value=payment)

        with (
            patch("src.api.services.payments.service.BillingRepository", return_value=repo),
            pytest.raises(PaymentVerificationError),
        ):
            await _service(gateway, AsyncMock()).handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )

    async def test_ratio_is_pay_currency_units_independent_of_fiat_price(self) -> None:
        """actually_paid=0.5, pay_amount=1.0 -> half tokens, regardless of
        payment.amount_usd — proves the ratio no longer mixes units."""
        payment = _make_payment(tokens_granted=1000, amount_usd=Decimal("237.19"))
        gateway = AsyncMock()
        gateway.verify_webhook = AsyncMock(
            return_value=_ratio_outcome(
                payment_id=payment.id,
                status=PaymentStatus.PARTIALLY_PAID,
                amount_paid="0.5",
                amount_due="1.0",
            )
        )
        repo = AsyncMock()
        repo.get_payment_for_update = AsyncMock(return_value=payment)
        repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)
        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))

        with patch("src.api.services.payments.service.BillingRepository", return_value=repo):
            await _service(gateway, AsyncMock(), billing).handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )

        assert billing.credit.await_args.args[1] == 500

    async def test_partially_paid_credits_proportional_and_sets_status(self) -> None:
        payment = _make_payment(tokens_granted=1000)
        gateway = AsyncMock()
        gateway.verify_webhook = AsyncMock(
            return_value=_ratio_outcome(
                payment_id=payment.id,
                status=PaymentStatus.PARTIALLY_PAID,
                amount_paid="4.00",
                amount_due="10.00",
            )
        )
        repo = AsyncMock()
        repo.get_payment_for_update = AsyncMock(return_value=payment)
        repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)
        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))

        with patch("src.api.services.payments.service.BillingRepository", return_value=repo):
            await _service(gateway, AsyncMock(), billing).handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )

        assert billing.credit.await_args.args[1] == 400
        assert payment.status == PaymentStatus.PARTIALLY_PAID.value

    async def test_partially_paid_redelivery_credits_zero_delta(self) -> None:
        payment = _make_payment(tokens_granted=1000)
        gateway = AsyncMock()
        gateway.verify_webhook = AsyncMock(
            return_value=_ratio_outcome(
                payment_id=payment.id,
                status=PaymentStatus.PARTIALLY_PAID,
                amount_paid="4.00",
                amount_due="10.00",
            )
        )
        repo = AsyncMock()
        repo.get_payment_for_update = AsyncMock(return_value=payment)
        repo.get_credited_tokens_for_payment = AsyncMock(side_effect=[0, 400])
        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))

        with patch("src.api.services.payments.service.BillingRepository", return_value=repo):
            service = _service(gateway, AsyncMock(), billing)
            envelope = WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex")
            await service.handle_webhook(PaymentProvider.STRIPE, envelope, session=AsyncMock())
            await service.handle_webhook(PaymentProvider.STRIPE, envelope, session=AsyncMock())

        billing.credit.assert_awaited_once()

    async def test_finished_after_partial_credits_exact_remainder(self) -> None:
        payment = _make_payment(tokens_granted=1000)
        gateway = AsyncMock()
        repo = AsyncMock()
        repo.get_payment_for_update = AsyncMock(return_value=payment)
        repo.get_credited_tokens_for_payment = AsyncMock(side_effect=[0, 400])
        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))
        service = _service(gateway, AsyncMock(), billing)

        with patch("src.api.services.payments.service.BillingRepository", return_value=repo):
            gateway.verify_webhook = AsyncMock(
                return_value=_ratio_outcome(
                    payment_id=payment.id,
                    status=PaymentStatus.PARTIALLY_PAID,
                    amount_paid="4.00",
                    amount_due="10.00",
                )
            )
            await service.handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )
            assert payment.status == PaymentStatus.PARTIALLY_PAID.value

            gateway.verify_webhook = AsyncMock(
                return_value=_ratio_outcome(
                    payment_id=payment.id,
                    status=PaymentStatus.COMPLETED,
                    amount_paid="10.00",
                    amount_due="10.00",
                )
            )
            await service.handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )

        assert billing.credit.await_count == 2
        first_delta = billing.credit.await_args_list[0].args[1]
        second_delta = billing.credit.await_args_list[1].args[1]
        assert first_delta == 400
        assert second_delta == 600
        assert first_delta + second_delta == payment.tokens_granted
        assert payment.status == PaymentStatus.COMPLETED.value

    async def test_underpaid_finished_credits_proportional_and_completes(self) -> None:
        payment = _make_payment(tokens_granted=1000)
        gateway = AsyncMock()
        gateway.verify_webhook = AsyncMock(
            return_value=_ratio_outcome(
                payment_id=payment.id,
                status=PaymentStatus.COMPLETED,
                amount_paid="7.00",
                amount_due="10.00",
            )
        )
        repo = AsyncMock()
        repo.get_payment_for_update = AsyncMock(return_value=payment)
        repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)
        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))

        with (
            patch("src.api.services.payments.service.BillingRepository", return_value=repo),
            patch("src.api.services.payments.service.logger") as mock_logger,
        ):
            await _service(gateway, AsyncMock(), billing).handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )

        assert billing.credit.await_args.args[1] == 700
        assert payment.status == PaymentStatus.COMPLETED.value
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args.args[0] == "payment.underpaid_credited"

    @pytest.mark.parametrize("amount_paid", ["9.95", "10.05"])
    async def test_ratio_within_band_snaps_to_full_tokens(self, amount_paid: str) -> None:
        payment = _make_payment(tokens_granted=1000)
        gateway = AsyncMock()
        gateway.verify_webhook = AsyncMock(
            return_value=_ratio_outcome(
                payment_id=payment.id,
                status=PaymentStatus.COMPLETED,
                amount_paid=amount_paid,
                amount_due="10.00",
            )
        )
        repo = AsyncMock()
        repo.get_payment_for_update = AsyncMock(return_value=payment)
        repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)
        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))

        with (
            patch("src.api.services.payments.service.BillingRepository", return_value=repo),
            patch("src.api.services.payments.service.logger") as mock_logger,
        ):
            await _service(gateway, AsyncMock(), billing).handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )

        assert billing.credit.await_args.args[1] == 1000
        assert payment.status == PaymentStatus.COMPLETED.value
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    async def test_overpaid_credits_proportional(self) -> None:
        payment = _make_payment(tokens_granted=1000)
        gateway = AsyncMock()
        gateway.verify_webhook = AsyncMock(
            return_value=_ratio_outcome(
                payment_id=payment.id,
                status=PaymentStatus.COMPLETED,
                amount_paid="15.00",
                amount_due="10.00",
            )
        )
        repo = AsyncMock()
        repo.get_payment_for_update = AsyncMock(return_value=payment)
        repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)
        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))

        with (
            patch("src.api.services.payments.service.BillingRepository", return_value=repo),
            patch("src.api.services.payments.service.logger") as mock_logger,
        ):
            await _service(gateway, AsyncMock(), billing).handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )

        assert billing.credit.await_args.args[1] == 1500
        assert payment.status == PaymentStatus.COMPLETED.value
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args.args[0] == "payment.overpaid_credited"

    @pytest.mark.parametrize("amount_paid", ["2.00", "25.00"])
    async def test_extreme_ratio_logs_error_but_credits(self, amount_paid: str) -> None:
        payment = _make_payment(tokens_granted=1000)
        gateway = AsyncMock()
        gateway.verify_webhook = AsyncMock(
            return_value=_ratio_outcome(
                payment_id=payment.id,
                status=PaymentStatus.COMPLETED,
                amount_paid=amount_paid,
                amount_due="10.00",
            )
        )
        repo = AsyncMock()
        repo.get_payment_for_update = AsyncMock(return_value=payment)
        repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)
        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))

        with (
            patch("src.api.services.payments.service.BillingRepository", return_value=repo),
            patch("src.api.services.payments.service.logger") as mock_logger,
        ):
            await _service(gateway, AsyncMock(), billing).handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )

        billing.credit.assert_awaited_once()
        assert billing.credit.await_args.args[1] > 0
        assert payment.status == PaymentStatus.COMPLETED.value
        mock_logger.error.assert_called_once()
        mock_logger.warning.assert_not_called()

    async def test_credit_returns_event_with_owner_targets(self) -> None:
        """R1: the event from a ratio-based credit carries the account owner
        as an SSE target — real BillingService, not a mock."""
        payment = _make_payment(tokens_granted=1000)
        owner_user_id = uuid4()
        account = MagicMock(user_id=owner_user_id, organization_id=None)

        gateway = AsyncMock()
        gateway.verify_webhook = AsyncMock(
            return_value=_ratio_outcome(
                payment_id=payment.id,
                status=PaymentStatus.COMPLETED,
                amount_paid="10.00",
                amount_due="10.00",
            )
        )
        repo = AsyncMock()
        repo.get_payment_for_update = AsyncMock(return_value=payment)
        repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)
        repo.get_account_for_update = AsyncMock(return_value=account)
        repo.get_balance = AsyncMock(return_value=0)
        repo.create_transaction = AsyncMock(return_value=MagicMock(id=uuid4()))

        billing = BillingService()
        with (
            patch("src.api.services.payments.service.BillingRepository", return_value=repo),
            patch("src.api.services.billing.BillingRepository", return_value=repo),
        ):
            event = await _service(gateway, AsyncMock(), billing).handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )

        assert event is not None
        assert event.user_ids == [owner_user_id]

    async def test_full_credit_returns_event_with_owner_targets(self) -> None:
        """Same as above but the plain COMPLETED / amount_paid=None path
        (the Stripe checkout branch, no ratio)."""
        payment = _make_payment(tokens_granted=500)
        owner_user_id = uuid4()
        account = MagicMock(user_id=owner_user_id, organization_id=None)

        gateway = AsyncMock()
        gateway.verify_webhook = AsyncMock(
            return_value=WebhookOutcome(
                lookup=PaymentLookup(by="external_id", value="cs_1"),
                status=PaymentStatus.COMPLETED,
            )
        )
        repo = AsyncMock()
        repo.get_payment_by_external_id_for_update = AsyncMock(return_value=payment)
        repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)
        repo.get_account_for_update = AsyncMock(return_value=account)
        repo.get_balance = AsyncMock(return_value=0)
        repo.create_transaction = AsyncMock(return_value=MagicMock(id=uuid4()))

        billing = BillingService()
        with (
            patch("src.api.services.payments.service.BillingRepository", return_value=repo),
            patch("src.api.services.billing.BillingRepository", return_value=repo),
        ):
            event = await _service(gateway, AsyncMock(), billing).handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )

        assert event is not None
        assert event.user_ids == [owner_user_id]


class TestLateIntermediateIPNStatusRegression:
    """P2-3: an out-of-order intermediate IPN (PENDING) arriving after a
    partial credit must not erase the PARTIALLY_PAID marker."""

    async def test_pending_after_partially_paid_does_not_regress_status(self) -> None:
        payment = _make_payment(status=PaymentStatus.PARTIALLY_PAID.value)
        gateway = AsyncMock()
        gateway.verify_webhook = AsyncMock(
            return_value=WebhookOutcome(
                lookup=PaymentLookup(by="payment_id", value=str(payment.id)),
                status=PaymentStatus.PENDING,
            )
        )
        repo = AsyncMock()
        repo.get_payment_for_update = AsyncMock(return_value=payment)
        billing = AsyncMock()

        with patch("src.api.services.payments.service.BillingRepository", return_value=repo):
            await _service(gateway, AsyncMock(), billing).handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )

        assert payment.status == PaymentStatus.PARTIALLY_PAID.value
        billing.credit.assert_not_awaited()

    async def test_pending_on_pending_payment_still_updates(self) -> None:
        """Sanity check: the guard is scoped to the PARTIALLY_PAID -> PENDING
        transition only — a normal PENDING -> PENDING update still applies."""
        payment = _make_payment(status=PaymentStatus.PENDING.value)
        gateway = AsyncMock()
        gateway.verify_webhook = AsyncMock(
            return_value=WebhookOutcome(
                lookup=PaymentLookup(by="payment_id", value=str(payment.id)),
                status=PaymentStatus.PENDING,
            )
        )
        repo = AsyncMock()
        repo.get_payment_for_update = AsyncMock(return_value=payment)

        with patch("src.api.services.payments.service.BillingRepository", return_value=repo):
            await _service(gateway, AsyncMock(), AsyncMock()).handle_webhook(
                PaymentProvider.STRIPE,
                WebhookEnvelope(raw_body=b"{}", headers={}, product_id="vex"),
                session=AsyncMock(),
            )

        assert payment.status == PaymentStatus.PENDING.value
