"""Unit tests for PaymentService: checkout/invoice creation and webhook parsing.

Complements the concurrency-focused integration tests in
tests/integration/test_payment_webhook_concurrency.py, which cover the
row-lock double-credit fix (C2) against a real database. These tests isolate
the pure logic — per-product credential resolution (C6/C10), the redirect
URL fix (C7), and the NowPayments payment-id resolution + HMAC canonicalization
(C1/C12) — with a mocked repository so they run fast and pin the exact
call arguments.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_module
import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import stripe

from src.api.services.billing import BillingService
from src.api.services.billing_errors import AccountNotFoundError, PaymentVerificationError
from src.api.services.payment import PaymentService
from src.core.config import Settings
from src.core.enums import PaymentStatus
from src.core.product_registry import SYNTHARA_CONFIG, VEX_CONFIG

pytestmark = pytest.mark.unit


def _make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "jwt_secret_key": "a_valid_test_secret_key_that_is_long_enough_256bits",
        "stripe_secret_key_vex": "sk_test_vex_123",
        "stripe_webhook_secret_vex": "whsec_vex_123",
        "nowpayments_api_key_vex": "np_key_vex_123",
        "nowpayments_ipn_secret_vex": "np_ipn_vex_123",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


class TestCreateStripeCheckout:
    async def test_uses_per_product_key_and_frontend_origin(self) -> None:
        """C6/C7: per-product Stripe key (no global stripe.api_key mutation)
        and success/cancel URLs built from ProductConfig.frontend_origin —
        not the hardcoded https://app.example.com placeholder."""
        settings = _make_settings()
        service = PaymentService(billing_service=AsyncMock(), settings=settings)

        account_id = uuid4()
        mock_repo = AsyncMock()
        mock_repo.get_account = AsyncMock(return_value=MagicMock(id=account_id))
        mock_repo.create_payment = AsyncMock()

        fake_session = MagicMock(
            id="cs_test_abc123", url="https://checkout.stripe.com/pay/cs_test_abc123"
        )
        create_async_mock = AsyncMock(return_value=fake_session)
        captured: dict[str, object] = {}

        def _fake_stripe_client(*, api_key: str) -> MagicMock:
            captured["api_key"] = api_key
            client = MagicMock()
            client.v1.checkout.sessions.create_async = create_async_mock
            return client

        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("stripe.StripeClient", side_effect=_fake_stripe_client),
        ):
            result = await service.create_stripe_checkout(
                account_id,
                100,
                uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

        assert captured["api_key"] == "sk_test_vex_123"
        assert result.checkout_url == fake_session.url
        assert result.session_id == "cs_test_abc123"

        params = create_async_mock.call_args.kwargs["params"]
        assert params["success_url"] == (
            f"{VEX_CONFIG.frontend_origin}{settings.stripe_checkout_success_path}"
        )
        assert params["cancel_url"] == (
            f"{VEX_CONFIG.frontend_origin}{settings.stripe_checkout_cancel_path}"
        )
        assert "example.com" not in params["success_url"]
        assert "example.com" not in params["cancel_url"]

        create_kwargs = mock_repo.create_payment.call_args.kwargs
        assert create_kwargs["product_id"] == "vex"
        assert create_kwargs["external_id"] == "cs_test_abc123"

    async def test_stripe_checkout_charges_total_due_and_grants_full_tokens(self) -> None:
        """D1: amount_usd=100 hits the 5% tier -> total_due=95.00, but
        tokens_granted is the full pre-discount value (100 * tokens_per_usd)."""
        settings = _make_settings()
        service = PaymentService(billing_service=AsyncMock(), settings=settings)

        mock_repo = AsyncMock()
        mock_repo.get_account = AsyncMock(return_value=MagicMock(id=uuid4()))
        mock_repo.create_payment = AsyncMock()

        fake_session = MagicMock(id="cs_test_tier", url="https://checkout.stripe.com/pay/x")
        create_async_mock = AsyncMock(return_value=fake_session)

        def _fake_stripe_client(*, api_key: str) -> MagicMock:
            del api_key
            client = MagicMock()
            client.v1.checkout.sessions.create_async = create_async_mock
            return client

        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("stripe.StripeClient", side_effect=_fake_stripe_client),
        ):
            await service.create_stripe_checkout(
                uuid4(), 100, uuid4(), session=AsyncMock(), product_config=VEX_CONFIG
            )

        params = create_async_mock.call_args.kwargs["params"]
        assert params["line_items"][0]["price_data"]["unit_amount"] == 9500  # $95.00 in cents
        assert params["metadata"]["credits_usd"] == "100"

        create_kwargs = mock_repo.create_payment.call_args.kwargs
        assert create_kwargs["amount_usd"] == Decimal("95.00")
        assert create_kwargs["tokens_granted"] == 100 * settings.billing_tokens_per_usd
        assert create_kwargs["provider_metadata"]["credits_usd"] == 100
        assert create_kwargs["provider_metadata"]["discount_pct"] == 5

    async def test_amount_below_min_rejected(self) -> None:
        settings = _make_settings()
        service = PaymentService(billing_service=AsyncMock(), settings=settings)
        with pytest.raises(ValueError, match="amount_usd must be between"):
            await service.create_stripe_checkout(
                uuid4(),
                settings.billing_min_topup_usd - 1,
                uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

    async def test_amount_above_max_rejected(self) -> None:
        settings = _make_settings()
        service = PaymentService(billing_service=AsyncMock(), settings=settings)
        with pytest.raises(ValueError, match="amount_usd must be between"):
            await service.create_stripe_checkout(
                uuid4(),
                settings.billing_max_topup_usd + 1,
                uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

    async def test_raises_when_account_not_found(self) -> None:
        mock_repo = AsyncMock()
        mock_repo.get_account = AsyncMock(return_value=None)
        service = PaymentService(billing_service=AsyncMock(), settings=_make_settings())
        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            pytest.raises(AccountNotFoundError),
        ):
            await service.create_stripe_checkout(
                uuid4(), 100, uuid4(), session=AsyncMock(), product_config=VEX_CONFIG
            )

    async def test_raises_when_product_key_unconfigured(self) -> None:
        """Fail loud rather than silently borrowing another product's key
        (or a legacy global one) when synthara has no key configured."""
        settings = _make_settings()
        mock_repo = AsyncMock()
        mock_repo.get_account = AsyncMock(return_value=MagicMock(id=uuid4()))
        service = PaymentService(billing_service=AsyncMock(), settings=settings)
        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            pytest.raises(RuntimeError, match="synthara"),
        ):
            await service.create_stripe_checkout(
                uuid4(),
                100,
                uuid4(),
                session=AsyncMock(),
                product_config=SYNTHARA_CONFIG,
            )


class TestCreateNowPaymentsInvoice:
    async def test_uses_per_product_api_key_and_embeds_internal_payment_id(self) -> None:
        """C1 setup half: order_id must embed our internal payment_id so the
        IPN handler can resolve it later (NowPayments' own payment_id, only
        known after the fact, can never be used for this)."""
        settings = _make_settings()
        service = PaymentService(billing_service=AsyncMock(), settings=settings)

        account_id = uuid4()
        mock_repo = AsyncMock()
        mock_repo.get_account = AsyncMock(return_value=MagicMock(id=account_id))
        mock_repo.create_payment = AsyncMock()

        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock()
        fake_response.json = MagicMock(
            return_value={"id": "np_invoice_123", "invoice_url": "https://nowpayments.io/pay/xyz"}
        )

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=fake_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("httpx.AsyncClient", return_value=mock_http_client),
        ):
            result = await service.create_nowpayments_invoice(
                account_id, 100, "btc", uuid4(), session=AsyncMock(), product_id="vex"
            )

        assert result.invoice_url == "https://nowpayments.io/pay/xyz"

        headers = mock_http_client.post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "np_key_vex_123"

        create_kwargs = mock_repo.create_payment.call_args.kwargs
        assert create_kwargs["external_id"] == "np_invoice_123"

        sent_json = mock_http_client.post.call_args.kwargs["json"]
        order = json.loads(sent_json["order_id"])
        assert order["payment_id"] == str(create_kwargs["id"])
        assert "package_id" not in order
        assert order["credits_usd"] == 100

    async def test_nowpayments_invoice_price_is_total_due(self) -> None:
        """D1: amount_usd=100 hits the 5% tier -> price_amount=95.0 (total_due
        as float), tokens_granted stays the full pre-discount value."""
        settings = _make_settings()
        service = PaymentService(billing_service=AsyncMock(), settings=settings)

        account_id = uuid4()
        mock_repo = AsyncMock()
        mock_repo.get_account = AsyncMock(return_value=MagicMock(id=account_id))
        mock_repo.create_payment = AsyncMock()

        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock()
        fake_response.json = MagicMock(
            return_value={"id": "np_invoice_tier", "invoice_url": "https://nowpayments.io/pay/x"}
        )

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=fake_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("httpx.AsyncClient", return_value=mock_http_client),
        ):
            await service.create_nowpayments_invoice(
                account_id, 100, "btc", uuid4(), session=AsyncMock(), product_id="vex"
            )

        sent_json = mock_http_client.post.call_args.kwargs["json"]
        assert sent_json["price_amount"] == 95.0

        create_kwargs = mock_repo.create_payment.call_args.kwargs
        assert create_kwargs["amount_usd"] == Decimal("95.00")
        assert create_kwargs["tokens_granted"] == 100 * settings.billing_tokens_per_usd
        assert create_kwargs["provider_metadata"]["credits_usd"] == 100
        assert create_kwargs["provider_metadata"]["discount_pct"] == 5

    async def test_amount_below_min_rejected(self) -> None:
        settings = _make_settings()
        service = PaymentService(billing_service=AsyncMock(), settings=settings)
        with pytest.raises(ValueError, match="amount_usd must be between"):
            await service.create_nowpayments_invoice(
                uuid4(),
                settings.billing_min_topup_usd - 1,
                "btc",
                uuid4(),
                session=AsyncMock(),
                product_id="vex",
            )

    async def test_amount_above_max_rejected(self) -> None:
        settings = _make_settings()
        service = PaymentService(billing_service=AsyncMock(), settings=settings)
        with pytest.raises(ValueError, match="amount_usd must be between"):
            await service.create_nowpayments_invoice(
                uuid4(),
                settings.billing_max_topup_usd + 1,
                "btc",
                uuid4(),
                session=AsyncMock(),
                product_id="vex",
            )


class TestHandleStripeWebhook:
    async def test_completed_checkout_credits_account(self) -> None:
        settings = _make_settings()

        fake_event = MagicMock()
        fake_event.type = "checkout.session.completed"
        fake_event.id = "evt_123"
        fake_event.data.object = {"id": "cs_test_xyz"}

        payment = MagicMock()
        payment.status = "pending"
        payment.account_id = uuid4()
        payment.tokens_granted = 500
        payment.product_id = "vex"
        payment.provider_metadata = {}

        mock_repo = AsyncMock()
        mock_repo.get_payment_by_external_id_for_update = AsyncMock(return_value=payment)

        billing = AsyncMock()
        service = PaymentService(billing_service=billing, settings=settings)

        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("stripe.Webhook.construct_event", return_value=fake_event),
        ):
            await service.handle_stripe_webhook(b"{}", "sig", session=AsyncMock(), product_id="vex")

        mock_repo.get_payment_by_external_id_for_update.assert_awaited_once_with("cs_test_xyz")
        billing.credit.assert_awaited_once()
        assert payment.status == "completed"

    async def test_uses_per_product_webhook_secret(self) -> None:
        settings = _make_settings()
        service = PaymentService(billing_service=AsyncMock(), settings=settings)
        captured: dict[str, object] = {}

        def _fake_construct_event(payload: bytes, sig: str, secret: str) -> None:  # noqa: ARG001
            captured["secret"] = secret
            raise stripe.SignatureVerificationError(  # type: ignore[no-untyped-call]
                "boom", sig_header=sig
            )

        with (
            patch("stripe.Webhook.construct_event", side_effect=_fake_construct_event),
            pytest.raises(PaymentVerificationError),
        ):
            await service.handle_stripe_webhook(b"{}", "sig", session=AsyncMock(), product_id="vex")

        assert captured["secret"] == "whsec_vex_123"

    async def test_non_checkout_event_is_ignored(self) -> None:
        settings = _make_settings()
        service = PaymentService(billing_service=AsyncMock(), settings=settings)

        fake_event = MagicMock()
        fake_event.type = "payment_intent.succeeded"

        mock_repo = AsyncMock()
        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("stripe.Webhook.construct_event", return_value=fake_event),
        ):
            await service.handle_stripe_webhook(b"{}", "sig", session=AsyncMock(), product_id="vex")

        mock_repo.get_payment_by_external_id_for_update.assert_not_called()

    async def test_already_completed_payment_is_idempotent_noop(self) -> None:
        settings = _make_settings()
        fake_event = MagicMock()
        fake_event.type = "checkout.session.completed"
        fake_event.data.object = {"id": "cs_test_done"}

        payment = MagicMock()
        payment.status = "completed"

        mock_repo = AsyncMock()
        mock_repo.get_payment_by_external_id_for_update = AsyncMock(return_value=payment)

        billing = AsyncMock()
        service = PaymentService(billing_service=billing, settings=settings)

        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("stripe.Webhook.construct_event", return_value=fake_event),
        ):
            await service.handle_stripe_webhook(b"{}", "sig", session=AsyncMock(), product_id="vex")

        billing.credit.assert_not_awaited()

    async def test_stripe_webhook_returns_event_with_owner_targets(self) -> None:
        """R1: the event returned by handle_stripe_webhook must carry the
        credited account's owner as an SSE target — proving credit()'s
        account-based target resolution flows through the payment layer,
        using the real BillingService (not a mock) end to end."""
        settings = _make_settings()

        fake_event = MagicMock()
        fake_event.type = "checkout.session.completed"
        fake_event.id = "evt_123"
        fake_event.data.object = {"id": "cs_test_xyz"}

        owner_user_id = uuid4()
        account = MagicMock()
        account.user_id = owner_user_id
        account.organization_id = None

        payment = MagicMock()
        payment.status = "pending"
        payment.account_id = uuid4()
        payment.tokens_granted = 500
        payment.product_id = "vex"
        payment.provider_metadata = {}

        mock_repo = AsyncMock()
        mock_repo.get_payment_by_external_id_for_update = AsyncMock(return_value=payment)
        mock_repo.get_account_for_update = AsyncMock(return_value=account)
        mock_repo.get_balance = AsyncMock(return_value=0)
        mock_repo.create_transaction = AsyncMock(return_value=MagicMock(id=uuid4()))

        billing = BillingService()
        service = PaymentService(billing_service=billing, settings=settings)

        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("src.api.services.billing.BillingRepository", return_value=mock_repo),
            patch("stripe.Webhook.construct_event", return_value=fake_event),
        ):
            event = await service.handle_stripe_webhook(
                b"{}", "sig", session=AsyncMock(), product_id="vex"
            )

        assert event is not None
        assert event.user_ids == [owner_user_id]


def _sign_nowpayments_payload(raw_payload: bytes, ipn_secret: str) -> str:
    parsed = json.loads(raw_payload, parse_float=str, parse_int=str)
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
    return hmac_module.new(ipn_secret.encode(), canonical, hashlib.sha512).hexdigest()


class TestHandleNowPaymentsWebhook:
    async def test_realistic_payload_resolves_via_order_id_not_ipn_payment_id(self) -> None:
        """C1: the payment is resolved via our internal payment_id embedded in
        order_id — NOT the IPN's top-level payment_id (NowPayments' own id,
        which is never what we stored as external_id). C12: price_amount's
        "10.00" lexeme survives the HMAC round trip."""
        settings = _make_settings()
        internal_payment_id = uuid4()
        account_id = uuid4()

        order_id_str = json.dumps(
            {
                "account_id": str(account_id),
                "package_id": "starter",
                "payment_id": str(internal_payment_id),
            }
        )
        raw_payload = (
            '{"payment_status":"finished","payment_id":"5077125060",'
            f"{json.dumps('order_id')}:{json.dumps(order_id_str)},"
            '"price_amount":10.00}'
        ).encode()
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        payment = MagicMock()
        payment.id = internal_payment_id
        payment.status = "pending"
        payment.account_id = account_id
        payment.tokens_granted = 1000
        payment.product_id = "vex"
        payment.amount_usd = Decimal("10.00")
        payment.provider_metadata = {}

        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)
        mock_repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)

        billing = AsyncMock()
        service = PaymentService(billing_service=billing, settings=settings)

        with patch("src.api.services.payment.BillingRepository", return_value=mock_repo):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        # Resolved by OUR internal id, never NowPayments' "5077125060".
        mock_repo.get_payment_for_update.assert_awaited_once_with(internal_payment_id)
        billing.credit.assert_awaited_once()
        # ratio == 1.0 (price_amount == amount_usd) → full tokens_granted credited.
        assert billing.credit.await_args.args[1] == 1000
        assert payment.status == "completed"

    async def test_order_id_json_parseable_by_webhook_without_package_id(self) -> None:
        """D7/C4: order_id created by the new tier-based invoice flow has no
        package_id at all — the webhook parser must resolve payment_id from
        this new-shape order_id just as readily as from an old-shape one."""
        settings = _make_settings()
        internal_payment_id = uuid4()
        account_id = uuid4()

        order_id_str = json.dumps(
            {
                "account_id": str(account_id),
                "payment_id": str(internal_payment_id),
                "credits_usd": 100,
            }
        )
        raw_payload = (
            '{"payment_status":"finished","payment_id":"5077125061",'
            f"{json.dumps('order_id')}:{json.dumps(order_id_str)},"
            '"price_amount":10.00}'
        ).encode()
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        payment = MagicMock()
        payment.id = internal_payment_id
        payment.status = "pending"
        payment.account_id = account_id
        payment.tokens_granted = 1000
        payment.product_id = "vex"
        payment.amount_usd = Decimal("10.00")
        payment.provider_metadata = {}

        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)
        mock_repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)

        billing = AsyncMock()
        service = PaymentService(billing_service=billing, settings=settings)

        with patch("src.api.services.payment.BillingRepository", return_value=mock_repo):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        mock_repo.get_payment_for_update.assert_awaited_once_with(internal_payment_id)
        billing.credit.assert_awaited_once()
        assert payment.status == "completed"

    async def test_wrong_signature_raises(self) -> None:
        service = PaymentService(billing_service=AsyncMock(), settings=_make_settings())
        with pytest.raises(PaymentVerificationError, match="HMAC"):
            await service.handle_nowpayments_webhook(
                b'{"payment_status":"finished"}',
                "not-the-real-signature",
                session=AsyncMock(),
                product_id="vex",
            )

    async def test_malformed_order_id_raises_verification_error(self) -> None:
        settings = _make_settings()
        service = PaymentService(billing_service=AsyncMock(), settings=settings)

        raw_payload = b'{"payment_status":"finished","payment_id":"123","order_id":"not-json"}'
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        with pytest.raises(PaymentVerificationError, match="order_id"):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

    async def test_already_completed_payment_is_idempotent_noop(self) -> None:
        settings = _make_settings()
        internal_payment_id = uuid4()

        order_id_str = json.dumps(
            {
                "account_id": str(uuid4()),
                "package_id": "starter",
                "payment_id": str(internal_payment_id),
            }
        )
        raw_payload = (
            '{"payment_status":"finished","payment_id":"999",'
            f"{json.dumps('order_id')}:{json.dumps(order_id_str)}}}"
        ).encode()
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        payment = MagicMock()
        payment.id = internal_payment_id
        payment.status = "completed"

        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)

        billing = AsyncMock()
        service = PaymentService(billing_service=billing, settings=settings)

        with patch("src.api.services.payment.BillingRepository", return_value=mock_repo):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        billing.credit.assert_not_awaited()

    async def test_payment_not_found_logs_and_returns_without_raising(self) -> None:
        settings = _make_settings()
        order_id_str = json.dumps(
            {"account_id": str(uuid4()), "package_id": "starter", "payment_id": str(uuid4())}
        )
        raw_payload = (
            '{"payment_status":"finished","payment_id":"999",'
            f"{json.dumps('order_id')}:{json.dumps(order_id_str)}}}"
        ).encode()
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=None)

        billing = AsyncMock()
        service = PaymentService(billing_service=billing, settings=settings)

        with patch("src.api.services.payment.BillingRepository", return_value=mock_repo):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        billing.credit.assert_not_awaited()


# ---------------------------------------------------------------------------
# D2 — proportional-credit IPN policy (F2b/F2c)
# ---------------------------------------------------------------------------


def _make_ipn_raw_payload(
    *,
    payment_status: str,
    internal_payment_id: object,
    np_payment_id: str = "999",
    actually_paid: str | None = None,
    price_amount: str | None = None,
) -> bytes:
    """Build a raw IPN payload with numeric literals preserved verbatim.

    Amount fields are inserted as raw JSON number literals (not through
    ``json.dumps`` of the whole payload) so the "10.00"-style lexeme survives
    exactly as NowPayments would send it — matching how the HMAC canonicalizes
    the real payload (``parse_float=str``).
    """
    order_id_str = json.dumps(
        {
            "account_id": str(uuid4()),
            "package_id": "starter",
            "payment_id": str(internal_payment_id),
        }
    )
    parts = [
        f'"payment_status":{json.dumps(payment_status)}',
        f'"payment_id":{json.dumps(np_payment_id)}',
        f"{json.dumps('order_id')}:{json.dumps(order_id_str)}",
    ]
    if actually_paid is not None:
        parts.append(f'"actually_paid":{actually_paid}')
    if price_amount is not None:
        parts.append(f'"price_amount":{price_amount}')
    return ("{" + ",".join(parts) + "}").encode()


def _make_ipn_payment(
    *,
    internal_payment_id: object,
    tokens_granted: int = 1000,
    amount_usd: Decimal = Decimal("10.00"),
    status: str = "pending",
    product_id: str = "vex",
) -> MagicMock:
    payment = MagicMock()
    payment.id = internal_payment_id
    payment.status = status
    payment.account_id = uuid4()
    payment.tokens_granted = tokens_granted
    payment.amount_usd = amount_usd
    payment.product_id = product_id
    payment.provider_metadata = {}
    return payment


class TestNowPaymentsIPNProportionalCreditPolicy:
    """D2: automatic proportional crediting for partial/under/over payment —
    never a hold-for-review state. Covers the telescoping delta-credit
    contract and the tolerance/extreme-ratio bands."""

    async def test_ipn_product_mismatch_raises_verification_error(self) -> None:
        settings = _make_settings()
        internal_payment_id = uuid4()
        payment = _make_ipn_payment(internal_payment_id=internal_payment_id, product_id="synthara")
        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)

        raw_payload = _make_ipn_raw_payload(
            payment_status="finished",
            internal_payment_id=internal_payment_id,
            actually_paid="10.00",
        )
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        service = PaymentService(billing_service=AsyncMock(), settings=settings)
        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            pytest.raises(PaymentVerificationError, match="product mismatch"),
        ):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

    async def test_ipn_zero_amount_usd_raises_verification_error(self) -> None:
        settings = _make_settings()
        internal_payment_id = uuid4()
        payment = _make_ipn_payment(
            internal_payment_id=internal_payment_id, amount_usd=Decimal("0.00")
        )
        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)

        raw_payload = _make_ipn_raw_payload(
            payment_status="finished",
            internal_payment_id=internal_payment_id,
            actually_paid="10.00",
        )
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        service = PaymentService(billing_service=AsyncMock(), settings=settings)
        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            pytest.raises(PaymentVerificationError),
        ):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

    async def test_ipn_partially_paid_credits_proportional_and_sets_status(self) -> None:
        settings = _make_settings()
        internal_payment_id = uuid4()
        payment = _make_ipn_payment(internal_payment_id=internal_payment_id, tokens_granted=1000)
        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)
        mock_repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)

        raw_payload = _make_ipn_raw_payload(
            payment_status="partially_paid",
            internal_payment_id=internal_payment_id,
            actually_paid="4.00",  # ratio 0.4 of amount_usd=10.00
        )
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))
        service = PaymentService(billing_service=billing, settings=settings)

        with patch("src.api.services.payment.BillingRepository", return_value=mock_repo):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        billing.credit.assert_awaited_once()
        assert billing.credit.await_args.args[1] == 400
        assert payment.status == PaymentStatus.PARTIALLY_PAID.value

    async def test_ipn_partially_paid_redelivery_credits_zero_delta(self) -> None:
        """Same IPN delivered twice → the second delivery credits nothing."""
        settings = _make_settings()
        internal_payment_id = uuid4()
        payment = _make_ipn_payment(internal_payment_id=internal_payment_id, tokens_granted=1000)
        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)
        # First delivery: nothing credited yet. Second: the 400 from the
        # first delivery is now visible in the ledger.
        mock_repo.get_credited_tokens_for_payment = AsyncMock(side_effect=[0, 400])

        raw_payload = _make_ipn_raw_payload(
            payment_status="partially_paid",
            internal_payment_id=internal_payment_id,
            actually_paid="4.00",
        )
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))
        service = PaymentService(billing_service=billing, settings=settings)

        with patch("src.api.services.payment.BillingRepository", return_value=mock_repo):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        billing.credit.assert_awaited_once()  # only the first delivery credited

    async def test_ipn_finished_after_partial_credits_exact_remainder(self) -> None:
        """partial 40% → finished 100% → total credited == tokens_granted, no drift."""
        settings = _make_settings()
        internal_payment_id = uuid4()
        payment = _make_ipn_payment(internal_payment_id=internal_payment_id, tokens_granted=1000)
        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)
        mock_repo.get_credited_tokens_for_payment = AsyncMock(side_effect=[0, 400])

        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))
        service = PaymentService(billing_service=billing, settings=settings)

        partial_payload = _make_ipn_raw_payload(
            payment_status="partially_paid",
            internal_payment_id=internal_payment_id,
            actually_paid="4.00",
        )
        partial_sig = _sign_nowpayments_payload(partial_payload, "np_ipn_vex_123")

        finished_payload = _make_ipn_raw_payload(
            payment_status="finished",
            internal_payment_id=internal_payment_id,
            actually_paid="10.00",
        )
        finished_sig = _sign_nowpayments_payload(finished_payload, "np_ipn_vex_123")

        with patch("src.api.services.payment.BillingRepository", return_value=mock_repo):
            await service.handle_nowpayments_webhook(
                partial_payload, partial_sig, session=AsyncMock(), product_id="vex"
            )
            assert payment.status == PaymentStatus.PARTIALLY_PAID.value

            # PARTIALLY_PAID is non-terminal — only COMPLETED blocks reprocessing,
            # so the "finished" IPN below is processed normally.
            await service.handle_nowpayments_webhook(
                finished_payload, finished_sig, session=AsyncMock(), product_id="vex"
            )

        assert billing.credit.await_count == 2
        first_delta = billing.credit.await_args_list[0].args[1]
        second_delta = billing.credit.await_args_list[1].args[1]
        assert first_delta == 400
        assert second_delta == 600
        assert first_delta + second_delta == payment.tokens_granted
        assert payment.status == PaymentStatus.COMPLETED.value

    async def test_ipn_underpaid_finished_credits_proportional_and_completes(self) -> None:
        settings = _make_settings()
        internal_payment_id = uuid4()
        payment = _make_ipn_payment(internal_payment_id=internal_payment_id, tokens_granted=1000)
        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)
        mock_repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)

        raw_payload = _make_ipn_raw_payload(
            payment_status="finished",
            internal_payment_id=internal_payment_id,
            actually_paid="7.00",  # ratio 0.7 — below tolerance, above extreme floor
        )
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))
        service = PaymentService(billing_service=billing, settings=settings)

        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("src.api.services.payment.logger") as mock_logger,
        ):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        assert billing.credit.await_args.args[1] == 700
        assert payment.status == PaymentStatus.COMPLETED.value
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args.args[0] == "payment.underpaid_credited"

    @pytest.mark.parametrize("actually_paid", ["9.95", "10.05"])
    async def test_ipn_ratio_within_band_snaps_to_full_tokens(self, actually_paid: str) -> None:
        """Ratios in [0.99, 1.01] snap to fully paid, no warning."""
        settings = _make_settings()
        internal_payment_id = uuid4()
        payment = _make_ipn_payment(internal_payment_id=internal_payment_id, tokens_granted=1000)
        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)
        mock_repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)

        raw_payload = _make_ipn_raw_payload(
            payment_status="finished",
            internal_payment_id=internal_payment_id,
            actually_paid=actually_paid,
        )
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))
        service = PaymentService(billing_service=billing, settings=settings)

        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("src.api.services.payment.logger") as mock_logger,
        ):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        assert billing.credit.await_args.args[1] == 1000  # full tokens_granted, no drift
        assert payment.status == PaymentStatus.COMPLETED.value
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    async def test_ipn_overpaid_credits_proportional(self) -> None:
        settings = _make_settings()
        internal_payment_id = uuid4()
        payment = _make_ipn_payment(internal_payment_id=internal_payment_id, tokens_granted=1000)
        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)
        mock_repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)

        raw_payload = _make_ipn_raw_payload(
            payment_status="finished",
            internal_payment_id=internal_payment_id,
            actually_paid="15.00",  # ratio 1.5 — uncapped overpayment
        )
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))
        service = PaymentService(billing_service=billing, settings=settings)

        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("src.api.services.payment.logger") as mock_logger,
        ):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        assert billing.credit.await_args.args[1] == 1500  # floor(1000 * 1.5), uncapped
        assert payment.status == PaymentStatus.COMPLETED.value
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args.args[0] == "payment.overpaid_credited"

    @pytest.mark.parametrize("actually_paid", ["2.00", "25.00"])
    async def test_ipn_extreme_ratio_logs_error_but_credits(self, actually_paid: str) -> None:
        """Ratio < 0.5 or > 2.0 → same proportional credit, but logged at error
        (ops attention signal) — still credited, never held."""
        settings = _make_settings()
        internal_payment_id = uuid4()
        payment = _make_ipn_payment(internal_payment_id=internal_payment_id, tokens_granted=1000)
        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)
        mock_repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)

        raw_payload = _make_ipn_raw_payload(
            payment_status="finished",
            internal_payment_id=internal_payment_id,
            actually_paid=actually_paid,
        )
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))
        service = PaymentService(billing_service=billing, settings=settings)

        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("src.api.services.payment.logger") as mock_logger,
        ):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        billing.credit.assert_awaited_once()
        assert billing.credit.await_args.args[1] > 0
        assert payment.status == PaymentStatus.COMPLETED.value
        mock_logger.error.assert_called_once()
        mock_logger.warning.assert_not_called()

    async def test_ipn_unknown_status_logs_and_keeps_status(self) -> None:
        settings = _make_settings()
        internal_payment_id = uuid4()
        payment = _make_ipn_payment(
            internal_payment_id=internal_payment_id, status=PaymentStatus.PENDING.value
        )
        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)

        raw_payload = _make_ipn_raw_payload(
            payment_status="some_future_status_we_dont_know",
            internal_payment_id=internal_payment_id,
        )
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        billing = AsyncMock()
        service = PaymentService(billing_service=billing, settings=settings)

        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("src.api.services.payment.logger") as mock_logger,
        ):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        billing.credit.assert_not_awaited()
        assert payment.status == PaymentStatus.PENDING.value  # unchanged, not silently altered
        mock_logger.warning.assert_called_once_with(
            "payment.ipn_unknown_status",
            payment_id=str(payment.id),
            raw_status="some_future_status_we_dont_know",
        )

    async def test_ipn_credit_returns_event_with_owner_targets(self) -> None:
        """R1: the event returned from an IPN credit must carry the account
        owner as an SSE target — real BillingService, not a mock."""
        settings = _make_settings()
        internal_payment_id = uuid4()
        owner_user_id = uuid4()
        account = MagicMock()
        account.user_id = owner_user_id
        account.organization_id = None

        payment = _make_ipn_payment(internal_payment_id=internal_payment_id, tokens_granted=1000)

        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)
        mock_repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)
        mock_repo.get_account_for_update = AsyncMock(return_value=account)
        mock_repo.get_balance = AsyncMock(return_value=0)
        mock_repo.create_transaction = AsyncMock(return_value=MagicMock(id=uuid4()))

        raw_payload = _make_ipn_raw_payload(
            payment_status="finished",
            internal_payment_id=internal_payment_id,
            actually_paid="10.00",
        )
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        billing = BillingService()
        service = PaymentService(billing_service=billing, settings=settings)

        with (
            patch("src.api.services.payment.BillingRepository", return_value=mock_repo),
            patch("src.api.services.billing.BillingRepository", return_value=mock_repo),
        ):
            event = await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        assert event is not None
        assert event.user_ids == [owner_user_id]

    @pytest.mark.parametrize(
        "status,paid",
        [("finished", "10.00"), ("partially_paid", "4.00")],
    )
    async def test_ipn_raw_payload_persisted_on_finished_and_partial(
        self, status: str, paid: str
    ) -> None:
        """R5: the raw IPN payload must be persisted alongside the extracted
        fields on both the finished and partially_paid branches — reconciliation
        needs the unknown-questions data the extracted fields can't answer."""
        settings = _make_settings()
        internal_payment_id = uuid4()
        payment = _make_ipn_payment(internal_payment_id=internal_payment_id, tokens_granted=1000)
        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)
        mock_repo.get_credited_tokens_for_payment = AsyncMock(return_value=0)

        raw_payload = _make_ipn_raw_payload(
            payment_status=status,
            internal_payment_id=internal_payment_id,
            actually_paid=paid,
        )
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        billing = AsyncMock()
        billing.credit = AsyncMock(return_value=MagicMock(event=None))
        service = PaymentService(billing_service=billing, settings=settings)

        with patch("src.api.services.payment.BillingRepository", return_value=mock_repo):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        assert "ipn_payload" in payment.provider_metadata
        assert payment.provider_metadata["ipn_payload"]["payment_status"] == status

    async def test_ipn_raw_payload_persisted_on_intermediate_status(self) -> None:
        """R5: even a non-crediting intermediate/terminal status (e.g. a
        failed/expired IPN) must persist the raw payload — support needs it
        for investigation even when nothing was credited."""
        settings = _make_settings()
        internal_payment_id = uuid4()
        payment = _make_ipn_payment(
            internal_payment_id=internal_payment_id, status=PaymentStatus.PENDING.value
        )
        mock_repo = AsyncMock()
        mock_repo.get_payment_for_update = AsyncMock(return_value=payment)

        raw_payload = _make_ipn_raw_payload(
            payment_status="waiting",
            internal_payment_id=internal_payment_id,
        )
        signature = _sign_nowpayments_payload(raw_payload, "np_ipn_vex_123")

        billing = AsyncMock()
        service = PaymentService(billing_service=billing, settings=settings)

        with patch("src.api.services.payment.BillingRepository", return_value=mock_repo):
            await service.handle_nowpayments_webhook(
                raw_payload, signature, session=AsyncMock(), product_id="vex"
            )

        billing.credit.assert_not_awaited()
        assert "ipn_payload" in payment.provider_metadata
        assert payment.provider_metadata["ipn_payload"]["payment_status"] == "waiting"
