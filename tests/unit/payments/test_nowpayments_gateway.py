"""NowPayments HMAC, status normalization, and settlement-field extraction tests."""

import hashlib
import hmac
import json
import os
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.services.billing_errors import PaymentVerificationError, PaymentVerificationReason
from src.api.services.payments.contracts import ChargeContext, ChargeResult, WebhookEnvelope
from src.api.services.payments.ipn_canonical import canonical_bytes, parse_ipn_body
from src.api.services.payments.nowpayments_gateway import NowPaymentsGateway
from src.core.config import Settings
from src.core.enums import PaymentStatus
from src.core.product_registry import VEX_CONFIG
from src.core.topup_pricing import build_quote, topup_tiers_for

pytestmark = pytest.mark.unit

_SECRET = "ipn-secret"


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "jwt_secret_key": "test-secret-key-that-is-definitely-long-enough-32bytes",
        "nowpayments_ipn_secret_vex": _SECRET,
        "nowpayments_api_key_vex": "np_key_vex_123",
    } | overrides
    return Settings(**defaults)  # type: ignore[arg-type]


def _raw_payload(
    *,
    status: str,
    actually_paid: str | None = "10.00",
    pay_amount: str | None = "10.00",
    include_actually_paid: bool = True,
    include_pay_amount: bool = True,
    pay_currency: str | None = None,
) -> bytes:
    payment_id = uuid4()
    order_id = json.dumps({"payment_id": str(payment_id)})
    parts = [
        f'"payment_status":{json.dumps(status)}',
        '"payment_id":"np-1"',
        f"{json.dumps('order_id')}:{json.dumps(order_id)}",
    ]
    if include_actually_paid:
        parts.append(f'"actually_paid":{actually_paid}')
    if include_pay_amount:
        parts.append(f'"pay_amount":{pay_amount}')
    if pay_currency is not None:
        parts.append(f'"pay_currency":{json.dumps(pay_currency)}')
    return ("{" + ",".join(parts) + "}").encode()


def _sign(raw: bytes) -> str:
    """Sign over the sorted canonical form — exercises the D1 fallback path.

    Deliberately not the raw-body fast path (D2): the hand-built payloads in
    this file assemble fields in a fixed, non-sorted wire order, so signing
    over canonical bytes (not raw wire bytes) is what a real dashboard
    delivery whose signed order differs from wire order looks like.
    """
    canonical = canonical_bytes(parse_ipn_body(raw))
    return hmac.new(_SECRET.encode(), canonical, hashlib.sha512).hexdigest()


def _envelope(
    status: str,
    *,
    signature: str | None = None,
    actually_paid: str | None = "10.00",
    pay_amount: str | None = "10.00",
    include_actually_paid: bool = True,
    include_pay_amount: bool = True,
    pay_currency: str | None = None,
) -> WebhookEnvelope:
    raw = _raw_payload(
        status=status,
        actually_paid=actually_paid,
        pay_amount=pay_amount,
        include_actually_paid=include_actually_paid,
        include_pay_amount=include_pay_amount,
        pay_currency=pay_currency,
    )
    return WebhookEnvelope(
        raw_body=raw,
        headers={"x-nowpayments-sig": signature or _sign(raw)},
        product_id="vex",
    )


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        ("waiting", PaymentStatus.PENDING),
        ("confirming", PaymentStatus.PENDING),
        ("sending", PaymentStatus.PENDING),
        ("partially_paid", PaymentStatus.PARTIALLY_PAID),
        ("failed", PaymentStatus.FAILED),
        ("expired", PaymentStatus.FAILED),
        ("refunded", PaymentStatus.REFUNDED),
        ("finished", PaymentStatus.COMPLETED),
        ("future_status", None),
    ],
)
async def test_status_mapping(raw_status: str, expected: PaymentStatus | None) -> None:
    outcome = await NowPaymentsGateway(_settings()).verify_webhook(_envelope(raw_status))
    assert outcome.status is expected


async def test_float_lexeme_is_preserved_for_hmac() -> None:
    outcome = await NowPaymentsGateway(_settings()).verify_webhook(_envelope("finished"))
    assert outcome.amount_paid is not None
    assert str(outcome.amount_paid) == "10.00"
    assert outcome.amount_due is not None
    assert str(outcome.amount_due) == "10.00"


async def test_raw_body_fast_path_verifies_without_canonicalization() -> None:
    """D2: a signature computed directly over the raw wire bytes (no key
    reordering) verifies via the fast path, without needing the canonical
    fallback at all."""
    raw = _raw_payload(status="finished")
    signature = hmac.new(_SECRET.encode(), raw, hashlib.sha512).hexdigest()
    outcome = await NowPaymentsGateway(_settings()).verify_webhook(
        WebhookEnvelope(raw_body=raw, headers={"x-nowpayments-sig": signature}, product_id="vex")
    )
    assert outcome.status is PaymentStatus.COMPLETED


@pytest.mark.parametrize(
    ("actually_paid", "pay_amount"),
    [
        ("10.00", "10.00"),  # bare JSON number lexeme (RawNumber)
        ('"10.00"', '"10.00"'),  # quoted JSON string (All-Strings)
    ],
)
async def test_amount_extraction_matches_for_bare_and_quoted_lexemes(
    actually_paid: str, pay_amount: str
) -> None:
    """D3: RawNumber and quoted-string amounts convert to the same Decimal."""
    envelope = _envelope("finished", actually_paid=actually_paid, pay_amount=pay_amount)
    outcome = await NowPaymentsGateway(_settings()).verify_webhook(envelope)
    assert outcome.amount_paid == Decimal("10.00")
    assert outcome.amount_due == Decimal("10.00")


async def test_bad_signature_raises() -> None:
    with pytest.raises(PaymentVerificationError, match="HMAC") as exc_info:
        await NowPaymentsGateway(_settings()).verify_webhook(_envelope("finished", signature="bad"))
    exc = exc_info.value
    assert exc.reason is PaymentVerificationReason.SIGNATURE_MISMATCH
    assert exc.context["received_sig_prefix"] == "bad"
    assert len(exc.context["computed_sig_prefix"]) == 8
    assert len(exc.context["body_sha256_prefix"]) == 16
    assert exc.context["secret_source"] == "per_product"
    assert exc.context["raw_path_checked"] == "true"
    assert "payment_status" in exc.context["payload_keys"]


async def test_missing_signature_header_raises() -> None:
    raw = _raw_payload(status="finished")
    envelope = WebhookEnvelope(raw_body=raw, headers={}, product_id="vex")
    with pytest.raises(PaymentVerificationError, match="signature header missing") as exc_info:
        await NowPaymentsGateway(_settings()).verify_webhook(envelope)
    exc = exc_info.value
    assert exc.reason is PaymentVerificationReason.MISSING_SIGNATURE_HEADER
    assert "received_sig_prefix" not in exc.context
    assert exc.context["body_len"] == str(len(raw))


async def test_invalid_json_raises() -> None:
    envelope = WebhookEnvelope(
        raw_body=b"not json", headers={"x-nowpayments-sig": "irrelevant"}, product_id="vex"
    )
    with pytest.raises(PaymentVerificationError, match="not valid JSON") as exc_info:
        await NowPaymentsGateway(_settings()).verify_webhook(envelope)
    exc = exc_info.value
    assert exc.reason is PaymentVerificationReason.MALFORMED_JSON
    assert exc.context["error"]
    assert exc.context["body_len"] == "8"


async def test_non_object_json_raises() -> None:
    raw = b"[1,2,3]"
    envelope = WebhookEnvelope(
        raw_body=raw, headers={"x-nowpayments-sig": "irrelevant"}, product_id="vex"
    )
    with pytest.raises(PaymentVerificationError, match="JSON object") as exc_info:
        await NowPaymentsGateway(_settings()).verify_webhook(envelope)
    exc = exc_info.value
    assert exc.reason is PaymentVerificationReason.MALFORMED_JSON
    assert exc.context["parsed_type"] == "list"


async def test_malformed_order_id_raises() -> None:
    raw = b'{"order_id":"bad","payment_status":"finished"}'
    signature = _sign(raw)
    with pytest.raises(PaymentVerificationError, match="order_id") as exc_info:
        await NowPaymentsGateway(_settings()).verify_webhook(
            WebhookEnvelope(
                raw_body=raw,
                headers={"x-nowpayments-sig": signature},
                product_id="vex",
            )
        )
    exc = exc_info.value
    assert exc.reason is PaymentVerificationReason.MALFORMED_ORDER_ID
    assert exc.context["error_type"] == "JSONDecodeError"


@pytest.mark.parametrize("status", ["finished", "partially_paid"])
async def test_missing_actually_paid_on_settled_status_raises(status: str) -> None:
    """P1-1: no silent full-credit fallback to price_amount when actually_paid
    is absent from a COMPLETED/PARTIALLY_PAID IPN."""
    envelope = _envelope(status, include_actually_paid=False)
    with pytest.raises(PaymentVerificationError, match="actually_paid/pay_amount") as exc_info:
        await NowPaymentsGateway(_settings()).verify_webhook(envelope)
    assert exc_info.value.reason is PaymentVerificationReason.AMOUNT_FIELDS_INVALID
    assert exc_info.value.context["raw_status"] == status


@pytest.mark.parametrize("status", ["finished", "partially_paid"])
async def test_missing_pay_amount_on_settled_status_raises(status: str) -> None:
    envelope = _envelope(status, include_pay_amount=False)
    with pytest.raises(PaymentVerificationError, match="actually_paid/pay_amount") as exc_info:
        await NowPaymentsGateway(_settings()).verify_webhook(envelope)
    assert exc_info.value.reason is PaymentVerificationReason.AMOUNT_FIELDS_INVALID


async def test_zero_pay_amount_raises() -> None:
    envelope = _envelope("finished", pay_amount="0")
    with pytest.raises(PaymentVerificationError, match="positive") as exc_info:
        await NowPaymentsGateway(_settings()).verify_webhook(envelope)
    assert exc_info.value.reason is PaymentVerificationReason.AMOUNT_FIELDS_INVALID


async def test_negative_pay_amount_raises() -> None:
    envelope = _envelope("finished", pay_amount="-1.00")
    with pytest.raises(PaymentVerificationError, match="positive") as exc_info:
        await NowPaymentsGateway(_settings()).verify_webhook(envelope)
    assert exc_info.value.reason is PaymentVerificationReason.AMOUNT_FIELDS_INVALID


async def test_malformed_amount_fields_raise() -> None:
    envelope = _envelope("finished", actually_paid='"not-a-number"')
    with pytest.raises(PaymentVerificationError, match="malformed") as exc_info:
        await NowPaymentsGateway(_settings()).verify_webhook(envelope)
    assert exc_info.value.reason is PaymentVerificationReason.AMOUNT_FIELDS_INVALID


async def test_intermediate_status_does_not_require_amount_fields() -> None:
    """PENDING statuses (waiting/confirming/sending) never carry settlement
    amounts — the fail-loud requirement is scoped to COMPLETED/PARTIALLY_PAID."""
    envelope = _envelope("waiting", include_actually_paid=False, include_pay_amount=False)
    outcome = await NowPaymentsGateway(_settings()).verify_webhook(envelope)
    assert outcome.status is PaymentStatus.PENDING
    assert outcome.amount_paid is None
    assert outcome.amount_due is None


@pytest.mark.parametrize("status", ["finished", "partially_paid"])
async def test_raw_ipn_payload_persisted_in_metadata(status: str) -> None:
    outcome = await NowPaymentsGateway(_settings()).verify_webhook(_envelope(status))
    assert "ipn_payload" in outcome.metadata_patch
    assert outcome.metadata_patch["ipn_payload"]["payment_status"] == status


async def test_raw_ipn_payload_persisted_on_intermediate_status() -> None:
    outcome = await NowPaymentsGateway(_settings()).verify_webhook(
        _envelope("waiting", include_actually_paid=False, include_pay_amount=False)
    )
    assert "ipn_payload" in outcome.metadata_patch
    assert outcome.metadata_patch["ipn_payload"]["payment_status"] == "waiting"


class TestCreateCharge:
    async def test_uses_per_product_api_key_and_embeds_internal_payment_id(self) -> None:
        settings = _settings()
        account_id = uuid4()
        payment_id = uuid4()
        quote = build_quote(
            100,
            tiers=topup_tiers_for("vex", settings),
            tokens_per_usd=settings.billing_tokens_per_usd,
        )

        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock()
        fake_response.json = MagicMock(
            return_value={"id": "np_invoice_123", "invoice_url": "https://nowpayments.io/pay/xyz"}
        )
        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=fake_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        gateway = NowPaymentsGateway(settings, client_factory=lambda: mock_http_client)
        result = await gateway.create_charge(
            ChargeContext(
                payment_id=payment_id,
                account_id=account_id,
                user_id=uuid4(),
                product_config=VEX_CONFIG,
                quote=quote,
                extra={"pay_currency": "btc"},
            )
        )

        assert result.external_id == "np_invoice_123"
        assert result.redirect_url == "https://nowpayments.io/pay/xyz"

        headers = mock_http_client.post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "np_key_vex_123"

        sent_json = mock_http_client.post.call_args.kwargs["json"]
        assert sent_json["price_amount"] == float(quote.total_due)
        order = json.loads(sent_json["order_id"])
        assert order["payment_id"] == str(payment_id)
        assert order["account_id"] == str(account_id)
        assert "package_id" not in order

    @staticmethod
    async def _create_charge(extra: dict[str, str]) -> tuple[dict[str, object], ChargeResult]:
        """Build a gateway + mock client, run create_charge, return (sent_json, result)."""
        settings = _settings()
        quote = build_quote(
            100,
            tiers=topup_tiers_for("vex", settings),
            tokens_per_usd=settings.billing_tokens_per_usd,
        )
        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock()
        fake_response.json = MagicMock(
            return_value={"id": "np_invoice_123", "invoice_url": "https://nowpayments.io/pay/xyz"}
        )
        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=fake_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        gateway = NowPaymentsGateway(settings, client_factory=lambda: mock_http_client)
        result = await gateway.create_charge(
            ChargeContext(
                payment_id=uuid4(),
                account_id=uuid4(),
                user_id=uuid4(),
                product_config=VEX_CONFIG,
                quote=quote,
                extra=extra,
            )
        )
        return mock_http_client.post.call_args.kwargs["json"], result

    async def test_pinned_currency_is_included_and_uppercased_on_result(self) -> None:
        sent_json, result = await self._create_charge({"pay_currency": "usdcmatic"})
        assert sent_json["pay_currency"] == "usdcmatic"
        assert result.currency == "USDCMATIC"

    @pytest.mark.parametrize("extra", [{}, {"pay_currency": ""}, {"pay_currency": "  "}])
    async def test_unset_currency_omits_key_from_payload(self, extra: dict[str, str]) -> None:
        sent_json, result = await self._create_charge(extra)
        assert "pay_currency" not in sent_json
        assert result.currency == "USD"


async def test_ipn_pay_currency_sets_settled_currency() -> None:
    outcome = await NowPaymentsGateway(_settings()).verify_webhook(
        _envelope("finished", pay_currency="usdttrc20")
    )
    assert outcome.settled_currency == "USDTTRC20"
    assert outcome.metadata_patch["ipn_pay_currency"] == "USDTTRC20"


async def test_ipn_without_pay_currency_leaves_settled_currency_none() -> None:
    outcome = await NowPaymentsGateway(_settings()).verify_webhook(_envelope("finished"))
    assert outcome.settled_currency is None
    assert "ipn_pay_currency" not in outcome.metadata_patch


async def test_ipn_pay_currency_extracted_on_intermediate_status() -> None:
    outcome = await NowPaymentsGateway(_settings()).verify_webhook(
        _envelope(
            "waiting",
            include_actually_paid=False,
            include_pay_amount=False,
            pay_currency="btc",
        )
    )
    assert outcome.settled_currency == "BTC"


# --- Incident regression: real captured IPN body + signature -----------------
#
# The exact wire bytes and x-nowpayments-sig from the incident that motivated
# this canonicalizer (agent_prompts/capture.json / np-request.txt) — a mixed
# Classic/nested body that the pre-fix `parse_float=str, parse_int=str`
# canonicalization rejected as SIGNATURE_MISMATCH. The secret is never
# committed; the test is skipped when it isn't supplied via env.
_INCIDENT_BODY = (
    b'{"actually_paid":4.98842,"actually_paid_at_fiat":0,"fee":{"currency":"usdtmatic",'
    b'"depositFee":"0.034133","serviceFee":"0.04883","withdrawalFee":"0.068969"},'
    b'"invoice_id":5872157465,"order_description":"500 tokens ($5 credit, 0% off)",'
    b'"order_id":"{\\"account_id\\": \\"019e1653-0483-784b-b172-958f5eb53f20\\", '
    b'\\"payment_id\\": \\"019f751d-2af1-7b50-a156-d2aeef629802\\", \\"credits_usd\\": 5}",'
    b'"outcome_amount":4.834165,"outcome_currency":"usdtmatic","parent_payment_id":null,'
    b'"pay_address":"0xc2Ad29a67e7576a4C980564B726e587f62cbbB16","pay_amount":4.98842014,'
    b'"pay_currency":"usdcbsc","payin_extra_id":null,"payment_extra_ids":null,'
    b'"payment_id":5954418507,"payment_status":"finished","price_amount":5,'
    b'"price_currency":"usd","purchase_id":"4787968135"}'
)
_INCIDENT_SIGNATURE = (
    "1b4476ddfb4c91379788056a93aa8f280a328a8dc36706a81924799f050fbfa8b423de8ca929b6278d"
    "14d69a80ef07a55e5774bf84bbddee1c41e41d29d6994c"
)


@pytest.mark.skipif(
    "NOWPAYMENTS_INCIDENT_IPN_SECRET" not in os.environ,
    reason="requires NOWPAYMENTS_INCIDENT_IPN_SECRET (the real per-product IPN secret)",
)
async def test_incident_capture_verifies() -> None:
    secret = os.environ["NOWPAYMENTS_INCIDENT_IPN_SECRET"]
    outcome = await NowPaymentsGateway(_settings(nowpayments_ipn_secret_vex=secret)).verify_webhook(
        WebhookEnvelope(
            raw_body=_INCIDENT_BODY,
            headers={"x-nowpayments-sig": _INCIDENT_SIGNATURE},
            product_id="vex",
        )
    )
    assert outcome.status is PaymentStatus.COMPLETED
    assert outcome.amount_paid == Decimal("4.98842")
    assert outcome.amount_due == Decimal("4.98842014")
    assert outcome.settled_currency == "USDCBSC"
