"""NowPayments HMAC and status normalization tests."""

import hashlib
import hmac
import json
from uuid import uuid4

import pytest

from src.api.services.billing_errors import PaymentVerificationError
from src.api.services.payments.contracts import WebhookEnvelope
from src.api.services.payments.nowpayments_gateway import NowPaymentsGateway
from src.core.config import Settings
from src.core.enums import PaymentStatus

pytestmark = pytest.mark.unit

_SECRET = "ipn-secret"


def _settings() -> Settings:
    return Settings(
        jwt_secret_key="test-secret-key-that-is-definitely-long-enough-32bytes",
        nowpayments_ipn_secret_vex=_SECRET,
    )


def _envelope(status: str, *, signature: str | None = None) -> WebhookEnvelope:
    payment_id = uuid4()
    order_id = json.dumps({"payment_id": str(payment_id)})
    raw = (
        f'{{"payment_status":"{status}","payment_id":"np-1",'
        f'"order_id":{json.dumps(order_id)},"actually_paid":10.00}}'
    ).encode()
    parsed = json.loads(raw, parse_float=str, parse_int=str)
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
    valid_signature = hmac.new(_SECRET.encode(), canonical, hashlib.sha512).hexdigest()
    return WebhookEnvelope(
        raw_body=raw,
        headers={"x-nowpayments-sig": signature or valid_signature},
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


async def test_bad_signature_raises() -> None:
    with pytest.raises(PaymentVerificationError, match="HMAC"):
        await NowPaymentsGateway(_settings()).verify_webhook(_envelope("finished", signature="bad"))


async def test_malformed_order_id_raises() -> None:
    raw = b'{"order_id":"bad","payment_status":"finished"}'
    parsed = json.loads(raw, parse_float=str, parse_int=str)
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(_SECRET.encode(), canonical, hashlib.sha512).hexdigest()
    with pytest.raises(PaymentVerificationError, match="order_id"):
        await NowPaymentsGateway(_settings()).verify_webhook(
            WebhookEnvelope(
                raw_body=raw,
                headers={"x-nowpayments-sig": signature},
                product_id="vex",
            )
        )
