"""Integration test for PaymentService webhook row-lock concurrency (C2).

Two concurrent Stripe webhook deliveries for the same checkout.session.completed
event must credit exactly once. Before the fix, both deliveries would read
status='pending' via an unlocked SELECT, both pass the "not yet completed"
check, and both credit — a double token grant.

The fix: BillingRepository.get_payment_by_external_id_for_update() acquires a
row-level lock (FOR UPDATE) on the payment, and the status check happens
under that lock, so concurrent deliveries serialize instead of racing.

Self-contained — does not use the standard ``db_session`` SAVEPOINT fixture
because that fixture only allocates one connection and we need two truly
independent connections to demonstrate the row lock. We commit setup data
directly and clean up via DELETE in ``finally`` (mirrors
test_partial_refund_concurrency.py).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.api.services.billing import BillingService
from src.api.services.payment import PaymentService
from src.core.config import Settings
from src.core.uid import new_id
from src.db.models.billing import Payment, TokenAccount, TokenTransaction
from src.db.models.user import User

pytestmark = pytest.mark.asyncio


def _make_settings() -> Settings:
    return Settings(
        jwt_secret_key="a_valid_test_secret_key_that_is_long_enough_256bits",
        stripe_secret_key_vex="sk_test_vex",  # noqa: S106
        stripe_webhook_secret_vex="whsec_test_vex",  # noqa: S106
        nowpayments_ipn_secret_vex="ipn_secret_vex",  # noqa: S106
    )


async def _seed_pending_payment(
    engine: AsyncEngine, *, external_id: str, tokens_granted: int
) -> tuple[User, TokenAccount, Payment]:
    """Seed a user, personal account, and a single pending Stripe payment. Commits."""
    user = User(
        id=new_id(),
        email=f"webhook-race-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id="vex",
        is_active=True,
    )
    account = TokenAccount(
        id=new_id(),
        user_id=user.id,
        account_type="personal",
        product_id="vex",
    )
    payment = Payment(
        id=new_id(),
        account_id=account.id,
        payment_provider="stripe",
        external_id=external_id,
        status="pending",
        amount_usd=Decimal("9.99"),
        tokens_granted=tokens_granted,
        currency="USD",
        product_id="vex",
        created_by=user.id,
    )
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add_all([user, account, payment])
        await session.commit()
        await session.refresh(user)
        await session.refresh(account)
        await session.refresh(payment)
    return user, account, payment


async def _cleanup(engine: AsyncEngine, account_id, user_id) -> None:  # type: ignore[no-untyped-def]
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(text("SET LOCAL session_replication_role = 'replica'"))
        await session.execute(
            delete(TokenTransaction).where(TokenTransaction.account_id == account_id)
        )
        await session.execute(delete(Payment).where(Payment.account_id == account_id))
        await session.execute(delete(TokenAccount).where(TokenAccount.id == account_id))
        await session.execute(delete(User).where(User.id == user_id))
        await session.commit()


async def test_concurrent_stripe_webhook_deliveries_credit_exactly_once(
    db_engine: AsyncEngine,
) -> None:
    """Two simultaneous deliveries of the same checkout.session.completed
    event must produce exactly one CREDIT transaction, not two."""
    external_id = f"cs_test_{uuid4().hex[:16]}"
    user, account, _payment = await _seed_pending_payment(
        db_engine, external_id=external_id, tokens_granted=1000
    )

    fake_event = MagicMock()
    fake_event.type = "checkout.session.completed"
    fake_event.id = f"evt_{uuid4().hex[:16]}"
    fake_event.data.object = {"id": external_id}

    settings = _make_settings()

    try:

        async def deliver() -> str:
            async with (
                AsyncSession(bind=db_engine, expire_on_commit=False) as session,
                session.begin(),
            ):
                service = PaymentService(
                    billing_service=BillingService(event_bus=None), settings=settings
                )
                with patch("stripe.Webhook.construct_event", return_value=fake_event):
                    await service.handle_stripe_webhook(
                        b"{}", "sig_ignored", session=session, product_id="vex"
                    )
                await session.commit()
                return "delivered"

        # Two truly independent connections racing on the same payment row.
        results = list(await asyncio.gather(deliver(), deliver()))
        assert results == ["delivered", "delivered"]

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            credit_count = (
                await session.execute(
                    text(
                        "SELECT COUNT(*) FROM token_transactions "
                        "WHERE account_id = :aid AND transaction_type = 'credit'"
                    ),
                    {"aid": account.id},
                )
            ).scalar_one()
            assert credit_count == 1, (
                f"Expected exactly 1 credit transaction, got {credit_count}. "
                "The row lock did NOT prevent a double credit."
            )

            balance = (
                await session.execute(
                    text(
                        "SELECT COALESCE(SUM(amount), 0) FROM token_transactions "
                        "WHERE account_id = :aid"
                    ),
                    {"aid": account.id},
                )
            ).scalar_one()
            assert balance == 1000, f"Expected balance 1000 (single credit), got {balance}"

            payment_status = (
                await session.execute(
                    text("SELECT status FROM payments WHERE external_id = :eid"),
                    {"eid": external_id},
                )
            ).scalar_one()
            assert payment_status == "completed"
    finally:
        await _cleanup(db_engine, account.id, user.id)


async def test_concurrent_nowpayments_ipn_deliveries_credit_exactly_once(
    db_engine: AsyncEngine,
) -> None:
    """Two simultaneous NowPayments IPN deliveries ('finished') for the same
    payment must produce exactly one CREDIT transaction."""
    invoice_id = f"np_invoice_{uuid4().hex[:12]}"
    user, account, payment = await _seed_pending_payment(
        db_engine, external_id=invoice_id, tokens_granted=500
    )
    payment.payment_provider = "nowpayments"

    settings = _make_settings()
    ipn_secret = settings.nowpayments_ipn_secret_for("vex")

    # order_id matches what create_nowpayments_invoice actually produces: a
    # JSON-encoded dict *string* embedded as a field value (NowPayments
    # echoes it back verbatim). price_amount is a raw JSON number with the
    # exact lexeme "10.00" — the classic case that breaks byte-equality if
    # the HMAC is verified against a re-serialized float (10.00 -> 10.0).
    order_id_json_string = json.dumps(
        {
            "account_id": str(account.id),
            "package_id": "starter",
            "payment_id": str(payment.id),
        }
    )
    raw_payload = (
        '{"payment_status":"finished","payment_id":"5077125060",'
        f"{json.dumps('order_id')}:{json.dumps(order_id_json_string)},"
        '"price_amount":10.00}'
    ).encode()

    parsed = json.loads(raw_payload, parse_float=str, parse_int=str)
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(ipn_secret.encode(), canonical, hashlib.sha512).hexdigest()

    try:

        async def deliver() -> str:
            async with (
                AsyncSession(bind=db_engine, expire_on_commit=False) as session,
                session.begin(),
            ):
                service = PaymentService(
                    billing_service=BillingService(event_bus=None), settings=settings
                )
                await service.handle_nowpayments_webhook(
                    raw_payload, signature, session=session, product_id="vex"
                )
                await session.commit()
                return "delivered"

        results = list(await asyncio.gather(deliver(), deliver()))
        assert results == ["delivered", "delivered"]

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            credit_count = (
                await session.execute(
                    text(
                        "SELECT COUNT(*) FROM token_transactions "
                        "WHERE account_id = :aid AND transaction_type = 'credit'"
                    ),
                    {"aid": account.id},
                )
            ).scalar_one()
            assert credit_count == 1, (
                f"Expected exactly 1 credit transaction, got {credit_count}. "
                "The row lock did NOT prevent a double credit."
            )

            balance = (
                await session.execute(
                    text(
                        "SELECT COALESCE(SUM(amount), 0) FROM token_transactions "
                        "WHERE account_id = :aid"
                    ),
                    {"aid": account.id},
                )
            ).scalar_one()
            assert balance == 500
    finally:
        await _cleanup(db_engine, account.id, user.id)
