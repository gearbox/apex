"""Payment service for Stripe and NowPayments integration."""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import stripe
import structlog

from src.api.services.billing_errors import (
    AccountNotFoundError,
    PaymentVerificationError,
)
from src.core.config import TOKEN_PACKAGES
from src.core.enums import PaymentStatus
from src.core.uid import new_id
from src.db.repositories.billing import BillingRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BillingService
    from src.core.config import Settings
    from src.core.product import ProductConfig

logger = structlog.get_logger(__name__)


@dataclasses.dataclass
class StripeCheckoutResult:
    """Result of Stripe checkout session creation."""

    checkout_url: str
    session_id: str
    payment_id: UUID


@dataclasses.dataclass
class NowPaymentsInvoiceResult:
    """Result of NowPayments invoice creation."""

    invoice_url: str
    payment_id: UUID


class PaymentService:
    """Service for payment processing via Stripe and NowPayments."""

    def __init__(
        self,
        billing_service: BillingService,
        settings: Settings,
    ) -> None:
        self._billing = billing_service
        self._settings = settings

    async def create_stripe_checkout(
        self,
        account_id: UUID,
        package_id: str,
        user_id: UUID,
        *,
        session: AsyncSession,
        product_config: ProductConfig,
    ) -> StripeCheckoutResult:
        """Create a Stripe Checkout Session.

        1. Validate package_id exists in TOKEN_PACKAGES config.
        2. Create Stripe Checkout Session with metadata.
        3. Store Payment(status='pending') in DB.
        4. Return checkout_url and session_id.
        """
        package = TOKEN_PACKAGES.get(package_id)
        if package is None:
            raise ValueError(f"Invalid package_id: {package_id}")

        product_id = product_config.slug

        repo = BillingRepository(session)
        account = await repo.get_account(account_id)
        if account is None:
            raise AccountNotFoundError(f"Account {account_id} not found")

        client = stripe.StripeClient(api_key=self._settings.stripe_secret_key_for(product_id))
        success_url = (
            f"{product_config.frontend_origin}{self._settings.stripe_checkout_success_path}"
        )
        cancel_url = f"{product_config.frontend_origin}{self._settings.stripe_checkout_cancel_path}"

        checkout_session = await client.v1.checkout.sessions.create_async(
            params={
                "mode": "payment",
                "line_items": [
                    {
                        "price_data": {
                            "currency": "usd",
                            "product_data": {
                                "name": f"{package.name} Token Package",
                                "description": f"{package.total_tokens} tokens",
                            },
                            "unit_amount": int(package.price_usd * 100),
                        },
                        "quantity": 1,
                    }
                ],
                "metadata": {
                    "account_id": str(account_id),
                    "package_id": package_id,
                },
                "success_url": success_url,
                "cancel_url": cancel_url,
            }
        )

        payment_id = new_id()
        await repo.create_payment(
            id=payment_id,
            account_id=account_id,
            payment_provider="stripe",
            external_id=checkout_session.id,
            status=PaymentStatus.PENDING.value,
            amount_usd=package.price_usd,
            tokens_granted=package.total_tokens,
            currency="USD",
            provider_metadata={"checkout_session_id": checkout_session.id},
            created_by=user_id,
            product_id=product_id,
        )

        return StripeCheckoutResult(
            checkout_url=checkout_session.url or "",
            session_id=checkout_session.id,
            payment_id=payment_id,
        )

    async def handle_stripe_webhook(
        self,
        payload: bytes,
        stripe_signature: str,
        *,
        session: AsyncSession,
        product_id: str,
    ) -> None:
        """Handle Stripe webhook event.

        1. Verify signature using the requesting product's webhook secret.
        2. Handle 'checkout.session.completed' only.
        3. Lock the payment row and re-check status under the lock (idempotent
           against concurrent/retried deliveries of the same event).
        4. Credit tokens.
        """
        webhook_secret = self._settings.stripe_webhook_secret_for(product_id)

        try:
            event = stripe.Webhook.construct_event(  # type: ignore[no-untyped-call]
                payload,
                stripe_signature,
                webhook_secret,
            )
        except (stripe.SignatureVerificationError, ValueError) as e:
            raise PaymentVerificationError(f"Stripe signature invalid: {e}") from e

        if event.type != "checkout.session.completed":
            return

        checkout_session = event.data.object
        external_id: str = checkout_session["id"]

        repo = BillingRepository(session)
        # Row lock: a concurrent/retried delivery for the same event blocks
        # here until the first delivery commits, then observes COMPLETED and
        # returns — preventing a double credit.
        payment = await repo.get_payment_by_external_id_for_update(external_id)
        if payment is None:
            logger.warning("payment.webhook_payment_not_found", external_id=external_id)
            return

        # Idempotent check — re-checked under the row lock above.
        if payment.status == PaymentStatus.COMPLETED.value:
            return

        # Update payment
        payment.status = PaymentStatus.COMPLETED.value
        payment.completed_at = datetime.now(UTC)
        payment.provider_metadata = {
            **payment.provider_metadata,
            "webhook_event_id": event.id,
        }
        await session.flush()

        # Credit tokens
        await self._billing.credit(
            payment.account_id,
            payment.tokens_granted,
            payment.id,
            description="Token purchase via Stripe",
            payment_provider="stripe",
            session=session,
            product_id=payment.product_id,
        )

    async def create_nowpayments_invoice(
        self,
        account_id: UUID,
        package_id: str,
        pay_currency: str,
        user_id: UUID,
        *,
        session: AsyncSession,
        product_id: str,
    ) -> NowPaymentsInvoiceResult:
        """Create a NowPayments invoice.

        1. Validate package_id and pay_currency.
        2. POST to NowPayments API.
        3. Store Payment(status='pending') in DB.
        4. Return invoice_url and payment_id.
        """
        import httpx

        package = TOKEN_PACKAGES.get(package_id)
        if package is None:
            raise ValueError(f"Invalid package_id: {package_id}")

        repo = BillingRepository(session)
        account = await repo.get_account(account_id)
        if account is None:
            raise AccountNotFoundError(f"Account {account_id} not found")

        payment_id = new_id()
        order_id = json.dumps(
            {
                "account_id": str(account_id),
                "package_id": package_id,
                "payment_id": str(payment_id),
            }
        )

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.nowpayments.io/v1/invoice",
                headers={
                    "x-api-key": self._settings.nowpayments_api_key_for(product_id),
                    "Content-Type": "application/json",
                },
                json={
                    "price_amount": float(package.price_usd),
                    "price_currency": "usd",
                    "pay_currency": pay_currency,
                    "order_id": order_id,
                    "order_description": f"{package.name} Token Package - {package.total_tokens} tokens",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        np_id = str(data.get("id", ""))
        invoice_url = data.get("invoice_url", "")

        await repo.create_payment(
            id=payment_id,
            account_id=account_id,
            payment_provider="nowpayments",
            external_id=np_id,
            status=PaymentStatus.PENDING.value,
            amount_usd=package.price_usd,
            tokens_granted=package.total_tokens,
            currency=pay_currency.upper(),
            provider_metadata=data,
            created_by=user_id,
            product_id=product_id,
        )

        return NowPaymentsInvoiceResult(
            invoice_url=invoice_url,
            payment_id=payment_id,
        )

    async def handle_nowpayments_webhook(
        self,
        raw_payload: bytes,
        hmac_signature: str,
        *,
        session: AsyncSession,
        product_id: str,
    ) -> None:
        """Handle NowPayments IPN webhook.

        1. Verify HMAC-SHA512 over the raw body, using NowPayments' own
           canonicalization (sort_keys, numeric lexemes preserved as strings
           so e.g. 10.00 doesn't become 10.0 and break byte-equality).
        2. Resolve our internal payment via the ``payment_id`` we embedded in
           ``order_id`` at invoice-creation time — NOT the IPN's top-level
           ``payment_id``, which is NowPayments' own payment id and never
           matches what we stored as ``external_id`` (the invoice id).
        3. Lock the payment row and re-check status under the lock (idempotent
           against concurrent/retried IPNs for the same payment).
        4. Credit tokens on 'finished'.
        """
        ipn_secret = self._settings.nowpayments_ipn_secret_for(product_id)

        try:
            parsed = json.loads(raw_payload, parse_float=str, parse_int=str)
        except json.JSONDecodeError as e:
            raise PaymentVerificationError("NowPayments IPN payload is not valid JSON") from e

        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
        expected = hmac.new(
            ipn_secret.encode(),
            canonical,
            hashlib.sha512,
        ).hexdigest()

        if not hmac.compare_digest(expected, hmac_signature):
            raise PaymentVerificationError("NowPayments HMAC signature invalid")

        payload: dict[str, Any] = parsed
        payment_status = payload.get("payment_status", "")

        try:
            order = json.loads(str(payload.get("order_id", "")))
            internal_payment_id = UUID(str(order["payment_id"]))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            raise PaymentVerificationError("NowPayments IPN order_id is malformed") from e

        repo = BillingRepository(session)
        # Row lock: a concurrent/retried IPN for the same payment blocks here
        # until the first delivery commits, then observes COMPLETED and
        # returns — preventing a double credit.
        payment = await repo.get_payment_for_update(internal_payment_id)
        if payment is None:
            logger.warning(
                "payment.webhook_payment_not_found",
                internal_payment_id=str(internal_payment_id),
            )
            return

        # Idempotent check — re-checked under the row lock above.
        if payment.status == PaymentStatus.COMPLETED.value:
            return

        # NowPayments' own payment id (distinct from our external_id, which
        # stores the invoice id) — kept for reconciliation, not used as a lookup key.
        np_payment_id = str(payload.get("payment_id", ""))

        if payment_status == "finished":
            payment.status = PaymentStatus.COMPLETED.value
            payment.completed_at = datetime.now(UTC)
            payment.provider_metadata = {
                **payment.provider_metadata,
                "ipn_payment_id": np_payment_id,
                "ipn_payload": payload,
            }
            await session.flush()

            await self._billing.credit(
                payment.account_id,
                payment.tokens_granted,
                payment.id,
                description="Token purchase via NowPayments",
                payment_provider="nowpayments",
                session=session,
                product_id=payment.product_id,
            )
        else:
            # Intermediate status — update payment but don't credit
            status_map = {
                "waiting": PaymentStatus.PENDING.value,
                "confirming": PaymentStatus.PENDING.value,
                "sending": PaymentStatus.PENDING.value,
                "failed": PaymentStatus.FAILED.value,
                "expired": PaymentStatus.FAILED.value,
                "refunded": PaymentStatus.REFUNDED.value,
            }
            new_status = status_map.get(payment_status, payment.status)
            payment.status = new_status
            payment.provider_metadata = {
                **payment.provider_metadata,
                "ipn_payment_id": np_payment_id,
            }
            await session.flush()
