"""Payment service for Stripe and NowPayments integration."""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
from datetime import UTC, datetime
from decimal import Decimal
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
from src.core.product import PaymentProvider
from src.core.uid import new_id
from src.db.repositories.billing import BillingRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BalanceEvent, BillingService
    from src.core.config import Settings
    from src.core.product import ProductConfig
    from src.db.models.billing import Payment

# Ratios within this band snap to "fully paid" on a `finished` IPN — crypto
# fee/fx rounding routinely lands a hair off 1.0 with no fraud implication.
_FULL_PAYMENT_TOLERANCE_LOW = Decimal("0.99")
_FULL_PAYMENT_TOLERANCE_HIGH = Decimal("1.01")
# Outside this band, ratios are still credited proportionally (never held
# for review) but logged at error level as an ops attention signal.
_EXTREME_RATIO_LOW = Decimal("0.5")
_EXTREME_RATIO_HIGH = Decimal("2.0")

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
            payment_provider=PaymentProvider.STRIPE.value,
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
    ) -> BalanceEvent | None:
        """Handle Stripe webhook event.

        1. Verify signature using the requesting product's webhook secret.
        2. Handle 'checkout.session.completed' only.
        3. Lock the payment row and re-check status under the lock (idempotent
           against concurrent/retried deliveries of the same event).
        4. Credit tokens.

        Returns the pending balance event for the credit, if any — the route
        commits first, then publishes it via ``EventBus.publish_balance``.
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
            return None

        checkout_session = event.data.object
        external_id: str = checkout_session["id"]

        repo = BillingRepository(session)
        # Row lock: a concurrent/retried delivery for the same event blocks
        # here until the first delivery commits, then observes COMPLETED and
        # returns — preventing a double credit.
        payment = await repo.get_payment_by_external_id_for_update(external_id)
        if payment is None:
            logger.warning("payment.webhook_payment_not_found", external_id=external_id)
            return None

        # Idempotent check — re-checked under the row lock above.
        if payment.status == PaymentStatus.COMPLETED.value:
            return None

        # Update payment
        payment.status = PaymentStatus.COMPLETED.value
        payment.completed_at = datetime.now(UTC)
        payment.provider_metadata = {
            **payment.provider_metadata,
            "webhook_event_id": event.id,
        }
        await session.flush()

        # Credit tokens
        credit_result = await self._billing.credit(
            payment.account_id,
            payment.tokens_granted,
            payment.id,
            description="Token purchase via Stripe",
            payment_provider=PaymentProvider.STRIPE.value,
            session=session,
            product_id=payment.product_id,
        )
        return credit_result.event

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
            payment_provider=PaymentProvider.NOWPAYMENTS.value,
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

    async def _apply_ipn_credit(
        self,
        payment: Payment,
        ratio: Decimal,
        *,
        session: AsyncSession,
        repo: BillingRepository,
    ) -> tuple[int, int, BalanceEvent | None]:
        """Telescoping delta credit — idempotent under IPN redelivery.

        ``target`` is the cumulative tokens that should be credited for this
        payment at ``ratio`` (floored, uncapped for overpayment). ``delta`` is
        what THIS call actually credits — the remainder after subtracting the
        ledger's already-credited sum, so a redelivered IPN credits zero and a
        later ``finished`` after a partial credit credits exactly the
        remainder, with no cumulative floor-rounding drift.

        Note:
            Must be called only while the payment row lock
            (``get_payment_for_update``) is held — reading ``already_credited``
            before the lock would reintroduce the double-credit race the lock
            exists to prevent.

        Args:
            payment: The locked payment row being credited.
            ratio: Effective paid/expected ratio to apply (already snapped to
                1.0 by the caller when within full-payment tolerance).
            session: DB session (passed through to ``billing_service.credit``).
            repo: Repository used to read the already-credited ledger sum.

        Returns:
            ``(target, delta, event)`` — ``target`` is the cumulative tokens
            that should be credited at ``ratio``; ``delta`` is what this call
            actually credited; ``event`` is the pending ``BalanceEvent`` for
            the credit, or ``None`` when ``delta <= 0``.
        """
        already_credited = await repo.get_credited_tokens_for_payment(payment.id)
        target = int(Decimal(payment.tokens_granted) * ratio)
        delta = target - already_credited
        if delta <= 0:
            return target, 0, None

        description = (
            "Token purchase via NowPayments (partial)"
            if ratio < _FULL_PAYMENT_TOLERANCE_LOW
            else "Token purchase via NowPayments"
        )
        credit_result = await self._billing.credit(
            payment.account_id,
            delta,
            payment.id,
            description=description,
            payment_provider=PaymentProvider.NOWPAYMENTS.value,
            session=session,
            product_id=payment.product_id,
        )
        return target, delta, credit_result.event

    async def handle_nowpayments_webhook(
        self,
        raw_payload: bytes,
        hmac_signature: str,
        *,
        session: AsyncSession,
        product_id: str,
    ) -> BalanceEvent | None:
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
        4. Cross-check the requesting product against the payment's own
           product_id (fail-loud on mismatch).
        5. Credit tokens proportionally — automatic partial credit with a
           warning log, never a hold-for-review state (policy decision).

        Returns the pending balance event for the credit, if any — the route
        commits first, then publishes it via ``EventBus.publish_balance``.
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
            return None

        # Idempotent check — re-checked under the row lock above.
        if payment.status == PaymentStatus.COMPLETED.value:
            return None

        # F2a: the payment must belong to the product that received this IPN —
        # fail-loud rather than silently crediting across products.
        if payment.product_id != product_id:
            raise PaymentVerificationError("NowPayments IPN product mismatch")

        # NowPayments' own payment id (distinct from our external_id, which
        # stores the invoice id) — kept for reconciliation, not used as a lookup key.
        np_payment_id = str(payload.get("payment_id", ""))

        if payment_status in ("finished", "partially_paid"):
            if payment.amount_usd <= 0:
                raise PaymentVerificationError(
                    "NowPayments payment has non-positive amount_usd; refusing to divide"
                )

            actually_paid_raw = payload.get("actually_paid") or payload.get("price_amount") or "0"
            actually_paid = Decimal(str(actually_paid_raw))
            expected_usd = Decimal(str(payment.amount_usd))
            ratio = actually_paid / expected_usd

            is_extreme = ratio < _EXTREME_RATIO_LOW or ratio > _EXTREME_RATIO_HIGH

            if payment_status == "finished":
                in_tolerance = _FULL_PAYMENT_TOLERANCE_LOW <= ratio <= _FULL_PAYMENT_TOLERANCE_HIGH
                effective_ratio = Decimal(1) if in_tolerance else ratio
                payment.status = PaymentStatus.COMPLETED.value
                payment.completed_at = datetime.now(UTC)
            else:
                in_tolerance = False
                effective_ratio = ratio
                payment.status = PaymentStatus.PARTIALLY_PAID.value

            target, delta_credited, event = await self._apply_ipn_credit(
                payment, effective_ratio, session=session, repo=repo
            )

            payment.provider_metadata = {
                **payment.provider_metadata,
                "ipn_payment_id": np_payment_id,
                "ipn_actually_paid": str(actually_paid),
                "expected_usd": str(expected_usd),
                "ratio": str(ratio),
                "tokens_credited_total": target,
                "ipn_payload": payload,
            }
            await session.flush()

            log_fields: dict[str, Any] = {
                "payment_id": str(payment.id),
                "expected_usd": str(expected_usd),
                "actually_paid": str(actually_paid),
                "ratio": str(ratio),
                "tokens_credited": delta_credited,
                "tokens_granted": payment.tokens_granted,
            }
            if payment_status == "finished":
                if in_tolerance:
                    logger.info("payment.completed", **log_fields)
                elif is_extreme:
                    event_name = (
                        "payment.overpaid_credited" if ratio > 1 else "payment.underpaid_credited"
                    )
                    logger.error(event_name, **log_fields)
                elif ratio > _FULL_PAYMENT_TOLERANCE_HIGH:
                    logger.warning("payment.overpaid_credited", **log_fields)
                else:
                    logger.warning("payment.underpaid_credited", **log_fields)
            else:
                log_fn = logger.error if is_extreme else logger.warning
                log_fn("payment.partially_paid_credited", **log_fields)

            return event

        # Intermediate/terminal-non-completed status — update payment but don't credit
        status_map = {
            "waiting": PaymentStatus.PENDING.value,
            "confirming": PaymentStatus.PENDING.value,
            "sending": PaymentStatus.PENDING.value,
            "failed": PaymentStatus.FAILED.value,
            "expired": PaymentStatus.FAILED.value,
            "refunded": PaymentStatus.REFUNDED.value,
        }
        if payment_status not in status_map:
            logger.warning(
                "payment.ipn_unknown_status",
                payment_id=str(payment.id),
                raw_status=payment_status,
            )
        new_status = status_map.get(payment_status, payment.status)
        payment.status = new_status
        payment.provider_metadata = {
            **payment.provider_metadata,
            "ipn_payment_id": np_payment_id,
            "ipn_payload": payload,
        }
        await session.flush()
        return None
