"""Provider-neutral payment orchestration and the single settlement write path."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from src.api.services.billing_errors import (
    AccountNotFoundError,
    PayCurrencySuppressedError,
    PaymentProviderDisabledError,
    PaymentVerificationError,
    TopUpAmountError,
)
from src.api.services.payments.contracts import (
    ChargeContext,
    CreatedCharge,
    WebhookEnvelope,
)
from src.core.enums import PaymentStatus
from src.core.topup_pricing import TopUpQuote, build_quote, topup_tiers_for
from src.core.uid import new_id
from src.db.models.billing import PAYMENT_CURRENCY_MAX_LEN
from src.db.repositories.billing import BillingRepository
from src.db.repositories.payment_currency import PaymentCurrencyRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BalanceEvent, BillingService
    from src.api.services.payment_provider_state import PaymentProviderStateService
    from src.api.services.payments.registry import GatewayRegistry
    from src.core.config import Settings
    from src.core.product import PaymentProvider, ProductConfig
    from src.db.models.billing import Payment

_FULL_PAYMENT_TOLERANCE_LOW = Decimal("0.99")
_FULL_PAYMENT_TOLERANCE_HIGH = Decimal("1.01")
_EXTREME_RATIO_LOW = Decimal("0.5")
_EXTREME_RATIO_HIGH = Decimal("2.0")

logger = structlog.get_logger(__name__)


class PaymentService:
    """Coordinate validation, gateway calls, persistence, and settlement."""

    def __init__(
        self,
        billing_service: BillingService,
        settings: Settings,
        registry: GatewayRegistry,
        provider_state_service: PaymentProviderStateService,
    ) -> None:
        self._billing = billing_service
        self._settings = settings
        self._registry = registry
        self._provider_state = provider_state_service

    def _quote_for_amount(
        self, amount_usd: int, *, product_id: str, provider: PaymentProvider
    ) -> TopUpQuote:
        min_usd = self._settings.billing_min_topup_usd
        max_usd = self._settings.billing_max_topup_usd
        if not min_usd <= amount_usd <= max_usd:
            raise TopUpAmountError(
                f"amount_usd must be between {min_usd} and {max_usd} (got {amount_usd})"
            )
        quote = build_quote(
            amount_usd,
            tiers=topup_tiers_for(product_id, self._settings),
            tokens_per_usd=self._settings.billing_tokens_per_usd,
        )
        logger.info(
            "payment.topup_quote",
            credits_usd=quote.credits_usd,
            discount_pct=quote.discount_pct,
            total_due=str(quote.total_due),
            tokens_granted=quote.tokens_granted,
            provider=provider.value,
            product_id=product_id,
        )
        return quote

    async def create_charge(
        self,
        provider: PaymentProvider,
        account_id: UUID,
        amount_usd: int,
        user_id: UUID,
        *,
        session: AsyncSession,
        product_config: ProductConfig,
        extra: dict[str, str] | None = None,
    ) -> CreatedCharge:
        """Create and persist a provider charge after capability/state checks."""

        quote = self._quote_for_amount(
            amount_usd, product_id=product_config.slug, provider=provider
        )
        repo = BillingRepository(session)
        if await repo.get_account(account_id) is None:
            raise AccountNotFoundError(f"Account {account_id} not found")
        if not await self._provider_state.is_effective(product_config, provider, session=session):
            raise PaymentProviderDisabledError(provider)

        pinned_ticker = (extra or {}).get("pay_currency", "").strip().upper()
        if pinned_ticker and await PaymentCurrencyRepository(session).is_ticker_suppressed(
            product_config.slug, provider.value, pinned_ticker
        ):
            raise PayCurrencySuppressedError(pinned_ticker)

        payment_id = new_id()
        result = await self._registry.get(provider).create_charge(
            ChargeContext(
                payment_id=payment_id,
                account_id=account_id,
                user_id=user_id,
                product_config=product_config,
                quote=quote,
                extra=extra or {},
            )
        )
        await repo.create_payment(
            id=payment_id,
            account_id=account_id,
            payment_provider=provider.value,
            external_id=result.external_id,
            status=PaymentStatus.PENDING.value,
            amount_usd=quote.total_due,
            tokens_granted=quote.tokens_granted,
            currency=result.currency,
            provider_metadata=result.provider_metadata,
            created_by=user_id,
            product_id=product_config.slug,
        )
        return CreatedCharge(
            redirect_url=result.redirect_url,
            external_id=result.external_id,
            payment_id=payment_id,
        )

    @staticmethod
    async def _credit_delta(
        payment: Payment,
        ratio: Decimal,
        *,
        repo: BillingRepository,
    ) -> tuple[int, int]:
        already_credited = await repo.get_credited_tokens_for_payment(payment.id)
        target = int(Decimal(payment.tokens_granted) * ratio)
        return target, max(target - already_credited, 0)

    async def handle_webhook(
        self,
        provider: PaymentProvider,
        envelope: WebhookEnvelope,
        *,
        session: AsyncSession,
    ) -> BalanceEvent | None:
        """Verify and settle a webhook without consulting runtime enablement.

        Provider disablement only stops new charges; in-flight payments must
        always be allowed to settle.
        """

        outcome = await self._registry.get(provider).verify_webhook(envelope)
        if outcome.status is None:
            return None

        repo = BillingRepository(session)
        if outcome.lookup.by == "external_id":
            payment = await repo.get_payment_by_external_id_for_update(outcome.lookup.value)
        else:
            try:
                payment_id = UUID(outcome.lookup.value)
            except ValueError as exc:
                raise PaymentVerificationError("Webhook payment_id is malformed") from exc
            payment = await repo.get_payment_for_update(payment_id)

        if payment is None:
            logger.warning(
                "payment.webhook_payment_not_found",
                lookup_by=outcome.lookup.by,
                lookup_value=outcome.lookup.value,
                provider=provider.value,
            )
            return None
        if payment.status == PaymentStatus.COMPLETED.value:
            return None
        if payment.product_id != envelope.product_id:
            raise PaymentVerificationError("Payment webhook product mismatch")

        # Do not let a late/out-of-order intermediate IPN (e.g. waiting/confirming)
        # regress a payment that has already been marked partially paid.
        if (
            payment.status != PaymentStatus.PARTIALLY_PAID.value
            or outcome.status is not PaymentStatus.PENDING
        ):
            payment.status = outcome.status.value
        if outcome.status is PaymentStatus.COMPLETED:
            payment.completed_at = datetime.now(UTC)

        if outcome.settled_currency and payment.currency != outcome.settled_currency:
            if len(outcome.settled_currency) > PAYMENT_CURRENCY_MAX_LEN:
                logger.warning(
                    "payment.settled_currency_overlong",
                    payment_id=str(payment.id),
                    settled_currency=outcome.settled_currency,
                )
            else:
                payment.currency = outcome.settled_currency

        event: BalanceEvent | None = None
        credit_ratio: Decimal | None = None
        credit_delta = 0
        metadata_patch: dict[str, Any] = dict(outcome.metadata_patch)
        if outcome.amount_paid is not None:
            if outcome.amount_due is None or outcome.amount_due <= 0:
                raise PaymentVerificationError(
                    "Webhook amount_paid provided without a valid amount_due"
                )
            expected_usd = Decimal(str(payment.amount_usd))
            ratio = outcome.amount_paid / outcome.amount_due
            in_tolerance = (
                outcome.status is PaymentStatus.COMPLETED
                and _FULL_PAYMENT_TOLERANCE_LOW <= ratio <= _FULL_PAYMENT_TOLERANCE_HIGH
            )
            effective_ratio = Decimal(1) if in_tolerance else ratio
            target, credit_delta = await self._credit_delta(payment, effective_ratio, repo=repo)
            credit_ratio = effective_ratio
            metadata_patch |= {
                "expected_usd": str(expected_usd),
                "amount_due": str(outcome.amount_due),
                "ratio": str(ratio),
                "tokens_credited_total": target,
            }
            self._log_proportional_settlement(
                payment=payment,
                status=outcome.status,
                amount_paid=outcome.amount_paid,
                amount_due=outcome.amount_due,
                ratio=ratio,
                delta=credit_delta,
                in_tolerance=in_tolerance,
            )
        elif outcome.status is PaymentStatus.COMPLETED:
            credit_ratio = Decimal(1)
            _, credit_delta = await self._credit_delta(payment, credit_ratio, repo=repo)

        payment.provider_metadata = {**payment.provider_metadata, **metadata_patch}
        await session.flush()

        # This is the only provider-payment credit write path. The payment row
        # remains locked through the status check, flush, and ledger insert.
        if credit_delta > 0 and credit_ratio is not None:
            description = f"Token purchase via {provider.value}"
            if credit_ratio < _FULL_PAYMENT_TOLERANCE_LOW:
                description += " (partial)"
            credit_result = await self._billing.credit(
                payment.account_id,
                credit_delta,
                payment.id,
                description=description,
                payment_provider=provider.value,
                session=session,
                product_id=payment.product_id,
            )
            event = credit_result.event
        return event

    @staticmethod
    def _log_proportional_settlement(
        *,
        payment: Payment,
        status: PaymentStatus,
        amount_paid: Decimal,
        amount_due: Decimal,
        ratio: Decimal,
        delta: int,
        in_tolerance: bool,
    ) -> None:
        fields = {
            "payment_id": str(payment.id),
            "amount_due": str(amount_due),
            "actually_paid": str(amount_paid),
            "ratio": str(ratio),
            "tokens_credited": delta,
            "tokens_granted": payment.tokens_granted,
        }
        is_extreme = ratio < _EXTREME_RATIO_LOW or ratio > _EXTREME_RATIO_HIGH
        if status is PaymentStatus.PARTIALLY_PAID:
            (logger.error if is_extreme else logger.warning)(
                "payment.partially_paid_credited", **fields
            )
        elif in_tolerance:
            logger.info("payment.completed", **fields)
        elif is_extreme:
            logger.error(
                "payment.overpaid_credited" if ratio > 1 else "payment.underpaid_credited",
                **fields,
            )
        else:
            logger.warning(
                "payment.overpaid_credited" if ratio > 1 else "payment.underpaid_credited",
                **fields,
            )
