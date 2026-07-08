"""Billing API routes — token balance, transactions, pricing, payments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

import msgspec
import structlog
from litestar import Controller, Request, Response, get, post
from litestar.di import Provide
from litestar.exceptions import HTTPException, PermissionDeniedException
from litestar.params import Parameter
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED, HTTP_403_FORBIDDEN
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.billing import (
    BalanceResponse,
    BillingAccountResponse,
    NowPaymentsInvoiceResponse,
    PricingRuleResponse,
    SetBillingAccountRequest,
    StripeCheckoutResponse,
    TopUpNowPaymentsRequest,
    TopUpOptionsResponse,
    TopUpStripeRequest,
    TopUpTierResponse,
    TransactionResponse,
)
from src.api.schemas.pagination import CursorPage, decode_cursor, encode_cursor
from src.api.security import auth_guard
from src.api.services.billing import BillingService
from src.api.services.billing_errors import OrganizationPermissionError
from src.api.services.event_bus import EventBus
from src.api.services.idempotency import IdempotencyReplayResult, IdempotencyService
from src.api.services.payment import PaymentService
from src.api.services.pricing import PricingService
from src.core.config import Settings
from src.core.product import PaymentProvider, ProductConfig
from src.core.topup_pricing import topup_tiers_for

if TYPE_CHECKING:
    from src.db.models.billing import TokenTransaction

logger = structlog.get_logger(__name__)


def _txn_to_response(txn: TokenTransaction) -> TransactionResponse:
    """Convert a TokenTransaction model to a response struct."""
    return TransactionResponse(
        id=txn.id,
        transaction_type=txn.transaction_type,
        amount=txn.amount,
        balance_after=txn.balance_after,
        description=txn.description,
        metadata=txn.metadata_,
        job_id=txn.job_id,
        payment_id=txn.payment_id,
        created_at=txn.created_at,
        created_by=txn.created_by,
    )


class BillingController(Controller):
    """Billing endpoints — balance, transactions, pricing, top-up."""

    path = "/v1/billing"
    tags: Sequence[str] | None = ["Billing"]
    guards = [auth_guard]  # noqa: RUF012
    dependencies = {  # noqa: RUF012
        "current_user_id": Provide(get_current_user_id),
    }

    @get("/balance")
    async def get_balance(
        self,
        current_user_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
    ) -> BalanceResponse:
        """Get current token balance for the authenticated user."""
        account = await billing_service.resolve_account_for_user(current_user_id, session=session)
        balance = await billing_service.get_balance(account.id, session=session)

        org_name: str | None = None
        if account.account_type == "enterprise" and account.organization is not None:
            org_name = account.organization.name

        return BalanceResponse(
            account_id=account.id,
            account_type=account.account_type,
            balance=balance,
            organization_name=org_name,
        )

    @get("/transactions")
    async def get_transactions(
        self,
        current_user_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
        limit: int = 50,
        type: str | None = None,
        cursor: str | None = None,
    ) -> CursorPage[TransactionResponse]:
        """Get transaction history for the authenticated user's account.

        Query parameters:
          - ``limit``: Page size (default 50)
          - ``type``: Filter by transaction type
          - ``cursor``: Opaque cursor from a previous response's ``next_cursor``
            field.  Pass to fetch the next page.
        """
        cursor_ts = None
        cursor_id = None
        if cursor is not None:
            cursor_ts, cursor_id = decode_cursor(cursor)

        account = await billing_service.resolve_account_for_user(current_user_id, session=session)
        transactions = await billing_service.get_transaction_history(
            account.id,
            limit=limit,
            transaction_type=type,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
            session=session,
        )

        has_more = len(transactions) > limit
        if has_more:
            transactions = list(transactions)[:limit]

        items = [_txn_to_response(t) for t in transactions]

        next_cursor: str | None = None
        if has_more and transactions:
            last = transactions[-1]
            next_cursor = encode_cursor(last.created_at, last.id)

        return CursorPage(
            items=items,
            limit=limit,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    @get("/pricing")
    async def get_pricing(
        self,
        session: AsyncSession,
        pricing_service: PricingService,
    ) -> list[PricingRuleResponse]:
        """Get active pricing rules."""
        rules = await pricing_service.list_catalog(active_only=True, session=session)
        return [
            PricingRuleResponse(
                id=r.id,
                provider=r.provider,
                generation_type=r.generation_type,
                model=r.model,
                token_cost=r.token_cost,
                is_active=r.is_active,
                effective_from=r.effective_from,
                effective_until=r.effective_until,
                notes=r.notes,
            )
            for r in rules
        ]

    @get("/topup/options")
    async def get_topup_options(
        self,
        settings: Settings,
        product_id: str,
    ) -> TopUpOptionsResponse:
        """Get top-up pricing configuration: bounds, rate, and discount tiers.

        Single source of truth with the actual charge — both this endpoint and
        the top-up creation paths call ``topup_tiers_for``/``build_quote``.
        """
        tiers = topup_tiers_for(product_id, settings)
        return TopUpOptionsResponse(
            min_amount_usd=settings.billing_min_topup_usd,
            max_amount_usd=settings.billing_max_topup_usd,
            tokens_per_usd=settings.billing_tokens_per_usd,
            tiers=[
                TopUpTierResponse(threshold_usd=t.threshold_usd, discount_pct=t.discount_pct)
                for t in tiers
            ],
        )

    @get("/account")
    async def get_billing_account(
        self,
        current_user_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
    ) -> BillingAccountResponse:
        """Get the current billing account preference."""
        preference = await billing_service.get_billing_account_preference(
            current_user_id, session=session
        )
        return BillingAccountResponse(
            preferred_account=preference,
            message="Current billing account preference",
        )

    @post("/account")
    async def set_billing_account(
        self,
        current_user_id: UUID,
        data: SetBillingAccountRequest,
        session: AsyncSession,
        billing_service: BillingService,
    ) -> BillingAccountResponse:
        """Set the preferred billing account (personal or enterprise).

        Persists the choice permanently until changed.
        Raises HTTP 400 if account type is invalid.
        Raises HTTP 403 if enterprise selected but user has no org membership.
        """
        try:
            await billing_service.set_billing_account_preference(
                current_user_id, data.account, session=session
            )
        except OrganizationPermissionError as exc:
            raise PermissionDeniedException(detail=str(exc)) from exc

        await session.commit()
        return BillingAccountResponse(
            preferred_account=data.account.value,
            message=f"Billing account preference set to '{data.account.value}'",
        )

    @post("/topup/stripe")
    async def topup_stripe(
        self,
        current_user_id: UUID,
        data: TopUpStripeRequest,
        session: AsyncSession,
        billing_service: BillingService,
        payment_service: PaymentService,
        product_id: str,
        product_config: ProductConfig,
        idempotency_service: IdempotencyService,
        idempotency_key_header: Annotated[
            str,
            Parameter(
                header="Idempotency-Key",
                max_length=64,
                description="Unique key for request deduplication (max 64 chars). Repeated requests with the same key return the cached response.",
            ),
        ],
    ) -> Response[StripeCheckoutResponse]:
        """Create a Stripe checkout session for token purchase (idempotent via Idempotency-Key)."""
        if not product_config.supports_payment_provider(PaymentProvider.STRIPE):
            logger.warning(
                "payment.provider_not_supported",
                provider="stripe",
                product=product_id,
            )
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Payment provider not available for this product",
            )

        request_hash = IdempotencyService.hash_request(msgspec.json.encode(data))

        check_result = await idempotency_service.check(
            user_id=current_user_id,
            product_id=product_id,
            idempotency_key=idempotency_key_header,
            operation="payment",
            request_hash=request_hash,
            session=session,
        )
        if isinstance(check_result, IdempotencyReplayResult):
            return Response(
                content=check_result.body,  # type: ignore[arg-type]
                status_code=check_result.status_code,
            )

        record_id = check_result
        try:
            account = await billing_service.resolve_account_for_user(
                current_user_id, session=session
            )
            result = await payment_service.create_stripe_checkout(
                account.id,
                data.amount_usd,
                current_user_id,
                session=session,
                product_config=product_config,
            )
            response = StripeCheckoutResponse(
                checkout_url=result.checkout_url,
                session_id=result.session_id,
                payment_id=result.payment_id,
            )
            response_body = msgspec.to_builtins(response)
            await idempotency_service.complete(
                record_id,
                resource_id=result.payment_id,
                response_status_code=HTTP_201_CREATED,
                response_body=response_body,
                session=session,
            )
            await session.commit()
            return Response(content=response, status_code=HTTP_201_CREATED)

        except Exception:
            await idempotency_service.fail(record_id, session=session)
            raise

    @post("/topup/nowpayments")
    async def topup_nowpayments(
        self,
        current_user_id: UUID,
        data: TopUpNowPaymentsRequest,
        session: AsyncSession,
        billing_service: BillingService,
        payment_service: PaymentService,
        product_id: str,
        product_config: ProductConfig,
        idempotency_service: IdempotencyService,
        idempotency_key_header: Annotated[
            str,
            Parameter(
                header="Idempotency-Key",
                max_length=64,
                description="Unique key for request deduplication (max 64 chars). Repeated requests with the same key return the cached response.",
            ),
        ],
    ) -> Response[NowPaymentsInvoiceResponse]:
        """Create a NowPayments invoice for token purchase (idempotent via Idempotency-Key)."""
        if not product_config.supports_payment_provider(PaymentProvider.NOWPAYMENTS):
            logger.warning(
                "payment.provider_not_supported",
                provider="nowpayments",
                product=product_id,
            )
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Payment provider not available for this product",
            )

        request_hash = IdempotencyService.hash_request(msgspec.json.encode(data))

        check_result = await idempotency_service.check(
            user_id=current_user_id,
            product_id=product_id,
            idempotency_key=idempotency_key_header,
            operation="payment",
            request_hash=request_hash,
            session=session,
        )
        if isinstance(check_result, IdempotencyReplayResult):
            return Response(
                content=check_result.body,  # type: ignore[arg-type]
                status_code=check_result.status_code,
            )

        record_id = check_result
        try:
            account = await billing_service.resolve_account_for_user(
                current_user_id, session=session
            )
            result = await payment_service.create_nowpayments_invoice(
                account.id,
                data.amount_usd,
                data.pay_currency,
                current_user_id,
                session=session,
                product_id=product_id,
            )
            response = NowPaymentsInvoiceResponse(
                invoice_url=result.invoice_url,
                payment_id=result.payment_id,
            )
            response_body = msgspec.to_builtins(response)
            await idempotency_service.complete(
                record_id,
                resource_id=result.payment_id,
                response_status_code=HTTP_201_CREATED,
                response_body=response_body,
                session=session,
            )
            await session.commit()
            return Response(content=response, status_code=HTTP_201_CREATED)

        except Exception:
            await idempotency_service.fail(record_id, session=session)
            raise


class BillingWebhookController(Controller):
    """Webhook endpoints — no auth guard, signature verified in handler."""

    path = "/v1/billing/webhooks"
    tags: Sequence[str] | None = ["Billing Webhooks"]

    @post("/stripe")
    async def stripe_webhook(
        self,
        request: Request[Any, Any, Any],
        session: AsyncSession,
        payment_service: PaymentService,
        event_bus: EventBus,
        product_id: str,
    ) -> Response[dict[str, Any]]:
        """Handle Stripe webhook.

        ``product_id`` is resolved by ProductMiddleware from the request Host —
        Stripe must be configured to post to the product's own domain so the
        correct per-product webhook secret is used for verification.
        """
        payload = await request.body()
        signature = request.headers.get("stripe-signature", "")
        balance_event = await payment_service.handle_stripe_webhook(
            payload, signature, session=session, product_id=product_id
        )
        await session.commit()
        await event_bus.publish_balance(balance_event)
        return Response(content={"received": True}, status_code=HTTP_200_OK)

    @post("/nowpayments")
    async def nowpayments_webhook(
        self,
        request: Request[Any, Any, Any],
        session: AsyncSession,
        payment_service: PaymentService,
        event_bus: EventBus,
        product_id: str,
    ) -> Response[dict[str, Any]]:
        """Handle NowPayments IPN webhook.

        Reads the raw body (not the parsed JSON) — HMAC verification must run
        over the exact bytes NowPayments signed, which a parse/re-serialize
        round trip through ``request.json()`` is not guaranteed to preserve.
        """
        raw = await request.body()
        hmac_sig = request.headers.get("x-nowpayments-sig", "")
        balance_event = await payment_service.handle_nowpayments_webhook(
            raw, hmac_sig, session=session, product_id=product_id
        )
        await session.commit()
        await event_bus.publish_balance(balance_event)
        return Response(content={"received": True}, status_code=HTTP_200_OK)
