"""Billing API routes — token balance, transactions, pricing, payments."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

import structlog
from litestar import Controller, Request, Response, get, post
from litestar.di import Provide
from litestar.exceptions import PermissionDeniedException
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.billing import (
    BalanceResponse,
    BillingAccountResponse,
    NowPaymentsInvoiceResponse,
    PricingRuleResponse,
    SetBillingAccountRequest,
    StripeCheckoutResponse,
    TokenPackageResponse,
    TopUpNowPaymentsRequest,
    TopUpStripeRequest,
    TransactionListResponse,
    TransactionResponse,
)
from src.api.security import auth_guard
from src.api.services.billing import BillingService
from src.api.services.billing_errors import OrganizationPermissionError
from src.api.services.payment import PaymentService
from src.api.services.pricing import PricingService
from src.core.config import TOKEN_PACKAGES

logger = structlog.get_logger(__name__)


def _txn_to_response(txn: object) -> TransactionResponse:
    """Convert a TokenTransaction model to a response struct."""
    from src.db.models.billing import TokenTransaction

    assert isinstance(txn, TokenTransaction)
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

    path = "/api/v1/billing"
    tags: Sequence[str] | None = ["Billing"]
    guards = [auth_guard]
    dependencies = {"current_user_id": Provide(get_current_user_id)}

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
        offset: int = 0,
        type: str | None = None,
    ) -> TransactionListResponse:
        """Get transaction history for the authenticated user's account."""
        account = await billing_service.resolve_account_for_user(current_user_id, session=session)
        transactions, total = await billing_service.get_transaction_history(
            account.id,
            limit=limit,
            offset=offset,
            transaction_type=type,
            session=session,
        )
        return TransactionListResponse(
            items=[_txn_to_response(t) for t in transactions],
            total=total,
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

    @get("/packages")
    async def get_packages(self) -> list[TokenPackageResponse]:
        """Get available token purchase packages."""
        return [
            TokenPackageResponse(
                id=p.id,
                name=p.name,
                tokens=p.tokens,
                bonus_tokens=p.bonus_tokens,
                total_tokens=p.total_tokens,
                price_usd=str(p.price_usd),
            )
            for p in TOKEN_PACKAGES.values()
        ]

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
    ) -> Response[StripeCheckoutResponse]:
        """Create a Stripe checkout session for token purchase."""
        account = await billing_service.resolve_account_for_user(current_user_id, session=session)
        result = await payment_service.create_stripe_checkout(
            account.id,
            data.package_id,
            current_user_id,
            session=session,
        )
        await session.commit()
        return Response(
            content=StripeCheckoutResponse(
                checkout_url=result.checkout_url,
                session_id=result.session_id,
                payment_id=result.payment_id,
            ),
            status_code=HTTP_201_CREATED,
        )

    @post("/topup/nowpayments")
    async def topup_nowpayments(
        self,
        current_user_id: UUID,
        data: TopUpNowPaymentsRequest,
        session: AsyncSession,
        billing_service: BillingService,
        payment_service: PaymentService,
    ) -> Response[NowPaymentsInvoiceResponse]:
        """Create a NowPayments invoice for token purchase."""
        account = await billing_service.resolve_account_for_user(current_user_id, session=session)
        result = await payment_service.create_nowpayments_invoice(
            account.id,
            data.package_id,
            data.pay_currency,
            current_user_id,
            session=session,
        )
        await session.commit()
        return Response(
            content=NowPaymentsInvoiceResponse(
                invoice_url=result.invoice_url,
                payment_id=result.payment_id,
            ),
            status_code=HTTP_201_CREATED,
        )


class BillingWebhookController(Controller):
    """Webhook endpoints — no auth guard, signature verified in handler."""

    path = "/api/v1/billing/webhooks"
    tags: Sequence[str] | None = ["Billing Webhooks"]

    @post("/stripe")
    async def stripe_webhook(
        self,
        request: Request,
        session: AsyncSession,
        payment_service: PaymentService,
    ) -> Response[dict]:
        """Handle Stripe webhook."""
        payload = await request.body()
        signature = request.headers.get("stripe-signature", "")
        await payment_service.handle_stripe_webhook(payload, signature, session=session)
        await session.commit()
        return Response(content={"received": True}, status_code=HTTP_200_OK)

    @post("/nowpayments")
    async def nowpayments_webhook(
        self,
        request: Request,
        session: AsyncSession,
        payment_service: PaymentService,
    ) -> Response[dict]:
        """Handle NowPayments IPN webhook."""
        payload = await request.json()
        hmac_sig = request.headers.get("x-nowpayments-sig", "")
        await payment_service.handle_nowpayments_webhook(payload, hmac_sig, session=session)
        await session.commit()
        return Response(content={"received": True}, status_code=HTTP_200_OK)
