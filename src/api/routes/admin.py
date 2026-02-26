"""Admin API routes — account management, pricing, payments."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from uuid import UUID

from litestar import Controller, Response, delete, get, patch, post
from litestar.di import Provide
from litestar.exceptions import NotFoundException
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_admin_user
from src.api.routes.billing import _txn_to_response
from src.api.schemas.billing import (
    AdminAdjustRequest,
    AdminAdjustResponse,
    BalanceResponse,
    CreatePricingRuleRequest,
    PatchPricingRuleRequest,
    PaymentListResponse,
    PaymentResponse,
    PricingRuleResponse,
    TransactionListResponse,
)
from src.api.security import auth_guard
from src.api.services.billing import BillingService
from src.api.services.billing_errors import AccountNotFoundError
from src.api.services.pricing import PricingService
from src.db.models import User
from src.db.repositories.billing import BillingRepository

logger = logging.getLogger(__name__)


class AdminController(Controller):
    """Admin endpoints for billing management."""

    path = "/api/v1/admin"
    tags: Sequence[str] | None = ["Admin"]
    guards = [auth_guard]
    dependencies = {
        "admin_user": Provide(get_current_admin_user),
    }

    # -------------------------------------------------------------------------
    # Account management
    # -------------------------------------------------------------------------

    @get("/accounts/{account_id:uuid}/balance")
    async def get_account_balance(
        self,
        admin_user: User,
        account_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
    ) -> BalanceResponse:
        """Get balance for any account."""
        logger.info("Admin %s viewing balance for account %s", admin_user.id, account_id)
        repo = BillingRepository(session)
        account = await repo.get_account_with_organization(account_id)
        if account is None:
            raise AccountNotFoundError(f"Account {account_id} not found")

        balance = await billing_service.get_balance(account_id, session=session)
        org_name: str | None = None
        if account.account_type == "enterprise" and account.organization is not None:
            org_name = account.organization.name

        return BalanceResponse(
            account_id=account.id,
            account_type=account.account_type,
            balance=balance,
            organization_name=org_name,
        )

    @get("/accounts/{account_id:uuid}/transactions")
    async def get_account_transactions(
        self,
        admin_user: User,
        account_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
        limit: int = 50,
        offset: int = 0,
        type: str | None = None,
    ) -> TransactionListResponse:
        """Get transaction history for any account."""
        logger.info("Admin %s viewing transactions for account %s", admin_user.id, account_id)
        transactions, total = await billing_service.get_transaction_history(
            account_id,
            limit=limit,
            offset=offset,
            transaction_type=type,
            session=session,
        )
        return TransactionListResponse(
            items=[_txn_to_response(t) for t in transactions],
            total=total,
        )

    @post("/accounts/{account_id:uuid}/adjust")
    async def adjust_account(
        self,
        admin_user: User,
        account_id: UUID,
        data: AdminAdjustRequest,
        session: AsyncSession,
        billing_service: BillingService,
    ) -> AdminAdjustResponse:
        """Admin balance adjustment. Positive = credit, negative = debit."""
        logger.info(
            "Admin %s adjusting account %s by %d tokens",
            admin_user.id,
            account_id,
            data.amount,
        )
        txn = await billing_service.admin_adjust(
            account_id,
            data.amount,
            admin_user.id,
            description=data.description,
            session=session,
        )
        await session.commit()
        new_balance = await billing_service.get_balance(account_id, session=session)
        return AdminAdjustResponse(
            transaction=_txn_to_response(txn),
            new_balance=new_balance,
        )

    # -------------------------------------------------------------------------
    # Pricing management
    # -------------------------------------------------------------------------

    @get("/pricing")
    async def list_pricing_rules(
        self,
        admin_user: User,
        session: AsyncSession,
        pricing_service: PricingService,
        active_only: bool = True,
    ) -> list[PricingRuleResponse]:
        """List pricing rules."""
        logger.info("Admin %s listing pricing rules", admin_user.id)
        rules = await pricing_service.list_catalog(active_only=active_only, session=session)
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

    @post("/pricing")
    async def create_pricing_rule(
        self,
        admin_user: User,
        data: CreatePricingRuleRequest,
        session: AsyncSession,
        pricing_service: PricingService,
    ) -> Response[PricingRuleResponse]:
        """Create a new pricing rule."""
        logger.info(
            f"Admin {admin_user.id} creating pricing rule: {data.provider}/{data.generation_type}/{data.model} at {data.token_cost} tokens",
        )
        rule = await pricing_service.create_rule(
            provider=data.provider,
            generation_type=data.generation_type,
            model=data.model,
            token_cost=data.token_cost,
            notes=data.notes,
            admin_id=admin_user.id,
            session=session,
        )
        await session.commit()
        return Response(
            content=PricingRuleResponse(
                id=rule.id,
                provider=rule.provider,
                generation_type=rule.generation_type,
                model=rule.model,
                token_cost=rule.token_cost,
                is_active=rule.is_active,
                effective_from=rule.effective_from,
                effective_until=rule.effective_until,
                notes=rule.notes,
            ),
            status_code=HTTP_201_CREATED,
        )

    @patch("/pricing/{rule_id:uuid}")
    async def update_pricing_rule(
        self,
        admin_user: User,
        rule_id: UUID,
        data: PatchPricingRuleRequest,
        session: AsyncSession,
        pricing_service: PricingService,
    ) -> PricingRuleResponse:
        """Update a pricing rule."""
        logger.info(f"Admin {admin_user.id} updating pricing rule {rule_id}")
        rule = await pricing_service.update_rule(
            rule_id,
            token_cost=data.token_cost,
            is_active=data.is_active,
            effective_until=data.effective_until,
            notes=data.notes,
            session=session,
        )
        await session.commit()
        return PricingRuleResponse(
            id=rule.id,
            provider=rule.provider,
            generation_type=rule.generation_type,
            model=rule.model,
            token_cost=rule.token_cost,
            is_active=rule.is_active,
            effective_from=rule.effective_from,
            effective_until=rule.effective_until,
            notes=rule.notes,
        )

    @delete("/pricing/{rule_id:uuid}", status_code=HTTP_200_OK)
    async def deactivate_pricing_rule(
        self,
        admin_user: User,
        rule_id: UUID,
        session: AsyncSession,
        pricing_service: PricingService,
    ) -> dict:
        """Soft deactivate a pricing rule."""
        logger.info("Admin %s deactivating pricing rule %s", admin_user.id, rule_id)
        await pricing_service.deactivate_rule(rule_id, session=session)
        await session.commit()
        return {"message": "Rule deactivated"}

    # -------------------------------------------------------------------------
    # Payment management
    # -------------------------------------------------------------------------

    @get("/payments")
    async def list_payments(
        self,
        admin_user: User,
        session: AsyncSession,
        status: str | None = None,
        payment_provider: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> PaymentListResponse:
        """List all payments."""
        logger.info("Admin %s listing payments", admin_user.id)
        repo = BillingRepository(session)
        payments, total = await repo.list_payments(
            status=status,
            payment_provider=payment_provider,
            limit=limit,
            offset=offset,
        )
        return PaymentListResponse(
            items=[
                PaymentResponse(
                    id=p.id,
                    payment_provider=p.payment_provider,
                    status=p.status,
                    amount_usd=str(p.amount_usd),
                    tokens_granted=p.tokens_granted,
                    currency=p.currency,
                    created_at=p.created_at,
                    completed_at=p.completed_at,
                )
                for p in payments
            ],
            total=total,
        )

    @get("/payments/{payment_id:uuid}")
    async def get_payment(
        self,
        admin_user: User,
        payment_id: UUID,
        session: AsyncSession,
    ) -> PaymentResponse:
        """Get a single payment."""
        logger.info("Admin %s viewing payment %s", admin_user.id, payment_id)
        repo = BillingRepository(session)
        payment = await repo.get_payment(payment_id)
        if payment is None:
            raise NotFoundException(detail=f"Payment {payment_id} not found")
        return PaymentResponse(
            id=payment.id,
            payment_provider=payment.payment_provider,
            status=payment.status,
            amount_usd=str(payment.amount_usd),
            tokens_granted=payment.tokens_granted,
            currency=payment.currency,
            created_at=payment.created_at,
            completed_at=payment.completed_at,
        )

    # -------------------------------------------------------------------------
    # User account lookup
    # -------------------------------------------------------------------------

    @get("/users/{user_id:uuid}/account")
    async def get_user_account(
        self,
        admin_user: User,
        user_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
    ) -> BalanceResponse:
        """Get a user's resolved account (personal or org) + balance."""
        logger.info(f"Admin {admin_user.id} viewing account for user {user_id}")
        account = await billing_service.resolve_account_for_user(user_id, session=session)
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

    @get("/organizations/{org_id:uuid}/account")
    async def get_org_account(
        self,
        admin_user: User,
        org_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
    ) -> BalanceResponse:
        """Get the enterprise token account and balance for an organization."""
        logger.info("Admin %s viewing account for org %s", admin_user.id, org_id)
        repo = BillingRepository(session)
        account = await repo.get_account_by_organization(org_id)
        if account is None:
            raise NotFoundException(detail=f"No token account found for organization {org_id}")

        balance = await billing_service.get_balance(account.id, session=session)
        org_name = account.organization.name if account.organization is not None else None

        return BalanceResponse(
            account_id=account.id,
            account_type=account.account_type,
            balance=balance,
            organization_name=org_name,
        )
