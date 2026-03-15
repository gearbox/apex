"""Admin API routes — account management, pricing, payments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

import structlog
from litestar import Controller, Response, delete, get, patch, post
from litestar.di import Provide
from litestar.exceptions import NotFoundException, PermissionDeniedException, ValidationException
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_admin_user
from src.api.routes.billing import _txn_to_response
from src.api.schemas.admin import (
    AdminOrgListResponse,
    AdminOrgResponse,
    AdminPatchUserRequest,
    AdminUserListResponse,
    AdminUserResponse,
)
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
from src.api.schemas.models import (
    GenerationModelResponse,
    ModelListResponse,
    SetModelEnabledRequest,
)
from src.api.security import auth_guard
from src.api.services.billing import BillingService
from src.api.services.billing_errors import AccountNotFoundError
from src.api.services.pricing import PricingService
from src.core.enums import UserRole
from src.db.models import User
from src.db.repositories.billing import BillingRepository
from src.db.repositories.generation_model import GenerationModelRepository
from src.db.repositories.user import UserRepository

logger = structlog.get_logger(__name__)


class AdminController(Controller):
    """Admin endpoints for billing management."""

    path = "/v1/admin"
    tags: Sequence[str] | None = ["Admin"]
    guards = [auth_guard]
    dependencies = {
        "admin_user": Provide(get_current_admin_user),
    }

    # -------------------------------------------------------------------------
    # User management
    # -------------------------------------------------------------------------

    @get("/users")
    async def list_users(
        self,
        admin_user: User,
        session: AsyncSession,
        is_active: bool | None = None,
        role: str | None = None,
        email: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> AdminUserListResponse:
        """List all users with optional filtering. Excludes SYSTEM role users."""
        logger.info(
            "admin.listing_users",
            admin_id=str(admin_user.id),
            is_active=is_active,
            role=role,
            email=email,
        )
        repo = UserRepository(session)
        users, total = await repo.list_users(
            is_active=is_active,
            role=role,
            email_contains=email,
            limit=limit,
            offset=offset,
        )
        return AdminUserListResponse(
            items=[
                AdminUserResponse(
                    id=u.id,
                    email=u.email,
                    display_name=u.display_name,
                    role=u.role.value if hasattr(u.role, "value") else u.role,
                    subscription_tier=(
                        u.subscription_tier.value
                        if hasattr(u.subscription_tier, "value")
                        else u.subscription_tier
                    ),
                    is_active=u.is_active,
                    email_verified_at=u.email_verified_at,
                    created_at=u.created_at,
                    updated_at=u.updated_at,
                )
                for u in users
            ],
            total=total,
        )

    @patch("/users/{user_id:uuid}", status_code=HTTP_200_OK)
    async def patch_user(
        self,
        admin_user: User,
        user_id: UUID,
        data: AdminPatchUserRequest,
        session: AsyncSession,
    ) -> AdminUserResponse:
        """Update a user's role, subscription tier, or active status."""
        if user_id == admin_user.id:
            raise PermissionDeniedException(
                detail="Admins cannot modify their own account via this endpoint"
            )
        if data.role == UserRole.SYSTEM:
            raise ValidationException(detail="Cannot set user role to system")

        logger.info(
            "admin.patching_user",
            admin_id=str(admin_user.id),
            target_user_id=str(user_id),
            role=data.role,
            subscription_tier=data.subscription_tier,
            is_active=data.is_active,
            locale=data.locale,
        )
        repo = UserRepository(session)
        user = await repo.update_user_admin(
            user_id,
            role=data.role.value if data.role is not None else None,
            subscription_tier=(
                data.subscription_tier.value if data.subscription_tier is not None else None
            ),
            is_active=data.is_active,
            locale=data.locale.value if data.locale is not None else None,
        )
        if user is None:
            raise NotFoundException(detail=f"User {user_id} not found")
        await session.commit()
        return AdminUserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            role=user.role.value if hasattr(user.role, "value") else user.role,
            subscription_tier=(
                user.subscription_tier.value
                if hasattr(user.subscription_tier, "value")
                else user.subscription_tier
            ),
            is_active=user.is_active,
            email_verified_at=user.email_verified_at,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

    @get("/organizations")
    async def list_organizations(
        self,
        admin_user: User,
        session: AsyncSession,
        is_active: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> AdminOrgListResponse:
        """List all organisations with member count and token balance."""
        logger.info("admin.listing_organizations", admin_id=str(admin_user.id))
        repo = BillingRepository(session)
        rows, total = await repo.list_organizations(
            is_active=is_active,
            limit=limit,
            offset=offset,
        )
        return AdminOrgListResponse(
            items=[
                AdminOrgResponse(
                    id=org.id,
                    name=org.name,
                    slug=org.slug,
                    owner_id=org.owner_id,
                    is_active=org.is_active,
                    member_count=member_count,
                    token_balance=token_balance,
                    created_at=org.created_at,
                )
                for org, member_count, token_balance in rows
            ],
            total=total,
        )

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
        logger.info(
            "admin.viewing_balance", admin_id=str(admin_user.id), account_id=str(account_id)
        )
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
        logger.info(
            "admin.viewing_transactions", admin_id=str(admin_user.id), account_id=str(account_id)
        )
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
            "admin.adjusting_account",
            admin_id=str(admin_user.id),
            account_id=str(account_id),
            amount=data.amount,
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
        logger.info("admin.listing_pricing_rules", admin_id=str(admin_user.id))
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
            "admin.creating_pricing_rule",
            admin_id=str(admin_user.id),
            provider=data.provider,
            generation_type=str(data.generation_type),
            model=str(data.model),
            token_cost=data.token_cost,
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
        logger.info(
            "admin.updating_pricing_rule", admin_id=str(admin_user.id), rule_id=str(rule_id)
        )
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
    ) -> dict[str, Any]:
        """Soft deactivate a pricing rule."""
        logger.info(
            "admin.deactivating_pricing_rule", admin_id=str(admin_user.id), rule_id=str(rule_id)
        )
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
        logger.info("admin.listing_payments", admin_id=str(admin_user.id))
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
        logger.info(
            "admin.viewing_payment", admin_id=str(admin_user.id), payment_id=str(payment_id)
        )
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
        logger.info("admin.viewing_user_account", admin_id=str(admin_user.id), user_id=str(user_id))
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
        logger.info("admin.viewing_org_account", admin_id=str(admin_user.id), org_id=str(org_id))
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

    # -------------------------------------------------------------------------
    # Model enable/disable management
    # -------------------------------------------------------------------------

    @get("/models")
    async def list_models(
        self,
        admin_user: User,
        session: AsyncSession,
        enabled_only: bool = False,
    ) -> ModelListResponse:
        """List all generation models. Pass enabled_only=true to filter."""
        logger.info("admin.listing_models", admin_id=str(admin_user.id), enabled_only=enabled_only)
        repo = GenerationModelRepository(session)
        models = await repo.list_enabled() if enabled_only else await repo.list_all()
        items = [
            GenerationModelResponse(
                model_key=m.model_key,
                provider=m.provider,
                name=m.name,
                description=m.description,
                is_enabled=m.is_enabled,
                created_at=m.created_at,
                updated_at=m.updated_at,
            )
            for m in models
        ]
        return ModelListResponse(items=items, total=len(items))

    @patch("/models/{model_key:str}")
    async def toggle_model(
        self,
        admin_user: User,
        model_key: str,
        data: SetModelEnabledRequest,
        session: AsyncSession,
    ) -> GenerationModelResponse:
        """Toggle the is_enabled flag for a model."""
        logger.info(
            "admin.toggling_model",
            admin_id=str(admin_user.id),
            model_key=model_key,
            is_enabled=data.is_enabled,
        )
        repo = GenerationModelRepository(session)
        model = await repo.set_enabled(model_key, data.is_enabled)
        if model is None:
            raise NotFoundException(detail=f"Model '{model_key}' not found")
        await session.commit()
        return GenerationModelResponse(
            model_key=model.model_key,
            provider=model.provider,
            name=model.name,
            description=model.description,
            is_enabled=model.is_enabled,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
