"""Repository for billing-related database operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import func, literal, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.core.enums import AccountType, TransactionType
from src.db.models.billing import (
    Organization,
    OrganizationMember,
    Payment,
    PricingRule,
    TokenAccount,
    TokenTransaction,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class BillingRepository:
    """Repository for billing and token account operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -------------------------------------------------------------------------
    # TokenAccount
    # -------------------------------------------------------------------------

    async def get_account(self, account_id: UUID) -> TokenAccount | None:
        return await self._session.get(TokenAccount, account_id)

    async def get_account_with_organization(self, account_id: UUID) -> TokenAccount | None:
        """Get a TokenAccount with its organization eagerly loaded."""
        result = await self._session.execute(
            select(TokenAccount)
            .where(TokenAccount.id == account_id)
            .options(joinedload(TokenAccount.organization))
        )
        return result.scalar_one_or_none()

    async def get_account_for_update(self, account_id: UUID) -> TokenAccount | None:
        """Get account with FOR UPDATE lock for balance operations."""
        result = await self._session.execute(
            select(TokenAccount).where(TokenAccount.id == account_id).with_for_update()
        )
        return result.scalar_one_or_none()

    async def get_account_by_user(self, user_id: UUID) -> TokenAccount | None:
        result = await self._session.execute(
            select(TokenAccount).where(
                TokenAccount.user_id == user_id,
                TokenAccount.account_type == AccountType.PERSONAL.value,
            )
        )
        return result.scalar_one_or_none()

    async def get_account_by_organization(self, organization_id: UUID) -> TokenAccount | None:
        result = await self._session.execute(
            select(TokenAccount)
            .options(joinedload(TokenAccount.organization))
            .where(
                TokenAccount.organization_id == organization_id,
                TokenAccount.account_type == AccountType.ENTERPRISE.value,
            )
        )
        return result.scalar_one_or_none()

    async def create_personal_account(
        self, *, id: UUID, user_id: UUID, product_id: str
    ) -> TokenAccount:
        account = TokenAccount(
            id=id,
            account_type=AccountType.PERSONAL.value,
            user_id=user_id,
            product_id=product_id,
        )
        self._session.add(account)
        await self._session.flush()
        return account

    async def create_enterprise_account(
        self, *, id: UUID, organization_id: UUID, product_id: str
    ) -> TokenAccount:
        account = TokenAccount(
            id=id,
            account_type=AccountType.ENTERPRISE.value,
            organization_id=organization_id,
            product_id=product_id,
        )
        self._session.add(account)
        await self._session.flush()
        return account

    # -------------------------------------------------------------------------
    # TokenTransaction
    # -------------------------------------------------------------------------

    async def get_balance(self, account_id: UUID) -> int:
        """Authoritative balance: SUM(amount) from ledger."""
        result = await self._session.execute(
            select(func.coalesce(func.sum(TokenTransaction.amount), 0)).where(
                TokenTransaction.account_id == account_id
            )
        )
        return int(result.scalar_one())

    async def create_transaction(
        self,
        *,
        id: UUID,
        account_id: UUID,
        transaction_type: str,
        amount: int,
        balance_after: int,
        product_id: str,
        job_id: UUID | None = None,
        payment_id: UUID | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: UUID | None = None,
    ) -> TokenTransaction:
        txn = TokenTransaction(
            id=id,
            account_id=account_id,
            transaction_type=transaction_type,
            amount=amount,
            balance_after=balance_after,
            job_id=job_id,
            payment_id=payment_id,
            description=description,
            metadata_=metadata or {},
            created_by=created_by,
            product_id=product_id,
        )
        self._session.add(txn)
        await self._session.flush()
        return txn

    async def get_debit_for_job(self, job_id: UUID) -> TokenTransaction | None:
        """Find the debit transaction for a given job."""
        result = await self._session.execute(
            select(TokenTransaction).where(
                TokenTransaction.job_id == job_id,
                TokenTransaction.transaction_type == TransactionType.DEBIT.value,
            )
        )
        return result.scalar_one_or_none()

    async def has_refund_for_job(self, job_id: UUID) -> bool:
        """Check if a refund already exists for a job."""
        result = await self._session.execute(
            select(func.count(TokenTransaction.id)).where(
                TokenTransaction.job_id == job_id,
                TokenTransaction.transaction_type == TransactionType.REFUND.value,
            )
        )
        return int(result.scalar_one()) > 0

    async def get_transaction_history(
        self,
        account_id: UUID,
        *,
        limit: int = 50,
        transaction_type: str | None = None,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[TokenTransaction]:
        """List transactions using cursor-based pagination.

        Uses limit+1 fetch pattern — caller checks ``len(result) > limit``
        to determine ``has_more``.

        Args:
            account_id: Account to list transactions for.
            limit: Max results (fetch limit+1 for has_more).
            transaction_type: Optional transaction type filter.
            cursor_ts: ``created_at`` of the last item on the previous page.
            cursor_id: ``id`` of the last item on the previous page.

        Returns:
            Sequence of TokenTransaction instances.
        """
        base = select(TokenTransaction).where(TokenTransaction.account_id == account_id)

        if transaction_type is not None:
            base = base.where(TokenTransaction.transaction_type == transaction_type)

        if cursor_ts is not None and cursor_id is not None:
            base = base.where(
                tuple_(TokenTransaction.created_at, TokenTransaction.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            base.order_by(TokenTransaction.created_at.desc(), TokenTransaction.id.desc()).limit(
                limit + 1
            )
        )
        return result.scalars().all()

    # -------------------------------------------------------------------------
    # PricingRule
    # -------------------------------------------------------------------------

    async def get_active_price(
        self,
        provider: str,
        generation_type: str,
        model: str | None,
    ) -> PricingRule | None:
        """Get the most specific active pricing rule.

        Priority: exact match (with model) first, then wildcard (model IS NULL).
        """
        now = datetime.now(UTC)
        active_filter = [
            PricingRule.is_active.is_(True),
            PricingRule.effective_from <= now,
            (PricingRule.effective_until.is_(None)) | (PricingRule.effective_until > now),
        ]

        # Try exact match first
        if model is not None:
            result = await self._session.execute(
                select(PricingRule)
                .where(
                    PricingRule.provider == provider,
                    PricingRule.generation_type == generation_type,
                    PricingRule.model == model,
                    *active_filter,
                )
                .order_by(PricingRule.effective_from.desc())
                .limit(1)
            )
            rule = result.scalar_one_or_none()
            if rule is not None:
                return rule

        # Fallback to wildcard
        result = await self._session.execute(
            select(PricingRule)
            .where(
                PricingRule.provider == provider,
                PricingRule.generation_type == generation_type,
                PricingRule.model.is_(None),
                *active_filter,
            )
            .order_by(PricingRule.effective_from.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_pricing_rules(self, *, active_only: bool = True) -> Sequence[PricingRule]:
        query = select(PricingRule)
        if active_only:
            now = datetime.now(UTC)
            query = query.where(
                PricingRule.is_active.is_(True),
                PricingRule.effective_from <= now,
                (PricingRule.effective_until.is_(None)) | (PricingRule.effective_until > now),
            )
        result = await self._session.execute(
            query.order_by(PricingRule.provider, PricingRule.generation_type)
        )
        return result.scalars().all()

    async def get_pricing_rule(self, rule_id: UUID) -> PricingRule | None:
        return await self._session.get(PricingRule, rule_id)

    async def create_pricing_rule(
        self,
        *,
        id: UUID,
        provider: str,
        generation_type: str,
        model: str | None,
        token_cost: int,
        notes: str | None,
        created_by: UUID,
    ) -> PricingRule:
        rule = PricingRule(
            id=id,
            provider=provider,
            generation_type=generation_type,
            model=model,
            token_cost=token_cost,
            notes=notes,
            created_by=created_by,
        )
        self._session.add(rule)
        await self._session.flush()
        return rule

    # -------------------------------------------------------------------------
    # Payment
    # -------------------------------------------------------------------------

    async def get_payment(self, payment_id: UUID) -> Payment | None:
        return await self._session.get(Payment, payment_id)

    async def get_payment_by_external_id(self, external_id: str) -> Payment | None:
        result = await self._session.execute(
            select(Payment).where(Payment.external_id == external_id)
        )
        return result.scalar_one_or_none()

    async def create_payment(
        self,
        *,
        id: UUID,
        account_id: UUID,
        payment_provider: str,
        external_id: str,
        status: str,
        amount_usd: object,  # Decimal
        tokens_granted: int,
        product_id: str,
        currency: str = "USD",
        provider_metadata: dict[str, Any] | None = None,
        created_by: UUID,
    ) -> Payment:
        payment = Payment(
            id=id,
            account_id=account_id,
            payment_provider=payment_provider,
            external_id=external_id,
            status=status,
            amount_usd=amount_usd,
            tokens_granted=tokens_granted,
            currency=currency,
            provider_metadata=provider_metadata or {},
            created_by=created_by,
            product_id=product_id,
        )
        self._session.add(payment)
        await self._session.flush()
        return payment

    async def list_payments(
        self,
        *,
        status: str | None = None,
        payment_provider: str | None = None,
        limit: int = 50,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[Payment]:
        """List payments using cursor-based pagination.

        Uses limit+1 fetch pattern — caller checks ``len(result) > limit``
        to determine ``has_more``.

        Args:
            status: Optional status filter.
            payment_provider: Optional provider filter.
            limit: Max results (fetch limit+1 for has_more).
            cursor_ts: ``created_at`` of the last item on the previous page.
            cursor_id: ``id`` of the last item on the previous page.

        Returns:
            Sequence of Payment instances.
        """
        base = select(Payment)

        if status is not None:
            base = base.where(Payment.status == status)
        if payment_provider is not None:
            base = base.where(Payment.payment_provider == payment_provider)

        if cursor_ts is not None and cursor_id is not None:
            base = base.where(
                tuple_(Payment.created_at, Payment.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            base.order_by(Payment.created_at.desc(), Payment.id.desc()).limit(limit + 1)
        )
        return result.scalars().all()

    # -------------------------------------------------------------------------
    # Organization
    # -------------------------------------------------------------------------

    async def get_organization(self, org_id: UUID) -> Organization | None:
        return await self._session.get(Organization, org_id)

    async def get_organization_by_slug(self, slug: str) -> Organization | None:
        result = await self._session.execute(select(Organization).where(Organization.slug == slug))
        return result.scalar_one_or_none()

    async def create_organization(
        self,
        *,
        id: UUID,
        name: str,
        slug: str,
        owner_id: UUID,
        product_id: str,
    ) -> Organization:
        org = Organization(id=id, name=name, slug=slug, owner_id=owner_id, product_id=product_id)
        self._session.add(org)
        await self._session.flush()
        return org

    async def get_active_membership(self, user_id: UUID) -> OrganizationMember | None:
        """Get user's active organization membership (if any)."""
        result = await self._session.execute(
            select(OrganizationMember)
            .join(Organization)
            .where(
                OrganizationMember.user_id == user_id,
                Organization.is_active.is_(True),
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_membership(self, org_id: UUID, user_id: UUID) -> OrganizationMember | None:
        result = await self._session.execute(
            select(OrganizationMember).where(
                OrganizationMember.organization_id == org_id,
                OrganizationMember.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def create_membership(
        self,
        *,
        id: UUID,
        organization_id: UUID,
        user_id: UUID,
        role: str,
        product_id: str,
    ) -> OrganizationMember:
        member = OrganizationMember(
            id=id,
            organization_id=organization_id,
            user_id=user_id,
            role=role,
            product_id=product_id,
        )
        self._session.add(member)
        await self._session.flush()
        return member

    async def delete_membership(self, org_id: UUID, user_id: UUID) -> bool:
        member = await self.get_membership(org_id, user_id)
        if member is None:
            return False
        await self._session.delete(member)
        await self._session.flush()
        return True

    async def list_organizations(
        self,
        *,
        is_active: bool | None = None,
        limit: int = 50,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[tuple[Organization, int, int]]:
        """List organisations with aggregated member count and token balance.

        Uses a single SQL statement with LEFT JOINs and GROUP BY to avoid N+1.
        Balance is the sum of all token_transactions for the org's enterprise account.
        Uses limit+1 fetch pattern — caller checks ``len(result) > limit``
        to determine ``has_more``.

        Args:
            is_active: Filter by active status when not None.
            limit: Maximum number of results to return (fetch limit+1 for has_more).
            cursor_ts: ``created_at`` of the last item on the previous page.
            cursor_id: ``id`` of the last item on the previous page.

        Returns:
            Sequence of (Organization, member_count, token_balance) tuples.
        """
        member_count_col = func.count(func.distinct(OrganizationMember.id)).label("member_count")
        token_balance_col = func.coalesce(func.sum(TokenTransaction.amount), 0).label(
            "token_balance"
        )

        base_q = (
            select(Organization, member_count_col, token_balance_col)
            .outerjoin(
                OrganizationMember,
                OrganizationMember.organization_id == Organization.id,
            )
            .outerjoin(
                TokenAccount,
                (TokenAccount.organization_id == Organization.id)
                & (TokenAccount.account_type == AccountType.ENTERPRISE.value),
            )
            .outerjoin(TokenTransaction, TokenTransaction.account_id == TokenAccount.id)
            .group_by(Organization.id)
        )

        if is_active is not None:
            base_q = base_q.where(Organization.is_active == is_active)

        if cursor_ts is not None and cursor_id is not None:
            base_q = base_q.where(
                tuple_(Organization.created_at, Organization.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            base_q.order_by(Organization.created_at.desc(), Organization.id.desc()).limit(limit + 1)
        )
        rows = result.all()
        return [(row[0], int(row[1]), int(row[2])) for row in rows]

    async def list_members(self, org_id: UUID) -> Sequence[OrganizationMember]:
        result = await self._session.execute(
            select(OrganizationMember).where(OrganizationMember.organization_id == org_id)
        )
        return result.scalars().all()
