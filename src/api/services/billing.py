"""Billing service for token account operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from src.api.services.billing_errors import (
    AccountInactiveError,
    AccountNotFoundError,
    InsufficientBalanceError,
    RefundNotEligibleError,
)
from src.core.enums import TransactionType
from src.db.repositories.billing import BillingRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.billing import TokenAccount, TokenTransaction

logger = logging.getLogger(__name__)


class BillingService:
    """Service for token account and transaction operations."""

    async def get_or_create_personal_account(
        self, user_id: UUID, *, session: AsyncSession
    ) -> TokenAccount:
        """Idempotent — returns existing account or creates new one."""
        repo = BillingRepository(session)
        account = await repo.get_account_by_user(user_id)
        if account is not None:
            return account
        return await repo.create_personal_account(id=uuid4(), user_id=user_id)

    async def resolve_account_for_user(
        self, user_id: UUID, *, session: AsyncSession
    ) -> TokenAccount:
        """Returns org TokenAccount if user has membership, else personal.

        Raises:
            AccountNotFoundError: If no account found.
        """
        repo = BillingRepository(session)
        membership = await repo.get_active_membership(user_id)
        if membership:
            account = await repo.get_account_by_organization(membership.organization_id)
            if account is not None:
                return account
        account = await repo.get_account_by_user(user_id)
        if account is None:
            raise AccountNotFoundError(f"No token account found for user {user_id}")
        return account

    async def get_balance(self, account_id: UUID, *, session: AsyncSession) -> int:
        """Authoritative: SELECT COALESCE(SUM(amount), 0) FROM token_transactions."""
        repo = BillingRepository(session)
        return await repo.get_balance(account_id)

    async def assert_sufficient_balance(
        self, account_id: UUID, token_cost: int, *, session: AsyncSession
    ) -> None:
        """Pre-flight balance check — must be called before any provider API call.

        Raises InsufficientBalanceError if the current balance is below token_cost.
        This is a non-locking read; the atomic debit is still performed afterwards
        via check_and_reserve once the provider call succeeds.
        """
        balance = await self.get_balance(account_id, session=session)
        if balance < token_cost:
            raise InsufficientBalanceError(balance=balance, required=token_cost)

    async def check_and_reserve(
        self,
        account_id: UUID,
        token_cost: int,
        job_id: UUID,
        *,
        metadata: dict,
        session: AsyncSession,
    ) -> TokenTransaction:
        """Atomically check balance and create a debit transaction.

        1. SELECT token_accounts WHERE id = account_id FOR UPDATE
        2. Compute live balance from SUM(amount)
        3. Raise InsufficientBalanceError if balance < token_cost
        4. Raise AccountInactiveError if account.is_active is False
        5. Insert TokenTransaction(type='debit', amount=-token_cost, ...)

        Raises:
            AccountNotFoundError: If account not found.
            AccountInactiveError: If account is inactive.
            InsufficientBalanceError: If balance insufficient.
        """
        repo = BillingRepository(session)

        # Lock account row
        account = await repo.get_account_for_update(account_id)
        if account is None:
            raise AccountNotFoundError(f"Token account {account_id} not found")
        if not account.is_active:
            raise AccountInactiveError(f"Token account {account_id} is inactive")

        # Compute live balance
        balance = await repo.get_balance(account_id)
        if balance < token_cost:
            raise InsufficientBalanceError(balance=balance, required=token_cost)

        # Create debit
        new_balance = balance - token_cost
        txn = await repo.create_transaction(
            id=uuid4(),
            account_id=account_id,
            transaction_type=TransactionType.DEBIT.value,
            amount=-token_cost,
            balance_after=new_balance,
            job_id=job_id,
            description="Generation charge",
            metadata=metadata,
        )

        logger.info(
            "token_debit account_id=%s job_id=%s amount=%d balance_after=%d",
            account_id,
            job_id,
            token_cost,
            new_balance,
            extra={
                "event": "token_debit",
                "account_id": str(account_id),
                "job_id": str(job_id),
                "amount": token_cost,
                "balance_after": new_balance,
                "provider": metadata.get("provider"),
                "generation_type": metadata.get("generation_type"),
                "model": metadata.get("model"),
            },
        )

        return txn

    async def refund(
        self,
        job_id: UUID,
        *,
        description: str,
        session: AsyncSession,
    ) -> TokenTransaction:
        """Create a refund (positive) transaction linked to job_id.

        Raises:
            RefundNotEligibleError: If no debit found or already refunded.
        """
        repo = BillingRepository(session)

        # Find original debit
        debit = await repo.get_debit_for_job(job_id)
        if debit is None:
            raise RefundNotEligibleError(f"No debit transaction found for job {job_id}")

        # Check if already refunded
        if await repo.has_refund_for_job(job_id):
            raise RefundNotEligibleError(f"Job {job_id} has already been refunded")

        # Lock account and compute balance
        account = await repo.get_account_for_update(debit.account_id)
        if account is None:
            raise RefundNotEligibleError("Account not found for refund")

        balance = await repo.get_balance(debit.account_id)
        refund_amount = abs(debit.amount)
        new_balance = balance + refund_amount

        txn = await repo.create_transaction(
            id=uuid4(),
            account_id=debit.account_id,
            transaction_type=TransactionType.REFUND.value,
            amount=refund_amount,
            balance_after=new_balance,
            job_id=job_id,
            description=description,
        )

        logger.info(
            "token_refund account_id=%s job_id=%s amount=%d",
            debit.account_id,
            job_id,
            refund_amount,
            extra={
                "event": "token_refund",
                "account_id": str(debit.account_id),
                "job_id": str(job_id),
                "amount": refund_amount,
                "balance_after": new_balance,
                "reason": description,
            },
        )

        return txn

    async def credit(
        self,
        account_id: UUID,
        amount: int,
        payment_id: UUID,
        *,
        description: str,
        payment_provider: str = "",
        session: AsyncSession,
    ) -> TokenTransaction:
        """Credit tokens to an account from a payment."""
        repo = BillingRepository(session)

        account = await repo.get_account_for_update(account_id)
        if account is None:
            raise AccountNotFoundError(f"Token account {account_id} not found")

        balance = await repo.get_balance(account_id)
        new_balance = balance + amount

        txn = await repo.create_transaction(
            id=uuid4(),
            account_id=account_id,
            transaction_type=TransactionType.CREDIT.value,
            amount=amount,
            balance_after=new_balance,
            payment_id=payment_id,
            description=description,
        )

        logger.info(
            "token_credit account_id=%s payment_id=%s amount=%d",
            account_id,
            payment_id,
            amount,
            extra={
                "event": "token_credit",
                "account_id": str(account_id),
                "payment_id": str(payment_id),
                "amount": amount,
                "balance_after": new_balance,
                "payment_provider": payment_provider,
            },
        )

        return txn

    async def admin_adjust(
        self,
        account_id: UUID,
        amount: int,
        admin_id: UUID,
        *,
        description: str,
        session: AsyncSession,
    ) -> TokenTransaction:
        """Admin adjustment: positive = credit, negative = debit.

        For negative adjustments, checks that result >= 0.
        """
        repo = BillingRepository(session)

        account = await repo.get_account_for_update(account_id)
        if account is None:
            raise AccountNotFoundError(f"Token account {account_id} not found")

        balance = await repo.get_balance(account_id)
        new_balance = balance + amount

        if new_balance < 0:
            raise InsufficientBalanceError(balance=balance, required=abs(amount))

        txn = await repo.create_transaction(
            id=uuid4(),
            account_id=account_id,
            transaction_type=TransactionType.ADMIN_ADJUSTMENT.value,
            amount=amount,
            balance_after=new_balance,
            description=description,
            created_by=admin_id,
        )

        logger.info(
            "admin_adjustment account_id=%s admin_id=%s amount=%d",
            account_id,
            admin_id,
            amount,
            extra={
                "event": "admin_adjustment",
                "account_id": str(account_id),
                "admin_id": str(admin_id),
                "amount": amount,
                "balance_after": new_balance,
                "description": description,
            },
        )

        return txn

    async def get_transaction_history(
        self,
        account_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
        transaction_type: str | None = None,
        session: AsyncSession,
    ) -> tuple[Sequence[TokenTransaction], int]:
        """Returns (transactions, total_count). Ordered by created_at DESC."""
        repo = BillingRepository(session)
        return await repo.list_transactions(
            account_id,
            limit=limit,
            offset=offset,
            transaction_type=transaction_type,
        )
