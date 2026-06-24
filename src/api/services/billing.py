"""Billing service for token account operations."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from src.api.schemas.events import BalanceUpdatedPayload, EventType
from src.api.services.billing_errors import (
    AccountInactiveError,
    AccountNotFoundError,
    InsufficientBalanceError,
    OrganizationPermissionError,
    RefundNotEligibleError,
)
from src.core.enums import AccountType, TransactionType
from src.core.uid import new_id
from src.db.repositories.billing import BillingRepository
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.event_bus import EventBus
    from src.db.models.billing import TokenAccount, TokenTransaction

logger = structlog.get_logger(__name__)


class BillingService:
    """Service for token account and transaction operations.

    Debit paths — every writer of a DEBIT/negative-amount transaction:

    | Path                       | Class      | Behaviour                                                  |
    |----------------------------|------------|------------------------------------------------------------|
    | ``check_and_reserve``      | **Refuse** | raises ``InsufficientBalanceError`` when balance < cost.   |
    | ``admin_adjust`` (negative)| **Refuse** | raises when ``new_balance < 0``. Unchanged.                |
    | ``settle_session_usage``   | **Record** | records full incurred usage; may drive balance negative.   |
    | ``refund`` / ``partial_refund`` / ``credit`` | n/a | credits only; never create debt. |

    No other code path calls ``create_transaction`` with a negative amount.
    A future chargeback/clawback handler MUST route through a *recording* primitive
    (extend ``settle_session_usage`` or add a sibling), never a refusing one.
    """

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._event_bus = event_bus

    async def _publish_balance_update(
        self,
        *,
        user_ids: Sequence[UUID],
        account_id: UUID,
        balance: int,
        delta: int,
        transaction_type: str,
    ) -> None:
        if self._event_bus is None or not user_ids:
            return

        payload = BalanceUpdatedPayload(
            account_id=account_id,
            balance=balance,
            delta=delta,
            transaction_type=transaction_type,
        )

        for uid in user_ids:
            await self._event_bus.publish(
                user_id=uid,
                event_type=EventType.BALANCE_UPDATED,
                payload=payload,
            )

    async def get_or_create_personal_account(
        self, user_id: UUID, *, session: AsyncSession, product_id: str
    ) -> TokenAccount:
        """Idempotent — returns existing account or creates new one."""
        repo = BillingRepository(session)
        account = await repo.get_account_by_user(user_id)
        if account is not None:
            return account
        return await repo.create_personal_account(
            id=new_id(), user_id=user_id, product_id=product_id
        )

    async def resolve_account_for_user(
        self, user_id: UUID, *, session: AsyncSession
    ) -> TokenAccount:
        """Returns the appropriate TokenAccount for the user.

        If the user has a stored preference, honours it strictly (no silent fallback).
        If no preference is set, defaults to enterprise account when org member,
        else personal account.

        Raises:
            AccountNotFoundError: If no account found.
        """
        user_repo = UserRepository(session)
        preference = await user_repo.get_preferred_billing_account(user_id)

        repo = BillingRepository(session)

        if preference == AccountType.PERSONAL.value:
            account = await repo.get_account_by_user(user_id)
            if account is None:
                raise AccountNotFoundError(f"No personal token account found for user {user_id}")
            return account

        if preference == AccountType.ENTERPRISE.value:
            membership = await repo.get_active_membership(user_id)
            if membership is None:
                raise AccountNotFoundError(
                    f"No active org membership found for user {user_id} (stale preference)"
                )
            account = await repo.get_account_by_organization(membership.organization_id)
            if account is None:
                raise AccountNotFoundError(
                    f"No enterprise token account found for org {membership.organization_id}"
                )
            return account

        # No preference — default: enterprise if member, else personal
        membership = await repo.get_active_membership(user_id)
        if membership:
            account = await repo.get_account_by_organization(membership.organization_id)
            if account is not None:
                return account
        account = await repo.get_account_by_user(user_id)
        if account is None:
            raise AccountNotFoundError(f"No token account found for user {user_id}")
        return account

    async def set_billing_account_preference(
        self,
        user_id: UUID,
        account_type: AccountType,
        *,
        session: AsyncSession,
    ) -> None:
        """Persist the user's preferred billing account.

        Validates that:
        - For AccountType.ENTERPRISE: user must have an active org membership,
          else raise OrganizationPermissionError.
        - For AccountType.PERSONAL: always valid.

        Raises:
            OrganizationPermissionError: If enterprise chosen but no membership.
            AccountNotFoundError: If user not found.
        """
        if account_type == AccountType.ENTERPRISE:
            repo = BillingRepository(session)
            membership = await repo.get_active_membership(user_id)
            if membership is None:
                raise OrganizationPermissionError(
                    "Cannot set enterprise billing preference: user has no active org membership"
                )

        user_repo = UserRepository(session)
        user = await user_repo.set_preferred_billing_account(user_id, account_type.value)
        if user is None:
            raise AccountNotFoundError(f"User {user_id} not found")

        logger.info(
            "billing.account_preference_set",
            user_id=str(user_id),
            account_type=account_type.value,
        )

    async def get_billing_account_preference(
        self,
        user_id: UUID,
        *,
        session: AsyncSession,
    ) -> str | None:
        """Return the user's stored billing account preference, or None if unset."""
        user_repo = UserRepository(session)
        return await user_repo.get_preferred_billing_account(user_id)

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
        job_id: UUID | None,
        *,
        metadata: dict[str, Any],
        session: AsyncSession,
        product_id: str,
        user_id: UUID | None = None,
        description: str = "Generation charge",
    ) -> TokenTransaction:
        """Atomically check balance and create a debit transaction.

        ``job_id`` may be None for billing events that are not 1:1 with a
        generation job (e.g. GPU session overage debits, where the parent
        session is recorded in ``metadata`` instead). The base reservation
        for a GPU session keeps using the session id as ``job_id`` so that
        full refund-on-failure via ``refund(job_id=...)`` continues to work.

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
            id=new_id(),
            account_id=account_id,
            transaction_type=TransactionType.DEBIT.value,
            amount=-token_cost,
            balance_after=new_balance,
            job_id=job_id,
            description=description,
            metadata=metadata,
            product_id=product_id,
        )

        logger.info(
            "billing.debit_processed",
            account_id=str(account_id),
            job_id=str(job_id) if job_id is not None else None,
            amount=token_cost,
            balance_after=new_balance,
            provider=metadata.get("provider"),
            generation_type=metadata.get("generation_type"),
            model=metadata.get("model"),
        )

        await self._publish_balance_update(
            user_ids=[user_id] if user_id is not None else [],
            account_id=account_id,
            balance=new_balance,
            delta=-token_cost,
            transaction_type=TransactionType.DEBIT.value,
        )

        return txn

    async def settle_session_usage(
        self,
        account_id: UUID,
        owed: int,
        *,
        session_id: UUID,
        model_type: str,
        session: AsyncSession,
        product_id: str,
        user_id: UUID | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[int, int, bool]:
        """Record already-incurred GPU session usage; may drive balance negative.

        Records the full ``owed`` cost unconditionally — this is the *recording*
        primitive for costs already physically incurred (GPU minutes consumed).
        Balance may go negative; that is intentional and visible. Prevention of new
        work lives in ``check_and_reserve``, not here.

        Metered debits use ``job_id=None`` and link to the session via
        ``metadata['session_id']`` so that ``get_settled_tokens_for_session`` covers them
        while preserving the one-debit-per-job invariant on the base reservation.

        Returns:
            Tuple of (settled_tokens, new_balance, fully_settled) where:
            - settled_tokens: tokens debited (== owed; full incurred cost recorded)
            - new_balance: balance after the debit (may be negative)
            - fully_settled: always True when owed > 0 (full cost recorded)
        """
        if owed <= 0:
            repo = BillingRepository(session)
            balance = await repo.get_balance(account_id)
            return 0, balance, True

        repo = BillingRepository(session)

        # Lock account row
        account = await repo.get_account_for_update(account_id)
        if account is None:
            raise AccountNotFoundError(f"Token account {account_id} not found")

        balance = await repo.get_balance(account_id)
        new_balance = balance - owed  # may be negative — recording, not preventing

        base_metadata: dict[str, Any] = {
            "type": "gpu_session_metered",
            "session_id": str(session_id),
            "model_type": model_type,
        }
        if extra_metadata:
            base_metadata.update(extra_metadata)
        description = (
            extra_metadata.get("description", f"GPU session metered usage ({model_type})")
            if extra_metadata
            else f"GPU session metered usage ({model_type})"
        )
        await repo.create_transaction(
            id=new_id(),
            account_id=account_id,
            transaction_type=TransactionType.DEBIT.value,
            amount=-owed,
            balance_after=new_balance,
            job_id=None,
            description=description,
            metadata=base_metadata,
            product_id=product_id,
        )

        logger.info(
            "billing.gpu_session_metered",
            account_id=str(account_id),
            session_id=str(session_id),
            owed=owed,
            settled=owed,
            balance_after=new_balance,
        )

        if new_balance < 0:
            logger.warning(
                "billing.balance_negative",
                account_id=str(account_id),
                session_id=str(session_id),
                balance=new_balance,
                incurred=owed,
            )

        await self._publish_balance_update(
            user_ids=[user_id] if user_id is not None else [],
            account_id=account_id,
            balance=new_balance,
            delta=-owed,
            transaction_type=TransactionType.DEBIT.value,
        )

        return owed, new_balance, True

    async def refund(
        self,
        job_id: UUID,
        *,
        description: str,
        session: AsyncSession,
        product_id: str,
        user_id: UUID | None = None,
    ) -> TokenTransaction:
        """Create a refund (positive) transaction linked to job_id.

        Raises:
            RefundNotEligibleError: If no debit found or already refunded.
        """
        repo = BillingRepository(session)

        # Find original debit
        debit = await repo.get_debit_for_job(job_id, for_update=True)
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
            id=new_id(),
            account_id=debit.account_id,
            transaction_type=TransactionType.REFUND.value,
            amount=refund_amount,
            balance_after=new_balance,
            job_id=job_id,
            description=description,
            product_id=product_id,
        )

        logger.info(
            "billing.credit_processed",
            account_id=str(debit.account_id),
            job_id=str(job_id),
            amount=refund_amount,
            balance_after=new_balance,
            reason=description,
        )

        await self._publish_balance_update(
            user_ids=[user_id] if user_id is not None else [],
            account_id=debit.account_id,
            balance=new_balance,
            delta=refund_amount,
            transaction_type=TransactionType.REFUND.value,
        )

        return txn

    async def partial_refund(
        self,
        job_id: UUID,
        amount: int,
        *,
        description: str,
        session: AsyncSession,
        product_id: str,
        user_id: UUID | None = None,
    ) -> TokenTransaction:
        """Create a partial refund for a variable-cost resource.

        Unlike refund(), which refunds the full original debit, this creates
        a REFUND transaction for exactly ``amount`` tokens. Used for GPU sessions
        where the user may have been overcharged relative to actual usage.

        Cumulative invariant: the sum of all partial refunds for a given job
        must never exceed the original debit. This is enforced by querying
        existing refunds on the same job_id and rejecting requests that would
        overflow.

        Raises:
            RefundNotEligibleError: If no debit found, ``amount`` is <= 0, or
                ``already_refunded + amount > original_amount``.
        """
        if amount <= 0:
            raise RefundNotEligibleError(f"Partial refund amount must be positive, got {amount}")

        repo = BillingRepository(session)

        # Lock the debit row so concurrent partial-refund callers serialize.
        # Without this, two concurrent refunds can both pass the cumulative
        # invariant check (sum_refunds + amount <= original) and both commit,
        # producing a cumulative over-refund.
        debit = await repo.get_debit_for_job(job_id, for_update=True)
        if debit is None:
            raise RefundNotEligibleError(f"No debit transaction found for job {job_id}")

        original_amount = abs(debit.amount)
        already_refunded = await repo.sum_refunds_for_job(job_id)
        if already_refunded + amount > original_amount:
            raise RefundNotEligibleError(
                f"Partial refund would exceed original debit for job {job_id}: "
                f"already_refunded={already_refunded}, requested={amount}, "
                f"original={original_amount}"
            )

        account = await repo.get_account_for_update(debit.account_id)
        if account is None:
            raise RefundNotEligibleError("Account not found for partial refund")

        balance = await repo.get_balance(debit.account_id)
        new_balance = balance + amount

        txn = await repo.create_transaction(
            id=new_id(),
            account_id=debit.account_id,
            transaction_type=TransactionType.REFUND.value,
            amount=amount,
            balance_after=new_balance,
            job_id=job_id,
            description=description,
            product_id=product_id,
        )

        logger.info(
            "billing.partial_refund_processed",
            account_id=str(debit.account_id),
            job_id=str(job_id),
            amount=amount,
            balance_after=new_balance,
            already_refunded_before=already_refunded,
            original_debit=original_amount,
            reason=description,
        )

        await self._publish_balance_update(
            user_ids=[user_id] if user_id is not None else [],
            account_id=debit.account_id,
            balance=new_balance,
            delta=amount,
            transaction_type=TransactionType.REFUND.value,
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
        product_id: str,
        user_id: UUID | None = None,
    ) -> TokenTransaction:
        """Credit tokens to an account from a payment."""
        repo = BillingRepository(session)

        account = await repo.get_account_for_update(account_id)
        if account is None:
            raise AccountNotFoundError(f"Token account {account_id} not found")

        balance = await repo.get_balance(account_id)
        new_balance = balance + amount

        txn = await repo.create_transaction(
            id=new_id(),
            account_id=account_id,
            transaction_type=TransactionType.CREDIT.value,
            amount=amount,
            balance_after=new_balance,
            payment_id=payment_id,
            description=description,
            product_id=product_id,
        )

        logger.info(
            "billing.credit_processed",
            account_id=str(account_id),
            payment_id=str(payment_id),
            amount=amount,
            balance_after=new_balance,
            payment_provider=payment_provider,
        )

        await self._publish_balance_update(
            user_ids=[user_id] if user_id is not None else [],
            account_id=account_id,
            balance=new_balance,
            delta=amount,
            transaction_type=TransactionType.CREDIT.value,
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
        product_id: str,
    ) -> TokenTransaction:
        """Admin adjustment: positive = credit, negative = debit.

        For negative adjustments, checks that result >= 0.

        Resolves the owning user(s) from the locked account row and publishes
        a ``balance.updated`` SSE event to each:
        - Personal account → ``account.user_id``
        - Enterprise account → all organisation members
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
            id=new_id(),
            account_id=account_id,
            transaction_type=TransactionType.ADMIN_ADJUSTMENT.value,
            amount=amount,
            balance_after=new_balance,
            description=description,
            created_by=admin_id,
            product_id=product_id,
        )

        logger.info(
            "billing.balance_updated",
            account_id=str(account_id),
            admin_id=str(admin_id),
            amount=amount,
            balance_after=new_balance,
            description=description,
        )

        # Resolve SSE target user(s) from the account we already hold
        target_user_ids: list[UUID] = []
        if account.user_id is not None:
            # Personal account — single owner
            target_user_ids = [account.user_id]
        elif account.organization_id is not None:
            # Enterprise account — notify all org members
            target_user_ids = await repo.get_member_user_ids(account.organization_id)

        await self._publish_balance_update(
            user_ids=target_user_ids,
            account_id=account_id,
            balance=new_balance,
            delta=amount,
            transaction_type=TransactionType.ADMIN_ADJUSTMENT.value,
        )

        return txn

    async def get_transaction_history(
        self,
        account_id: UUID,
        *,
        limit: int = 50,
        transaction_type: str | None = None,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
        session: AsyncSession,
    ) -> Sequence[TokenTransaction]:
        """Returns transactions ordered by created_at DESC.

        Uses limit+1 fetch pattern — caller checks ``len(result) > limit``
        to determine ``has_more``.
        """
        repo = BillingRepository(session)
        return await repo.get_transaction_history(
            account_id,
            limit=limit,
            transaction_type=transaction_type,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
        )
