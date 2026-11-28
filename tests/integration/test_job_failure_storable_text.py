"""R2: provider text PostgreSQL cannot store never leaves a charged job in flight.

``transition_to_failed`` moves a job to ``failed`` and refunds it in one transaction. A provider's
failure text that holds U+0000, or a lone surrogate, made the guarded ``UPDATE`` raise: the job
stayed in flight and charged, and the poller met the same failure again on its next tick.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from src.api.services.billing import BillingService
from src.api.services.job_state_transition import JobStateTransitionService
from src.core.enums import JobStatus, TransactionType
from src.core.uid import new_id
from src.db.models.billing import TokenTransaction
from src.db.repositories.billing import BillingRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from tests.integration.conftest import JobFactory, TokenAccountFactory, UserFactory

_BALANCE = 100
_PRICE = 7


@pytest.mark.parametrize(
    ("text_from_provider", "stored"),
    [
        pytest.param("node crashed:\x00 bad tensor", "node crashed: bad tensor", id="nul"),
        pytest.param(
            "cannot open /models/\udcff.safetensors",
            "cannot open /models/?.safetensors",
            id="lone-surrogate",
        ),
    ],
)
async def test_failing_a_job_with_unstorable_provider_text_commits_and_refunds(
    db_session: AsyncSession,
    make_user: UserFactory,
    make_token_account: TokenAccountFactory,
    make_job: JobFactory,
    text_from_provider: str,
    stored: str,
) -> None:
    billing = BillingService()
    ledger = BillingRepository(db_session)
    user = await make_user(email=f"storable-{uuid4().hex}@example.com")
    account = await make_token_account(user=user)
    await ledger.create_transaction(
        id=new_id(),
        account_id=account.id,
        transaction_type=TransactionType.CREDIT.value,
        amount=_BALANCE,
        balance_after=_BALANCE,
        product_id="vex",
    )
    job = await make_job(user=user, status=JobStatus.RUNNING.value)
    reservation = await billing.check_and_reserve(
        account.id,
        _PRICE,
        job.id,
        metadata={"provider": "grok"},
        session=db_session,
        product_id="vex",
    )
    job.token_cost = _PRICE
    job.debit_transaction_id = reservation.txn.id
    await db_session.commit()
    assert await ledger.get_balance(account.id) == _BALANCE - _PRICE

    transitions = JobStateTransitionService(
        session=db_session, event_bus=None, billing_service=billing
    )
    failed, did_transition = await transitions.transition_to_failed(
        job.id,
        error_message=text_from_provider,
        public_error_message=text_from_provider,
        failure_code=text_from_provider,
        refund=True,
        product_id="vex",
    )

    assert did_transition is True
    await db_session.refresh(failed)  # what the database holds
    assert str(failed.status) == JobStatus.FAILED.value
    assert failed.error_message == stored
    assert failed.public_error_message == stored
    assert failed.failure_code == stored
    assert await ledger.get_balance(account.id) == _BALANCE
    refunds = await db_session.scalar(
        select(func.count(TokenTransaction.id)).where(
            TokenTransaction.job_id == job.id,
            TokenTransaction.transaction_type == TransactionType.REFUND.value,
        )
    )
    assert refunds == 1
