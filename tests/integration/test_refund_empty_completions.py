"""Integration tests for the empty-completion fix: the poller's failure settlement and
the refund_empty_completions backfill with its repository guard.

Runs against a real Postgres session: both are a money movement paired with a status
change, so the NOT EXISTS predicate, the per-job transaction boundary, the ledger
arithmetic and the persisted (redacted) error text must be exercised for real.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select

from src.api.services.billing import BalanceEvent, BillingService
from src.api.services.generation.aisha_failures import AishaFailure
from src.api.services.job_state_transition import JobStateTransitionService
from src.cli.refund_empty_completions import RefundReport, _Outcome, _settle_one, run_refund
from src.core.enums import JobStatus, TransactionType
from src.core.uid import new_id
from src.db.models.billing import TokenAccount, TokenTransaction
from src.db.models.storage import GenerationJob, GenerationOutput
from src.db.repositories.billing import BillingRepository
from src.db.repositories.job import JobRepository
from src.workers.aisha_job_poller import AishaJobPoller, AishaPollerConfig

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.user import User
    from tests.integration.conftest import (
        GpuSessionFactory,
        JobFactory,
        TokenAccountFactory,
        UserFactory,
    )

pytestmark = pytest.mark.asyncio

_DEBIT = 20
_STARTING_BALANCE = 100


@pytest.fixture
def product_id() -> str:
    """An isolated product so whole-table selection cannot see other tests' rows."""
    return f"ec-{uuid4().hex[:21]}"


class _SpyEventBus:
    """Records ``publish_balance`` calls without touching Redis."""

    def __init__(self) -> None:
        self.published: list[BalanceEvent | None] = []

    async def publish_balance(self, event: BalanceEvent | None) -> None:
        self.published.append(event)


class _FailingRefundBilling(BillingService):
    """A BillingService whose refund raises for one job — a 'bad row'."""

    def __init__(self, bad_job_id: UUID) -> None:
        super().__init__()
        self._bad_job_id = bad_job_id

    async def refund(self, job_id: UUID, **kwargs):  # type: ignore[no-untyped-def,override]  # noqa: ANN003
        if job_id == self._bad_job_id:
            raise RuntimeError("ledger exploded")
        return await super().refund(job_id, **kwargs)


class _Seed:
    """Builds the rows one scenario needs, all inside the test's SAVEPOINT session."""

    def __init__(
        self,
        session: AsyncSession,
        product_id: str,
        make_user: UserFactory,
        make_token_account: TokenAccountFactory,
        make_job: JobFactory,
        make_gpu_session: GpuSessionFactory,
    ) -> None:
        self.session = session
        self.product_id = product_id
        self._make_user = make_user
        self._make_account = make_token_account
        self._make_job = make_job
        self._make_gpu_session = make_gpu_session
        self.user: User | None = None
        self.account: TokenAccount | None = None
        # Plain ids survive the rollbacks the code under test performs, which expire
        # ORM instances (and a lazy refresh outside the greenlet would raise).
        self.account_id: UUID | None = None
        self.gpu_session_id: UUID | None = None

    async def account_for_user(self) -> tuple[User, TokenAccount]:
        """One user + funded personal account, shared across a scenario's jobs."""
        if self.user is None or self.account is None:
            self.user = await self._make_user(
                email=f"ec-{uuid4().hex[:12]}@example.com", product_id=self.product_id
            )
            self.account = await self._make_account(user=self.user, product_id=self.product_id)
            self.account_id = self.account.id
            await BillingRepository(self.session).create_transaction(
                id=new_id(),
                account_id=self.account.id,
                transaction_type=TransactionType.CREDIT.value,
                amount=_STARTING_BALANCE,
                balance_after=_STARTING_BALANCE,
                product_id=self.product_id,
                description="seed",
            )
            # ck_generation_jobs_aisha_has_session: every Aisha job carries a session.
            gpu_session = await self._make_gpu_session(
                user=self.user, status="stopped", product_id=self.product_id
            )
            self.gpu_session_id = gpu_session.id
        return self.user, self.account

    async def job(
        self,
        *,
        status: str = JobStatus.COMPLETED.value,
        provider: str = "aisha",
        is_deleted: bool = False,
        debit: bool = True,
        with_output: bool = False,
        created_at: datetime | None = None,
    ) -> GenerationJob:
        user, account = await self.account_for_user()
        job = await self._make_job(
            user=user,
            status=status,
            provider=provider,
            model="aisha-image",
            product_id=self.product_id,
            is_deleted=is_deleted,
            gpu_session_id=self.gpu_session_id,
        )
        if created_at is not None:
            job.created_at = created_at
            await self.session.flush()
        if debit:
            repo = BillingRepository(self.session)
            balance = await repo.get_balance(account.id)
            txn = await repo.create_transaction(
                id=new_id(),
                account_id=account.id,
                transaction_type=TransactionType.DEBIT.value,
                amount=-_DEBIT,
                balance_after=balance - _DEBIT,
                job_id=job.id,
                product_id=self.product_id,
                description="Generation charge",
            )
            # What check_and_reserve records; transition_to_failed keys its refund off it.
            job.token_cost = _DEBIT
            job.debit_transaction_id = txn.id
            await self.session.flush()
        if with_output:
            self.session.add(
                GenerationOutput(
                    id=uuid4(),
                    user_id=user.id,
                    job_id=job.id,
                    product_id=self.product_id,
                    storage_key=f"users/{user.id}/outputs/{job.id}/{uuid4()}.png",
                    content_type="image/png",
                    size_bytes=1024,
                    format="png",
                    output_index=0,
                    expires_at=datetime.now(UTC) + timedelta(days=7),
                )
            )
            await self.session.flush()
        return job

    async def balance(self) -> int:
        assert self.account_id is not None
        return await BillingRepository(self.session).get_balance(self.account_id)


@pytest.fixture
def seed(
    db_session: AsyncSession,
    product_id: str,
    make_user: UserFactory,
    make_token_account: TokenAccountFactory,
    make_job: JobFactory,
    make_gpu_session: GpuSessionFactory,
) -> _Seed:
    return _Seed(db_session, product_id, make_user, make_token_account, make_job, make_gpu_session)


async def _reload(session: AsyncSession, job: GenerationJob) -> GenerationJob:
    await session.refresh(job)
    return job


async def _refund_count(session: AsyncSession, job_id: UUID) -> int:
    return int(
        (
            await session.execute(
                select(func.count(TokenTransaction.id)).where(
                    TokenTransaction.job_id == job_id,
                    TokenTransaction.transaction_type == TransactionType.REFUND.value,
                )
            )
        ).scalar_one()
    )


async def _run(
    session: AsyncSession, product_id: str, *, apply: bool, **kwargs: object
) -> RefundReport:
    # The CLI runs against committed data; commit the seed the same way. Its
    # per-job rollbacks would otherwise discard the still-pending seed rows.
    await session.commit()
    return await run_refund(
        session,
        billing=BillingService(),
        product=product_id,
        apply=apply,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# The backfill command
# ---------------------------------------------------------------------------


class TestBackfill:
    async def test_dry_run_lists_the_empty_completions_and_changes_nothing(
        self, db_session: AsyncSession, seed: _Seed, product_id: str
    ) -> None:
        empty_a = await seed.job()
        empty_b = await seed.job()
        with_output = await seed.job(with_output=True)
        balance_before = await seed.balance()

        report = await _run(db_session, product_id, apply=False)

        assert {r.job_id for r in report.found} == {empty_a.id, empty_b.id}
        assert with_output.id not in {r.job_id for r in report.found}
        assert report.total_debit_tokens == 2 * _DEBIT
        assert all(r.debit_amount == _DEBIT for r in report.found)
        assert all(r.account_id == seed.account_id for r in report.found)
        assert report.refunded == 0

        for job in (empty_a, empty_b, with_output):
            assert (await _reload(db_session, job)).status == JobStatus.COMPLETED
            assert await _refund_count(db_session, job.id) == 0
        assert await seed.balance() == balance_before

    async def test_apply_refunds_both_and_marks_them_failed_leaving_the_third(
        self, db_session: AsyncSession, seed: _Seed, product_id: str
    ) -> None:
        empty_a = await seed.job()
        empty_b = await seed.job()
        with_output = await seed.job(with_output=True)
        balance_before = await seed.balance()
        bus = _SpyEventBus()

        report = await _run(db_session, product_id, apply=True, event_bus=bus)

        assert report.refunded == 2
        assert report.already_refunded == 0
        assert report.skipped == []
        for job in (empty_a, empty_b):
            reloaded = await _reload(db_session, job)
            assert reloaded.status == JobStatus.FAILED
            assert reloaded.failure_code == AishaFailure.PROVIDER_EXECUTION_FAILED.value
            assert reloaded.public_error_message == (
                AishaFailure.PROVIDER_EXECUTION_FAILED.public_message
            )
            assert reloaded.error_message is not None
            assert await _refund_count(db_session, job.id) == 1
        untouched = await _reload(db_session, with_output)
        assert untouched.status == JobStatus.COMPLETED
        assert untouched.failure_code is None
        assert await _refund_count(db_session, with_output.id) == 0
        assert await seed.balance() == balance_before + 2 * _DEBIT
        # One live balance event per refund, published after each commit.
        assert len(bus.published) == 2
        assert all(e is not None for e in bus.published)

    async def test_second_apply_is_a_no_op(
        self, db_session: AsyncSession, seed: _Seed, product_id: str
    ) -> None:
        await seed.job()
        await seed.job()
        first = await _run(db_session, product_id, apply=True)
        assert first.refunded == 2
        balance_after_first = await seed.balance()
        transactions_after_first = int(
            (await db_session.execute(select(func.count(TokenTransaction.id)))).scalar_one()
        )

        second = await _run(db_session, product_id, apply=True)

        assert second.found == []
        assert second.refunded == 0
        assert await seed.balance() == balance_after_first
        assert (
            int((await db_session.execute(select(func.count(TokenTransaction.id)))).scalar_one())
            == transactions_after_first
        )

    async def test_already_refunded_job_gets_its_status_corrected_without_a_second_refund(
        self, db_session: AsyncSession, seed: _Seed, product_id: str
    ) -> None:
        job = await seed.job()
        # e.g. an operator already compensated the user by hand.
        await BillingService().refund(
            job.id, description="manual", session=db_session, product_id=product_id
        )
        balance_before = await seed.balance()
        bus = _SpyEventBus()

        report = await _run(db_session, product_id, apply=True, event_bus=bus)

        assert report.already_refunded == 1
        assert report.refunded == 0
        assert report.skipped == []
        assert (await _reload(db_session, job)).status == JobStatus.FAILED
        assert await _refund_count(db_session, job.id) == 1  # still just the manual one
        assert await seed.balance() == balance_before
        assert bus.published == []  # nothing new to tell the user

    async def test_a_bad_row_does_not_roll_back_the_others(
        self, db_session: AsyncSession, seed: _Seed, product_id: str
    ) -> None:
        good_before = await seed.job(created_at=datetime.now(UTC) - timedelta(hours=3))
        bad = await seed.job(created_at=datetime.now(UTC) - timedelta(hours=2))
        good_after = await seed.job(created_at=datetime.now(UTC) - timedelta(hours=1))
        await db_session.commit()
        bad_id = bad.id  # a rollback expires ORM instances; read the id first

        report = await run_refund(
            db_session,
            billing=_FailingRefundBilling(bad_id),
            product=product_id,
            apply=True,
        )

        assert report.refunded == 2
        assert [s.job_id for s in report.skipped] == [bad_id]
        assert (await _reload(db_session, good_before)).status == JobStatus.FAILED
        assert (await _reload(db_session, good_after)).status == JobStatus.FAILED
        # The failed job's status correction was rolled back with its refund.
        assert (await _reload(db_session, bad)).status == JobStatus.COMPLETED
        assert await _refund_count(db_session, bad_id) == 0

    async def test_job_without_a_debit_is_skipped_and_left_untouched(
        self, db_session: AsyncSession, seed: _Seed, product_id: str
    ) -> None:
        no_debit = await seed.job(debit=False)
        no_debit_id = no_debit.id  # a rollback expires ORM instances; read the id first

        report = await _run(db_session, product_id, apply=True)

        assert [r.debit_amount for r in report.found] == [None]
        assert report.refunded == 0
        assert [s.job_id for s in report.skipped] == [no_debit_id]
        assert "no_debit_found" in report.skipped[0].detail
        assert (await _reload(db_session, no_debit)).status == JobStatus.COMPLETED

    async def test_only_aisha_completed_live_jobs_are_selected(
        self, db_session: AsyncSession, seed: _Seed, product_id: str
    ) -> None:
        target = await seed.job()
        await seed.job(provider="grok")
        await seed.job(status=JobStatus.FAILED.value)
        await seed.job(status=JobStatus.RUNNING.value)
        # The retention sweeper deletes expired outputs and soft-deletes the job:
        # a completed job with no outputs that is the *normal* end of its life.
        await seed.job(is_deleted=True)

        report = await _run(db_session, product_id, apply=False)

        assert [r.job_id for r in report.found] == [target.id]

    async def test_a_job_that_gained_outputs_after_selection_is_not_refunded(
        self, db_session: AsyncSession, seed: _Seed, product_id: str
    ) -> None:
        """The predicate guard is re-checked at write time, not trusted from selection."""
        job = await seed.job()
        await db_session.commit()
        [row] = await JobRepository(db_session).list_empty_completions(product_id=product_id)
        balance_before = await seed.balance()
        # Between selection and settlement the job acquires an output.
        db_session.add(
            GenerationOutput(
                id=uuid4(),
                user_id=row.user_id,
                job_id=row.job_id,
                product_id=product_id,
                storage_key=f"users/{row.user_id}/outputs/{row.job_id}/late.png",
                content_type="image/png",
                size_bytes=1,
                format="png",
                output_index=0,
                expires_at=datetime.now(UTC) + timedelta(days=7),
            )
        )
        await db_session.commit()

        outcome, event, _ = await _settle_one(db_session, BillingService(), row)

        assert outcome is _Outcome.NO_LONGER_EMPTY
        assert event is None
        assert (await _reload(db_session, job)).status == JobStatus.COMPLETED
        assert await _refund_count(db_session, row.job_id) == 0
        assert await seed.balance() == balance_before

    async def test_job_id_restricts_to_a_reviewed_subset(
        self, db_session: AsyncSession, seed: _Seed, product_id: str
    ) -> None:
        reviewed = await seed.job()
        other = await seed.job()

        report = await _run(db_session, product_id, apply=True, job_ids=[reviewed.id])

        assert report.refunded == 1
        assert (await _reload(db_session, reviewed)).status == JobStatus.FAILED
        assert (await _reload(db_session, other)).status == JobStatus.COMPLETED

    async def test_explicit_ids_process_valid_jobs_and_report_jobs_with_outputs(
        self, db_session: AsyncSession, seed: _Seed, product_id: str
    ) -> None:
        """A reviewed log-id list is safe even if one candidate is now stale."""
        empty_a = await seed.job()
        empty_b = await seed.job()
        has_output = await seed.job(with_output=True)

        report = await _run(
            db_session,
            product_id,
            apply=True,
            job_ids=[empty_a.id, has_output.id, empty_b.id],
        )

        assert report.refunded == 2
        assert [(skip.job_id, skip.outcome) for skip in report.skipped] == [
            (has_output.id, _Outcome.NO_LONGER_EMPTY)
        ]
        assert (await _reload(db_session, empty_a)).status == JobStatus.FAILED
        assert (await _reload(db_session, empty_b)).status == JobStatus.FAILED
        assert (await _reload(db_session, has_output)).status == JobStatus.COMPLETED

    async def test_since_and_limit_narrow_the_selection(
        self, db_session: AsyncSession, seed: _Seed, product_id: str
    ) -> None:
        now = datetime.now(UTC)
        await seed.job(created_at=now - timedelta(days=10))
        recent_a = await seed.job(created_at=now - timedelta(days=2))
        await seed.job(created_at=now - timedelta(days=1))

        since = await _run(db_session, product_id, apply=False, since=now - timedelta(days=5))
        assert len(since.found) == 2

        limited = await _run(
            db_session, product_id, apply=False, since=now - timedelta(days=5), limit=1
        )
        assert [r.job_id for r in limited.found] == [recent_a.id]  # oldest first

    async def test_other_products_are_not_touched(
        self,
        db_session: AsyncSession,
        seed: _Seed,
        product_id: str,
        make_user: UserFactory,
        make_token_account: TokenAccountFactory,
        make_job: JobFactory,
        make_gpu_session: GpuSessionFactory,
    ) -> None:
        mine = await seed.job()
        other_seed = _Seed(
            db_session,
            f"ec-{uuid4().hex[:21]}",
            make_user,
            make_token_account,
            make_job,
            make_gpu_session,
        )
        theirs = await other_seed.job()

        report = await _run(db_session, product_id, apply=True)

        assert [r.job_id for r in report.found] == [mine.id]
        assert (await _reload(db_session, mine)).status == JobStatus.FAILED
        assert (await _reload(db_session, theirs)).status == JobStatus.COMPLETED


# ---------------------------------------------------------------------------
# The predicate-guarded repository method (D6)
# ---------------------------------------------------------------------------


class TestMarkEmptyCompletionFailed:
    async def test_job_with_outputs_is_refused_and_left_unchanged(
        self, db_session: AsyncSession, seed: _Seed
    ) -> None:
        job = await seed.job(with_output=True)

        changed = await JobRepository(db_session).mark_empty_completion_failed(
            job.id, failure_code="provider_execution_failed", error_message="x"
        )

        assert changed is False
        reloaded = await _reload(db_session, job)
        assert reloaded.status == JobStatus.COMPLETED
        assert reloaded.failure_code is None
        assert reloaded.error_message is None

    async def test_empty_completion_becomes_failed(
        self, db_session: AsyncSession, seed: _Seed
    ) -> None:
        job = await seed.job()
        job.completed_at = datetime.now(UTC)
        await db_session.flush()

        changed = await JobRepository(db_session).mark_empty_completion_failed(
            job.id,
            failure_code="provider_execution_failed",
            error_message="backfill",
            public_error_message="The generation engine could not complete the request.",
        )

        assert changed is True
        reloaded = await _reload(db_session, job)
        assert reloaded.status == JobStatus.FAILED
        assert reloaded.failure_code == "provider_execution_failed"
        assert reloaded.error_message == "backfill"
        assert reloaded.public_error_message == (
            "The generation engine could not complete the request."
        )
        assert reloaded.completed_at is None

    @pytest.mark.parametrize(
        "status",
        [JobStatus.RUNNING, JobStatus.QUEUED, JobStatus.FAILED],
    )
    async def test_only_a_completed_job_can_be_corrected(
        self, db_session: AsyncSession, seed: _Seed, status: JobStatus
    ) -> None:
        job = await seed.job(status=status.value)

        changed = await JobRepository(db_session).mark_empty_completion_failed(
            job.id, failure_code="provider_execution_failed", error_message="x"
        )

        assert changed is False
        assert (await _reload(db_session, job)).status == status

    async def test_unknown_job_returns_false(self, db_session: AsyncSession) -> None:
        assert (
            await JobRepository(db_session).mark_empty_completion_failed(
                uuid4(), failure_code="provider_execution_failed", error_message="x"
            )
            is False
        )


# ---------------------------------------------------------------------------
# The poller's own settlement of a ComfyUI execution error
# ---------------------------------------------------------------------------

_TRACEBACK_MARKER = "TRACEBACK_MARKER_execution.py"


class TestPollerSettlesExecutionErrors:
    async def test_execution_error_fails_and_refunds_once_across_two_ticks(
        self, db_session: AsyncSession, seed: _Seed, product_id: str
    ) -> None:
        job = await seed.job(status=JobStatus.RUNNING.value)
        await db_session.commit()
        balance_before = await seed.balance()
        poller = AishaJobPoller(
            session_factory=MagicMock(),
            event_bus=None,
            billing_service=BillingService(),
            r2_storage=None,
            config=AishaPollerConfig(tunnel_allowed_suffix="gpu.test"),
            redis_client_factory=MagicMock(),
        )
        ts = JobStateTransitionService(
            session=db_session, event_bus=None, billing_service=BillingService()
        )
        history = {
            "outputs": {},
            "status": {
                "status_str": "error",
                "completed": False,
                "messages": [
                    [
                        "execution_error",
                        {
                            "node_type": "PatchFlashAttentionKJ",
                            "exception_type": "ImportError",
                            "exception_message": "no flash attention; fetch https://a/x?token=SECRET",
                            "traceback": [_TRACEBACK_MARKER],
                            "current_inputs": {"model": ["<ModelPatcher object>"]},
                        },
                    ],
                ],
            },
        }

        # The same failed history entry on two consecutive ticks.
        for _ in range(2):
            await poller._handle_history_complete(
                client=AsyncMock(),
                job=job,
                history_entry=history,
                product_id=product_id,
                ts=ts,
            )

        reloaded = await _reload(db_session, job)
        assert reloaded.status == JobStatus.FAILED
        assert reloaded.failure_code == AishaFailure.PROVIDER_EXECUTION_FAILED.value
        assert reloaded.public_error_message == (
            AishaFailure.PROVIDER_EXECUTION_FAILED.public_message
        )
        # Persisted detail: readable, redacted, and free of node state.
        assert reloaded.error_message is not None
        assert "PatchFlashAttentionKJ" in reloaded.error_message
        assert "ImportError" in reloaded.error_message
        assert "SECRET" not in reloaded.error_message
        assert _TRACEBACK_MARKER not in reloaded.error_message
        assert "ModelPatcher" not in reloaded.error_message
        # Exactly one refund; the user is made whole.
        assert await _refund_count(db_session, job.id) == 1
        assert await seed.balance() == balance_before + _DEBIT
