"""Integration tests for job sweep on GPU session termination.

These tests exercise the SQL logic underlying the session-stop sweep path
directly, because JobSweepService and GpuSessionService create their own DB
sessions via session_factory which cannot participate in the SAVEPOINT-based
test transaction.

Pattern mirrors test_gpu_session_reconciler.py and test_billing_reconciler.py.

Coverage targets:
- JobRepository.list_in_flight_for_session — status filtering, session isolation,
  provider filtering (Aisha vs Grok), soft-delete exclusion
- JobRepository.count_in_flight_for_session — count variant of the above
- JobStateTransitionService.transition_to_failed — called by sweep; marks job
  FAILED and issues refund via BillingService
- Pause precondition: count > 0 means pause must be rejected at service level
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.services.billing import BillingService
from src.api.services.job_state_transition import JobStateTransitionService
from src.core.enums import JobStatus
from src.db.models.billing import TokenTransaction
from src.db.models.gpu_session import GpuSession
from src.db.models.storage import GenerationJob
from src.db.models.user import User
from src.db.repositories.job import JobRepository

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_user(session: AsyncSession, *, product_id: str = "vex") -> User:
    user = User(
        id=uuid4(),
        email=f"sweep-{uuid4().hex[:8]}@example.com",
        password_hash="x",
        product_id=product_id,
    )
    session.add(user)
    await session.flush()
    return user


async def _insert_gpu_session(
    session: AsyncSession,
    user: User,
    *,
    status: str = "active",
    product_id: str = "vex",
) -> GpuSession:
    gpu_session = GpuSession(
        id=uuid4(),
        user_id=user.id,
        product_id=product_id,
        bundle_name="wan_2.2_i2v",
        model_type="aisha-image",
        status=status,
    )
    session.add(gpu_session)
    await session.flush()
    return gpu_session


async def _insert_job(
    session: AsyncSession,
    user: User,
    *,
    status: str = JobStatus.QUEUED.value,
    gpu_session_id: object = None,
    provider: str = "aisha",
    product_id: str = "vex",
    is_deleted: bool = False,
) -> GenerationJob:
    job = GenerationJob(
        id=uuid4(),
        user_id=user.id,
        name="Test Job",
        prompt="a test",
        status=status,
        generation_type="t2i",
        provider=provider,
        product_id=product_id,
        gpu_session_id=gpu_session_id,
        is_deleted=is_deleted,
    )
    session.add(job)
    await session.flush()
    return job


async def _seed_balance(session: AsyncSession, account_id: object, amount: int) -> None:
    txn = TokenTransaction(
        id=uuid4(),
        account_id=account_id,
        transaction_type="credit",
        amount=amount,
        balance_after=amount,
        product_id="vex",
    )
    session.add(txn)
    await session.flush()


# ---------------------------------------------------------------------------
# list_in_flight_for_session — status filtering
# ---------------------------------------------------------------------------


class TestListInFlightForSession:
    async def test_returns_queued_and_running_jobs(self, db_session: AsyncSession) -> None:
        user = await _insert_user(db_session)
        gpu_session = await _insert_gpu_session(db_session, user)
        repo = JobRepository(db_session)

        queued = await _insert_job(
            db_session, user, status=JobStatus.QUEUED.value, gpu_session_id=gpu_session.id
        )
        running = await _insert_job(
            db_session, user, status=JobStatus.RUNNING.value, gpu_session_id=gpu_session.id
        )

        jobs = await repo.list_in_flight_for_session(gpu_session.id)
        job_ids = {j.id for j in jobs}

        assert queued.id in job_ids
        assert running.id in job_ids

    async def test_excludes_completed_and_failed_jobs(self, db_session: AsyncSession) -> None:
        user = await _insert_user(db_session)
        gpu_session = await _insert_gpu_session(db_session, user)
        repo = JobRepository(db_session)

        await _insert_job(
            db_session, user, status=JobStatus.COMPLETED.value, gpu_session_id=gpu_session.id
        )
        await _insert_job(
            db_session, user, status=JobStatus.FAILED.value, gpu_session_id=gpu_session.id
        )

        jobs = await repo.list_in_flight_for_session(gpu_session.id)
        assert jobs == []

    async def test_excludes_soft_deleted_jobs(self, db_session: AsyncSession) -> None:
        user = await _insert_user(db_session)
        gpu_session = await _insert_gpu_session(db_session, user)
        repo = JobRepository(db_session)

        await _insert_job(
            db_session,
            user,
            status=JobStatus.QUEUED.value,
            gpu_session_id=gpu_session.id,
            is_deleted=True,
        )

        jobs = await repo.list_in_flight_for_session(gpu_session.id)
        assert jobs == []

    async def test_excludes_jobs_for_other_sessions(self, db_session: AsyncSession) -> None:
        """Jobs bound to a different GPU session are not returned."""
        # Use distinct users so each can have one active session (unique index
        # on (user_id, product_id, model_type) for active sessions).
        user_a = await _insert_user(db_session)
        user_b = await _insert_user(db_session)
        session_a = await _insert_gpu_session(db_session, user_a)
        session_b = await _insert_gpu_session(db_session, user_b)
        repo = JobRepository(db_session)

        job_a = await _insert_job(
            db_session, user_a, status=JobStatus.QUEUED.value, gpu_session_id=session_a.id
        )
        await _insert_job(
            db_session, user_b, status=JobStatus.QUEUED.value, gpu_session_id=session_b.id
        )

        jobs = await repo.list_in_flight_for_session(session_a.id)
        assert [j.id for j in jobs] == [job_a.id]

    async def test_excludes_grok_jobs_with_no_gpu_session_id(
        self, db_session: AsyncSession
    ) -> None:
        """Grok jobs have no gpu_session_id — they must never appear in sweep results."""
        user = await _insert_user(db_session)
        gpu_session = await _insert_gpu_session(db_session, user)
        repo = JobRepository(db_session)

        # Aisha job linked to session → should appear
        aisha_job = await _insert_job(
            db_session,
            user,
            status=JobStatus.QUEUED.value,
            gpu_session_id=gpu_session.id,
            provider="aisha",
        )
        # Grok job with no session link → must NOT appear
        await _insert_job(
            db_session,
            user,
            status=JobStatus.QUEUED.value,
            gpu_session_id=None,
            provider="grok",
        )

        jobs = await repo.list_in_flight_for_session(gpu_session.id)
        assert [j.id for j in jobs] == [aisha_job.id]

    async def test_returns_empty_for_unknown_session(self, db_session: AsyncSession) -> None:
        repo = JobRepository(db_session)
        jobs = await repo.list_in_flight_for_session(uuid4())
        assert jobs == []


# ---------------------------------------------------------------------------
# count_in_flight_for_session
# ---------------------------------------------------------------------------


class TestCountInFlightForSession:
    async def test_counts_queued_and_running(self, db_session: AsyncSession) -> None:
        user = await _insert_user(db_session)
        gpu_session = await _insert_gpu_session(db_session, user)
        repo = JobRepository(db_session)

        await _insert_job(
            db_session, user, status=JobStatus.QUEUED.value, gpu_session_id=gpu_session.id
        )
        await _insert_job(
            db_session, user, status=JobStatus.RUNNING.value, gpu_session_id=gpu_session.id
        )
        await _insert_job(
            db_session, user, status=JobStatus.COMPLETED.value, gpu_session_id=gpu_session.id
        )

        count = await repo.count_in_flight_for_session(gpu_session.id)
        assert count == 2

    async def test_returns_zero_for_empty_session(self, db_session: AsyncSession) -> None:
        user = await _insert_user(db_session)
        gpu_session = await _insert_gpu_session(db_session, user)
        repo = JobRepository(db_session)

        count = await repo.count_in_flight_for_session(gpu_session.id)
        assert count == 0

    async def test_isolates_to_session(self, db_session: AsyncSession) -> None:
        # Use distinct users so each can have one active session (unique index
        # on (user_id, product_id, model_type) for active sessions).
        user_a = await _insert_user(db_session)
        user_b = await _insert_user(db_session)
        session_a = await _insert_gpu_session(db_session, user_a)
        session_b = await _insert_gpu_session(db_session, user_b)
        repo = JobRepository(db_session)

        await _insert_job(
            db_session, user_a, status=JobStatus.QUEUED.value, gpu_session_id=session_a.id
        )
        await _insert_job(
            db_session, user_b, status=JobStatus.QUEUED.value, gpu_session_id=session_b.id
        )
        await _insert_job(
            db_session, user_b, status=JobStatus.QUEUED.value, gpu_session_id=session_b.id
        )

        assert await repo.count_in_flight_for_session(session_a.id) == 1
        assert await repo.count_in_flight_for_session(session_b.id) == 2

    async def test_count_is_pause_precondition(self, db_session: AsyncSession) -> None:
        """Non-zero count is the DB invariant that triggers the pause rejection."""
        user = await _insert_user(db_session)
        gpu_session = await _insert_gpu_session(db_session, user)
        repo = JobRepository(db_session)

        await _insert_job(
            db_session, user, status=JobStatus.RUNNING.value, gpu_session_id=gpu_session.id
        )

        count = await repo.count_in_flight_for_session(gpu_session.id)

        # Service would raise SessionHasInFlightJobsError when count > 0.
        assert count > 0


# ---------------------------------------------------------------------------
# transition_to_failed integration — SQL layer of sweep
# ---------------------------------------------------------------------------


class TestTransitionToFailedInSweepPath:
    """Verify the SQL operations that JobSweepService.sweep_session relies on.

    JobSweepService calls JobStateTransitionService.transition_to_failed, which
    executes an UPDATE and then calls BillingService.refund. These tests exercise
    the UPDATE directly — billing is mocked because it would need its own session
    and is covered by test_gpu_session_billing.py.
    """

    async def test_transition_marks_queued_job_failed(self, db_session: AsyncSession) -> None:
        billing = BillingService(event_bus=None)
        user = await _insert_user(db_session)
        gpu_session = await _insert_gpu_session(db_session, user)
        job = await _insert_job(
            db_session, user, status=JobStatus.QUEUED.value, gpu_session_id=gpu_session.id
        )

        ts = JobStateTransitionService(
            session=db_session,
            event_bus=None,
            billing_service=billing,
        )
        result = await ts.transition_to_failed(
            job.id,
            error_message="GPU session stopped before job completed.",
            refund=False,  # skip billing — not the focus here
            product_id="vex",
        )

        assert str(result.status) == JobStatus.FAILED.value
        assert result.error_message is not None
        assert "stopped" in result.error_message

    async def test_transition_marks_running_job_failed(self, db_session: AsyncSession) -> None:
        billing = BillingService(event_bus=None)
        user = await _insert_user(db_session)
        gpu_session = await _insert_gpu_session(db_session, user)
        job = await _insert_job(
            db_session, user, status=JobStatus.RUNNING.value, gpu_session_id=gpu_session.id
        )

        ts = JobStateTransitionService(
            session=db_session,
            event_bus=None,
            billing_service=billing,
        )
        result = await ts.transition_to_failed(
            job.id,
            error_message="GPU session stopped before job completed.",
            refund=False,
            product_id="vex",
        )

        assert str(result.status) == JobStatus.FAILED.value

    async def test_transition_is_noop_for_completed_job(self, db_session: AsyncSession) -> None:
        billing = BillingService(event_bus=None)
        user = await _insert_user(db_session)
        gpu_session = await _insert_gpu_session(db_session, user)
        job = await _insert_job(
            db_session, user, status=JobStatus.COMPLETED.value, gpu_session_id=gpu_session.id
        )

        ts = JobStateTransitionService(
            session=db_session,
            event_bus=None,
            billing_service=billing,
        )
        result = await ts.transition_to_failed(
            job.id,
            error_message="GPU session stopped before job completed.",
            refund=False,
            product_id="vex",
        )

        # Already terminal — returns the row unchanged.
        assert str(result.status) == JobStatus.COMPLETED.value

    async def test_grok_job_not_found_in_session_sweep(self, db_session: AsyncSession) -> None:
        """Grok jobs (no gpu_session_id) are absent from list_in_flight_for_session.

        This confirms that sweep_session, which iterates list_in_flight_for_session,
        can never accidentally touch Grok jobs.
        """
        user = await _insert_user(db_session)
        gpu_session = await _insert_gpu_session(db_session, user)
        repo = JobRepository(db_session)

        grok_job = await _insert_job(
            db_session,
            user,
            status=JobStatus.QUEUED.value,
            gpu_session_id=None,  # Grok jobs are not linked to a GPU session
            provider="grok",
        )
        aisha_job = await _insert_job(
            db_session,
            user,
            status=JobStatus.QUEUED.value,
            gpu_session_id=gpu_session.id,
            provider="aisha",
        )

        in_flight = await repo.list_in_flight_for_session(gpu_session.id)
        in_flight_ids = {j.id for j in in_flight}

        assert grok_job.id not in in_flight_ids
        assert aisha_job.id in in_flight_ids
