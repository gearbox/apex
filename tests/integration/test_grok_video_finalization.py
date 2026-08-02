"""PostgreSQL race tests for durable Grok video finalization and settlement.

These deliberately use independent sessions instead of the normal SAVEPOINT
fixture.  The finalization lease, row-level refund lock, and conditional state
updates only have their production semantics across separate connections.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.api.services.billing import BillingService
from src.api.services.grok import GrokVideoResult
from src.api.services.grok.job_service import (
    GrokJobService,
    _MaterializedVideo,
)
from src.api.services.job_state_transition import (
    GenerationOutputData,
    JobStateTransitionService,
)
from src.core.enums import GenerationType, JobStatus, Provider, TransactionType
from src.core.uid import new_id
from src.db.models.billing import TokenAccount, TokenTransaction
from src.db.models.storage import GenerationJob, GenerationOutput
from src.db.models.user import User

pytestmark = pytest.mark.asyncio


@dataclass(frozen=True)
class _Seed:
    user_id: UUID
    account_id: UUID | None
    job_id: UUID


async def _seed_video_job(
    engine: AsyncEngine,
    *,
    with_debit: bool = False,
    created_at: datetime | None = None,
    finalization_claim_token: str | None = None,
    finalization_lease_expires_at: datetime | None = None,
) -> _Seed:
    """Commit one minimal Grok video job, optionally with a debit to refund."""
    user = User(
        id=new_id(),
        email=f"grok-finalization-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id="vex",
        is_active=True,
    )
    job = GenerationJob(
        id=new_id(),
        user_id=user.id,
        name="Grok video",
        prompt="a test video",
        generation_type=GenerationType.T2V.value,
        status=JobStatus.RUNNING.value,
        provider=Provider.GROK.value,
        model="grok-imagine-video",
        product_id="vex",
        external_request_id="video-request-id",
        created_at=created_at or datetime.now(UTC),
        finalization_claim_token=finalization_claim_token,
        finalization_lease_expires_at=finalization_lease_expires_at,
    )

    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add(user)
        await session.flush()
        session.add(job)

        account_id: UUID | None = None
        if with_debit:
            account = TokenAccount(
                id=new_id(),
                user_id=user.id,
                account_type="personal",
                product_id="vex",
            )
            credit = TokenTransaction(
                id=new_id(),
                account_id=account.id,
                transaction_type=TransactionType.CREDIT.value,
                amount=100,
                balance_after=100,
                product_id="vex",
            )
            session.add_all([account, credit])
            await session.flush()
            reserve = await BillingService().check_and_reserve(
                account.id,
                25,
                job.id,
                metadata={"type": "generation", "provider": "grok"},
                session=session,
                product_id="vex",
            )
            job.token_cost = 25
            job.debit_transaction_id = reserve.txn.id
            account_id = account.id

        await session.commit()

    return _Seed(user_id=user.id, account_id=account_id, job_id=job.id)


async def _cleanup(engine: AsyncEngine, seed: _Seed) -> None:
    """Remove test rows despite the immutable-ledger trigger."""
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(text("SET LOCAL session_replication_role = 'replica'"))
        if seed.account_id is not None:
            await session.execute(
                delete(TokenTransaction).where(TokenTransaction.account_id == seed.account_id)
            )
        await session.execute(
            delete(GenerationOutput).where(GenerationOutput.job_id == seed.job_id)
        )
        await session.execute(delete(GenerationJob).where(GenerationJob.id == seed.job_id))
        if seed.account_id is not None:
            await session.execute(delete(TokenAccount).where(TokenAccount.id == seed.account_id))
        await session.execute(delete(User).where(User.id == seed.user_id))
        await session.commit()


def _materialized_video(job_id: UUID) -> _MaterializedVideo:
    output_id = new_id()
    return _MaterializedVideo(
        outputs=[
            GenerationOutputData(
                id=output_id,
                storage_key=f"test/grok/{job_id}/{output_id}.mp4",
                content_type="video/mp4",
                size_bytes=42,
                format="mp4",
                output_index=0,
                expires_at=datetime.now(UTC) + timedelta(days=7),
            )
        ],
        storage_keys=[],
    )


async def test_concurrent_finalizers_materialize_and_persist_one_output(
    db_engine: AsyncEngine,
) -> None:
    """A claim owner blocks a concurrent finalizer before it can write output."""
    seed = await _seed_video_job(db_engine)
    service = GrokJobService(MagicMock(), MagicMock(), billing_service=BillingService())
    materializer_started = asyncio.Event()
    allow_materializer_to_finish = asyncio.Event()
    materializer_calls = 0

    async def materialize(**_: object) -> _MaterializedVideo:
        nonlocal materializer_calls
        materializer_calls += 1
        materializer_started.set()
        await allow_materializer_to_finish.wait()
        return _materialized_video(seed.job_id)

    async def finalize() -> GenerationJob:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            return await service._finalize_completed_video(
                session,
                job_id=seed.job_id,
                result=GrokVideoResult(url="https://provider.invalid/video.mp4"),
                product_id="vex",
            )

    try:
        with patch.object(service, "_materialize_video_result", new=materialize):
            winner = asyncio.create_task(finalize())
            await asyncio.wait_for(materializer_started.wait(), timeout=2)

            # The first task has committed its claim but is blocked before
            # materialization. The independent second session must lose the
            # PostgreSQL CAS rather than creating another output.
            loser = await asyncio.wait_for(finalize(), timeout=2)
            assert loser.status == JobStatus.RUNNING.value
            assert materializer_calls == 1

            allow_materializer_to_finish.set()
            completed = await asyncio.wait_for(winner, timeout=2)

        assert completed.status == JobStatus.COMPLETED.value

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify_session:
            job = await verify_session.get(GenerationJob, seed.job_id)
            outputs = await verify_session.scalar(
                select(func.count())
                .select_from(GenerationOutput)
                .where(
                    GenerationOutput.job_id == seed.job_id,
                    GenerationOutput.is_thumbnail.is_(False),
                )
            )
            assert job is not None
            assert job.finalization_claim_token is None
            assert outputs == 1
    finally:
        await _cleanup(db_engine, seed)


async def test_expired_finalization_lease_is_recovered_by_a_later_poller(
    db_engine: AsyncEngine,
) -> None:
    """A crash leaves only a bounded lease, not a permanently stuck job."""
    seed = await _seed_video_job(
        db_engine,
        finalization_claim_token="stale-owner",
        finalization_lease_expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    service = GrokJobService(MagicMock(), MagicMock(), billing_service=BillingService())

    async def materialize(**_: object) -> _MaterializedVideo:
        return _materialized_video(seed.job_id)

    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            with patch.object(service, "_materialize_video_result", new=materialize):
                result = await service._finalize_completed_video(
                    session,
                    job_id=seed.job_id,
                    result=GrokVideoResult(url="https://provider.invalid/video.mp4"),
                    product_id="vex",
                )
        assert result.status == JobStatus.COMPLETED.value

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify_session:
            output_count = await verify_session.scalar(
                select(func.count())
                .select_from(GenerationOutput)
                .where(GenerationOutput.job_id == seed.job_id)
            )
            assert output_count == 1
    finally:
        await _cleanup(db_engine, seed)


async def test_refund_write_failure_rolls_back_the_terminal_transition(
    db_engine: AsyncEngine,
) -> None:
    """A failed refund leaves the real PostgreSQL job eligible for retry."""
    seed = await _seed_video_job(db_engine, with_debit=True)
    event_bus = AsyncMock()

    class FailingBilling:
        async def refund(self, **_: object) -> object:
            raise RuntimeError("ledger unavailable")

    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            transition = JobStateTransitionService(
                session=session,
                event_bus=event_bus,
                billing_service=FailingBilling(),  # type: ignore[arg-type]
            )
            with pytest.raises(RuntimeError, match="ledger unavailable"):
                await transition.transition_to_failed(
                    seed.job_id,
                    public_error_message="The AI provider timed out while processing the request.",
                    failure_code="provider_timeout",
                    refund=True,
                    product_id="vex",
                )

        event_bus.publish.assert_not_awaited()
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify_session:
            job = await verify_session.get(GenerationJob, seed.job_id)
            assert job is not None
            assert job.status == JobStatus.RUNNING.value
            assert job.failure_code is None
    finally:
        await _cleanup(db_engine, seed)


async def test_two_failure_settlers_create_one_refund_and_one_terminal_state(
    db_engine: AsyncEngine,
) -> None:
    """Two sessions racing to fail a billable job settle its debit exactly once."""
    seed = await _seed_video_job(db_engine, with_debit=True)
    assert seed.account_id is not None

    async def settle() -> bool:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            _, did_transition = await JobStateTransitionService(
                session=session,
                event_bus=None,
                billing_service=BillingService(),
            ).transition_to_failed(
                seed.job_id,
                public_error_message="The AI provider timed out while processing the request.",
                failure_code="provider_timeout",
                refund=True,
                product_id="vex",
            )
            return did_transition

    try:
        first, second = await asyncio.gather(settle(), settle())
        assert (first, second).count(True) == 1

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify_session:
            job = await verify_session.get(GenerationJob, seed.job_id)
            refund_count = await verify_session.scalar(
                select(func.count())
                .select_from(TokenTransaction)
                .where(
                    TokenTransaction.job_id == seed.job_id,
                    TokenTransaction.transaction_type == TransactionType.REFUND.value,
                )
            )
            balance = await BillingService().get_balance(seed.account_id, session=verify_session)
            assert job is not None
            assert job.status == JobStatus.FAILED.value
            assert refund_count == 1
            assert balance == 100
    finally:
        await _cleanup(db_engine, seed)


async def test_workerless_read_through_settles_an_overdue_video_once(
    db_engine: AsyncEngine,
) -> None:
    """GET-style poll-on-read uses the shared timeout/refund path without a worker."""
    seed = await _seed_video_job(
        db_engine,
        with_debit=True,
        created_at=datetime.now(UTC) - timedelta(minutes=30),
    )
    assert seed.account_id is not None
    grok_client = MagicMock()
    grok_client.get_video_result = AsyncMock()
    service = GrokJobService(
        grok_client,
        MagicMock(),
        billing_service=BillingService(),
        max_poll_time=60,
    )

    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            settled = await service.poll_video_job(session, seed.job_id)

        assert settled is not None
        assert settled.status == JobStatus.FAILED.value
        assert settled.failure_code == "provider_timeout"
        assert (
            settled.public_error_message
            == "The AI provider timed out while processing the request."
        )
        grok_client.get_video_result.assert_not_awaited()

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify_session:
            refund_count = await verify_session.scalar(
                select(func.count())
                .select_from(TokenTransaction)
                .where(
                    TokenTransaction.job_id == seed.job_id,
                    TokenTransaction.transaction_type == TransactionType.REFUND.value,
                )
            )
            assert refund_count == 1
            assert (
                await BillingService().get_balance(seed.account_id, session=verify_session) == 100
            )
    finally:
        await _cleanup(db_engine, seed)


async def test_current_claim_owner_completes_after_lease_expiry_without_takeover(
    db_engine: AsyncEngine,
) -> None:
    """Lease expiry permits takeover; it does not invalidate the same token."""
    seed = await _seed_video_job(db_engine)
    service = GrokJobService(MagicMock(), MagicMock(), billing_service=BillingService())
    materializer_started = asyncio.Event()
    allow_materializer_to_finish = asyncio.Event()

    async def materialize(**_: object) -> _MaterializedVideo:
        materializer_started.set()
        await allow_materializer_to_finish.wait()
        return _materialized_video(seed.job_id)

    async def finalize() -> GenerationJob:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            return await service._finalize_completed_video(
                session,
                job_id=seed.job_id,
                result=GrokVideoResult(url="https://provider.invalid/video.mp4"),
                product_id="vex",
            )

    try:
        with patch.object(service, "_materialize_video_result", new=materialize):
            owner = asyncio.create_task(finalize())
            await asyncio.wait_for(materializer_started.wait(), timeout=2)
            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                await session.execute(
                    update(GenerationJob)
                    .where(GenerationJob.id == seed.job_id)
                    .values(finalization_lease_expires_at=func.now() - text("interval '1 second'"))
                )
                await session.commit()
            allow_materializer_to_finish.set()
            completed = await asyncio.wait_for(owner, timeout=2)

        assert completed.status == JobStatus.COMPLETED.value
    finally:
        await _cleanup(db_engine, seed)


async def test_live_claim_blocks_failure_and_expired_claim_failure_clears_pair(
    db_engine: AsyncEngine,
) -> None:
    """Timeout/provider failure cannot preempt a live finalizer claim."""
    seed = await _seed_video_job(
        db_engine,
        finalization_claim_token="live-owner",
        finalization_lease_expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )
    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            job, did_transition = await JobStateTransitionService(
                session=session,
                event_bus=None,
                billing_service=BillingService(),
            ).transition_to_failed(
                seed.job_id,
                public_error_message="The AI provider timed out while processing the request.",
                failure_code="provider_timeout",
                refund=False,
                product_id="vex",
            )
            assert not did_transition
            assert job.status == JobStatus.RUNNING.value

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            await session.execute(
                update(GenerationJob)
                .where(GenerationJob.id == seed.job_id)
                .values(finalization_lease_expires_at=func.now() - text("interval '1 second'"))
            )
            await session.commit()

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            job, did_transition = await JobStateTransitionService(
                session=session,
                event_bus=None,
                billing_service=BillingService(),
            ).transition_to_failed(
                seed.job_id,
                public_error_message="The AI provider timed out while processing the request.",
                failure_code="provider_timeout",
                refund=False,
                product_id="vex",
            )
            assert did_transition
            assert job.status == JobStatus.FAILED.value
            assert job.finalization_claim_token is None
            assert job.finalization_lease_expires_at is None
    finally:
        await _cleanup(db_engine, seed)
