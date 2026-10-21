"""Accepted-task receipts against real PostgreSQL (R3 / P1-1).

A provider accepts an async video task inside the still-open request
transaction; a failure anywhere before the route commits rolls back the job row
and the reservation while the task keeps running upstream. These tests drive the
real ``GenerationService`` with the real Wan and Grok job services and assert the
task id survives as exactly one receipt — and that a successful request leaves
none.

The probes inject failures at every window between acceptance and commit:
``update_status``, ``_require_job``, the job-source write, and the commit itself
(including an acknowledgement lost after a successful commit).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.ops_events import OpsEventType
from src.api.schemas.unified_generation import SourceMediaReference, UnifiedGenerationRequest
from src.api.services.billing import BillingService
from src.api.services.generation.grok_provider import GrokGenerationProvider
from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
from src.api.services.generation.rate_limiter import ModelRateLimiter
from src.api.services.generation.service import GenerationError, GenerationService
from src.api.services.generation.wan_provider import WanGenerationProvider
from src.api.services.grok import GrokVideoJobStarted
from src.api.services.model_studio import ModelStudioTaskRef
from src.api.services.model_studio.wan_job_service import WanJobService
from src.api.services.video_jobs.protocol import (
    OrphanCancelOutcome,
    OrphanObservation,
    OrphanTaskState,
)
from src.api.services.video_jobs.receipts import AcceptedTask, AcceptedTaskReceipts
from src.api.services.video_jobs.reconciler import AcceptedTaskReconciler
from src.core.config import Settings
from src.core.enums import GenerationType, JobStatus, ModelType, Provider
from src.core.product_registry import VEX_CONFIG
from src.core.uid import new_id
from src.db.models.accepted_task_receipt import AcceptedTaskReceipt
from src.db.models.billing import TokenAccount, TokenTransaction
from src.db.models.storage import GenerationJob, UserImage
from src.db.models.user import User
from src.db.repositories.accepted_task_receipt import AcceptedTaskReceiptRepository
from src.db.repositories.job import JobRepository
from src.db.session import DatabaseManager
from tests.grok_support import make_grok_job_service
from tests.video_receipt_support import stub_receipts

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.asyncio

_WAN_TASK = "task-accepted-1"
_GROK_TASK = "req-accepted-1"
_REAL_UPDATE_STATUS = JobRepository.update_status


# ---- fixtures and helpers --------------------------------------------------------------


class _Transient(Exception):
    pass


def _fake_compensator(
    *,
    observe: OrphanObservation | None = None,
    cancel: OrphanCancelOutcome = OrphanCancelOutcome.CANCELLED,
    observe_error: Exception | None = None,
    provider: Provider = Provider.ALIBABA,
) -> Any:
    observed = observe or OrphanObservation(OrphanTaskState.PENDING, "PENDING")
    return SimpleNamespace(
        provider=provider,
        transient_errors=(_Transient,),
        cancel=AsyncMock(return_value=cancel),
        observe=AsyncMock(side_effect=observe_error)
        if observe_error
        else AsyncMock(return_value=observed),
    )


@pytest.fixture
async def receipt_db(
    test_database_url: str, db_engine: AsyncEngine
) -> AsyncIterator[DatabaseManager]:
    del db_engine  # ensures migrations ran
    manager = DatabaseManager(
        test_database_url, pool_size=2, max_overflow=2, pool_timeout=2, pool_recycle=300
    )
    yield manager
    await manager.close()


@pytest.fixture
def ops_bus() -> MagicMock:
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def receipts(receipt_db: DatabaseManager, ops_bus: MagicMock) -> AcceptedTaskReceipts:
    return AcceptedTaskReceipts(receipt_db, ops_bus, record_attempts=2)


async def _seed(engine: AsyncEngine, *, balance: int = 1000) -> tuple[User, TokenAccount]:
    user = User(
        id=new_id(),
        email=f"receipt-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id="vex",
        is_active=True,
    )
    account = TokenAccount(id=new_id(), user_id=user.id, account_type="personal", product_id="vex")
    credit = TokenTransaction(
        id=new_id(),
        account_id=account.id,
        transaction_type="credit",
        amount=balance,
        balance_after=balance,
        product_id="vex",
    )
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add(user)
        await session.flush()
        session.add_all([account, credit])
        await session.commit()
    return user, account


async def _cleanup(engine: AsyncEngine, user_id: UUID, account_id: UUID) -> None:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(text("SET LOCAL session_replication_role = 'replica'"))
        await session.execute(
            delete(AcceptedTaskReceipt).where(AcceptedTaskReceipt.user_id == user_id)
        )
        await session.execute(
            delete(TokenTransaction).where(TokenTransaction.account_id == account_id)
        )
        await session.execute(delete(GenerationJob).where(GenerationJob.user_id == user_id))
        await session.execute(delete(UserImage).where(UserImage.user_id == user_id))
        await session.execute(delete(TokenAccount).where(TokenAccount.id == account_id))
        await session.execute(delete(User).where(User.id == user_id))
        await session.commit()


async def _receipt_rows(engine: AsyncEngine, user_id: UUID) -> list[AcceptedTaskReceipt]:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        result = await session.execute(
            select(AcceptedTaskReceipt).where(AcceptedTaskReceipt.user_id == user_id)
        )
        return list(result.scalars())


async def _count(engine: AsyncEngine, model: type, *conditions: Any) -> int:
    async with AsyncSession(bind=engine) as session:
        return int(
            await session.scalar(select(func.count()).select_from(model).where(*conditions)) or 0
        )


def _service(provider_key: Provider, provider: Any) -> GenerationService:
    pricing = AsyncMock()
    pricing.quote = AsyncMock(return_value=50)
    return GenerationService(
        providers={provider_key: provider},
        billing_service=BillingService(),
        pricing_service=pricing,
        rate_limiter=MagicMock(spec=ModelRateLimiter),
    )


def _wan_stack(
    receipts: AcceptedTaskReceipts, compensator: Any
) -> tuple[GenerationService, MagicMock]:
    client = MagicMock()
    client.create_video_task = AsyncMock(return_value=ModelStudioTaskRef(task_id=_WAN_TASK))
    wan = WanJobService(
        client,
        MagicMock(),
        MagicMock(),
        receipts=receipts,
        compensator=compensator,
        billing_policy=ProviderBillingPolicyRegistry.from_settings(Settings.model_construct()),
    )
    r2 = MagicMock()
    r2.get_presigned_url = AsyncMock(return_value=SimpleNamespace(presigned_url="https://r2/x"))
    provider = WanGenerationProvider(wan, r2, input_url_ttl_seconds=900)
    return _service(Provider.ALIBABA, provider), client.create_video_task


def _grok_stack(
    receipts: AcceptedTaskReceipts, compensator: Any
) -> tuple[GenerationService, MagicMock]:
    client = MagicMock()
    client.start_video_generation = AsyncMock(
        return_value=GrokVideoJobStarted(request_id=_GROK_TASK)
    )
    job_service = make_grok_job_service(
        client,
        MagicMock(),
        media_ingestor=MagicMock(),
        billing_service=BillingService(),
        receipts=receipts,
    )
    job_service._compensator = compensator
    r2 = MagicMock()
    r2.get_presigned_url = AsyncMock(return_value=SimpleNamespace(presigned_url="https://r2/x"))
    provider = GrokGenerationProvider(job_service, r2)
    return _service(Provider.GROK, provider), client.start_video_generation


def _request(kind: str, source_ref: str | None = None) -> UnifiedGenerationRequest:
    if kind == "wan":
        if source_ref is not None:
            return UnifiedGenerationRequest(
                prompt="a fox",
                generation_type=GenerationType.I2V,
                model=ModelType.WAN_3_VIDEO,
                duration=5,
                source_media=[SourceMediaReference(asset_ref=source_ref)],
            )
        return UnifiedGenerationRequest(
            prompt="a fox",
            generation_type=GenerationType.T2V,
            model=ModelType.WAN_3_VIDEO,
            duration=5,
        )
    if source_ref is not None:
        return UnifiedGenerationRequest(
            prompt="a fox",
            generation_type=GenerationType.I2V,
            model=ModelType.GROK_IMAGINE_VIDEO,
            duration=5,
            source_media=[SourceMediaReference(asset_ref=source_ref)],
        )
    return UnifiedGenerationRequest(
        prompt="a fox",
        generation_type=GenerationType.T2V,
        model=ModelType.GROK_IMAGINE_VIDEO,
        duration=5,
    )


async def _fail_queued(self: JobRepository, job_id: UUID, status: JobStatus, **kwargs: Any) -> Any:
    if status is JobStatus.QUEUED:
        raise OperationalError("UPDATE generation_jobs", {}, Exception("connection reset"))
    return await _REAL_UPDATE_STATUS(self, job_id, status, **kwargs)


async def _vanishing_queued(
    self: JobRepository, job_id: UUID, status: JobStatus, **kwargs: Any
) -> Any:
    if status is JobStatus.QUEUED:
        await _REAL_UPDATE_STATUS(self, job_id, status, **kwargs)
        return None
    return await _REAL_UPDATE_STATUS(self, job_id, status, **kwargs)


async def _generate(
    engine: AsyncEngine,
    service: GenerationService,
    request: UnifiedGenerationRequest,
    user: User,
    *,
    session_patch: Any = None,
    extra_patches: tuple[Any, ...] = (),
    expect_error: bool = True,
) -> None:
    session = AsyncSession(bind=engine, expire_on_commit=False)
    enabled = MagicMock(is_enabled=True)
    try:
        patches = [
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=enabled),
            ),
            patch(
                "src.db.repositories.user.UserRepository.get_user",
                new=AsyncMock(return_value=MagicMock(age_verified_at=datetime.now(UTC))),
            ),
            *extra_patches,
        ]
        if session_patch is not None:
            patches.append(patch.object(session, "commit", new=session_patch(session)))
        from contextlib import ExitStack

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            if expect_error:
                with pytest.raises(GenerationError):
                    await service.generate(
                        request, user_id=user.id, session=session, product_config=VEX_CONFIG
                    )
            else:
                await service.generate(
                    request, user_id=user.id, session=session, product_config=VEX_CONFIG
                )
    finally:
        await session.close()


def _commit_aborts(session: AsyncSession) -> Any:
    real_commit, real_rollback = session.commit, session.rollback
    failed = False

    async def commit() -> None:
        nonlocal failed
        if failed:
            await real_commit()
            return
        failed = True
        await real_rollback()  # the server aborted the transaction
        raise OperationalError("COMMIT", {}, Exception("connection reset"))

    return commit


def _commit_ack_lost(session: AsyncSession) -> Any:
    real_commit = session.commit
    failed = False

    async def commit() -> None:
        nonlocal failed
        await real_commit()
        if not failed:
            failed = True
            raise OperationalError("COMMIT", {}, Exception("connection reset"))

    return commit


async def _seed_upload(engine: AsyncEngine, user_id: UUID) -> UUID:
    image = UserImage(
        id=new_id(),
        user_id=user_id,
        product_id="vex",
        storage_key=f"users/{user_id}/uploads/{uuid4()}.png",
        original_filename="a.png",
        content_type="image/png",
        size_bytes=1024,
        format="png",
        expires_at=datetime.now(UTC) + timedelta(days=1),
    )
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add(image)
        await session.commit()
    return image.id


async def _assert_not_locked(engine: AsyncEngine, receipt_id: UUID) -> None:
    async with AsyncSession(bind=engine) as session:
        locked = await session.execute(
            text("SELECT id FROM accepted_task_receipts WHERE id = :id FOR UPDATE NOWAIT"),
            {"id": receipt_id},
        )
        assert locked.scalar_one() == receipt_id


# ---- P1-1: every window leaves exactly one receipt --------------------------------------

_IN_JOB_SERVICE = pytest.param("update_status", id="update_status")
_WINDOWS = [
    pytest.param("wan", "update_status", id="wan-update_status"),
    pytest.param("wan", "require_job", id="wan-require_job"),
    pytest.param("wan", "commit", id="wan-commit"),
    pytest.param("grok", "update_status", id="grok-update_status"),
    pytest.param("grok", "require_job", id="grok-require_job"),
    pytest.param("grok", "commit", id="grok-commit"),
]


@pytest.mark.parametrize(("kind", "window"), _WINDOWS)
async def test_a_task_accepted_upstream_is_never_lost(
    db_engine: AsyncEngine,
    receipts: AcceptedTaskReceipts,
    receipt_db: DatabaseManager,
    ops_bus: MagicMock,
    kind: str,
    window: str,
) -> None:
    user, account = await _seed(db_engine)
    task_id = _WAN_TASK if kind == "wan" else _GROK_TASK
    compensator = _fake_compensator(provider=Provider.ALIBABA if kind == "wan" else Provider.GROK)
    stack = _wan_stack if kind == "wan" else _grok_stack
    service, upstream = stack(receipts, compensator)
    request = _request(kind)
    extra: tuple[Any, ...] = ()
    session_patch = None
    if window == "update_status":
        extra = (patch.object(JobRepository, "update_status", new=_fail_queued),)
    elif window == "require_job":
        extra = (patch.object(JobRepository, "update_status", new=_vanishing_queued),)
    elif window == "sources":
        upload_id = await _seed_upload(db_engine, user.id)
        request = _request(kind, source_ref=f"upload:{upload_id}")
        extra = (
            patch(
                "src.api.services.generation.service.GenerationJobSourceRepository.create_many",
                new=AsyncMock(
                    side_effect=OperationalError("INSERT", {}, Exception("connection reset"))
                ),
            ),
        )
    else:
        session_patch = _commit_aborts

    try:
        await _generate(
            db_engine, service, request, user, session_patch=session_patch, extra_patches=extra
        )

        # The provider accepted the task, the client saw a 500, nothing else survived …
        assert upstream.await_count == 1
        assert await _count(db_engine, GenerationJob, GenerationJob.user_id == user.id) == 0
        async with AsyncSession(bind=db_engine) as session:
            assert await BillingService().get_balance(account.id, session=session) == 1000
        # … except exactly one receipt carrying the upstream task id.
        rows = await _receipt_rows(db_engine, user.id)
        assert [r.external_task_id for r in rows] == [task_id]
        (row,) = rows
        assert row.token_cost == 50
        assert row.provider == compensator.provider.value

        if window in {"update_status", "require_job"}:
            compensator.cancel.assert_awaited_once_with(task_id)
            assert row.status == "cancelled"
            assert row.resolved_at is not None
            published = [c.kwargs["event_type"] for c in ops_bus.publish.await_args_list]
            assert published == [OpsEventType.VIDEO_TASK_ORPHANED]
        else:
            compensator.cancel.assert_not_awaited()
            assert row.status == "pending"
            await _assert_not_locked(db_engine, row.id)

            reconciler = AcceptedTaskReconciler(receipt_db, compensator, ops_bus, grace_seconds=0)
            assert await reconciler.run_once() == 1
            (resolved,) = await _receipt_rows(db_engine, user.id)
            assert resolved.status == "cancelled"
            compensator.cancel.assert_awaited_once_with(task_id)
            async with AsyncSession(bind=db_engine) as session:
                assert await BillingService().get_balance(account.id, session=session) == 1000
        assert (
            await _count(db_engine, TokenTransaction, TokenTransaction.account_id == account.id)
            == 1
        )  # only the seed credit: no debit, refund or adopt-and-charge
    finally:
        await _cleanup(db_engine, user.id, account.id)


def _failing_source_write() -> Any:
    """A non-DB error after ``submit`` (the session is *not* poisoned)."""
    return patch(
        "src.api.services.generation.service.GenerationJobSourceRepository.create_many",
        new=AsyncMock(side_effect=ValueError("unexpected source row")),
    )


@pytest.mark.parametrize("kind", ["wan", "grok"])
async def test_a_failed_source_write_restores_the_receipt_instead_of_committing_a_job(
    db_engine: AsyncEngine,
    receipts: AcceptedTaskReceipts,
    receipt_db: DatabaseManager,
    ops_bus: MagicMock,
    kind: str,
) -> None:
    """P1-3: a post-submit failure with the receipt held rolls everything back.

    No job row (so no poller can complete it for free), an unchanged balance, and
    one ``pending`` receipt that the reconciler resolves — exactly like every
    other window after acceptance.
    """
    user, account = await _seed(db_engine)
    task_id = _WAN_TASK if kind == "wan" else _GROK_TASK
    compensator = _fake_compensator(provider=Provider.ALIBABA if kind == "wan" else Provider.GROK)
    stack = _wan_stack if kind == "wan" else _grok_stack
    service, upstream = stack(receipts, compensator)
    upload_id = await _seed_upload(db_engine, user.id)
    try:
        await _generate(
            db_engine,
            service,
            _request(kind, source_ref=f"upload:{upload_id}"),
            user,
            extra_patches=(_failing_source_write(),),
        )

        assert upstream.await_count == 1
        assert await _count(db_engine, GenerationJob, GenerationJob.user_id == user.id) == 0
        async with AsyncSession(bind=db_engine) as session:
            assert await BillingService().get_balance(account.id, session=session) == 1000
        (row,) = await _receipt_rows(db_engine, user.id)
        assert (row.external_task_id, row.status) == (task_id, "pending")
        await _assert_not_locked(db_engine, row.id)

        reconciler = AcceptedTaskReconciler(receipt_db, compensator, ops_bus, grace_seconds=0)
        assert await reconciler.run_once() == 1
        (resolved,) = await _receipt_rows(db_engine, user.id)
        assert resolved.status == "cancelled"
        compensator.cancel.assert_awaited_once_with(task_id)
    finally:
        await _cleanup(db_engine, user.id, account.id)


async def test_a_failed_source_write_without_a_receipt_fails_and_refunds_the_job(
    db_engine: AsyncEngine,
) -> None:
    """P1-3: no receipt to restore ⇒ the job is committed terminal, never left live."""
    user, account = await _seed(db_engine)
    compensator = _fake_compensator()
    service, upstream = _wan_stack(stub_receipts(recorded=False), compensator)
    upload_id = await _seed_upload(db_engine, user.id)
    try:
        await _generate(
            db_engine,
            service,
            _request("wan", source_ref=f"upload:{upload_id}"),
            user,
            extra_patches=(_failing_source_write(),),
        )

        assert upstream.await_count == 1
        async with AsyncSession(bind=db_engine) as session:
            (job,) = (
                (
                    await session.execute(
                        select(GenerationJob).where(GenerationJob.user_id == user.id)
                    )
                )
                .scalars()
                .all()
            )
            assert job.status == JobStatus.FAILED.value
            assert job.failure_code == "internal_error"
            assert job.public_error_message == "Generation failed due to an internal error."
            assert await BillingService().get_balance(account.id, session=session) == 1000
            ledger = (
                await session.execute(
                    select(TokenTransaction.transaction_type)
                    .where(TokenTransaction.account_id == account.id)
                    .order_by(TokenTransaction.created_at, TokenTransaction.id)
                )
            ).scalars()
            assert sorted(ledger) == ["credit", "debit", "refund"]  # refunded exactly once
            polled = await JobRepository(session).list_pending_video_jobs(provider=Provider.ALIBABA)
            assert job.id not in {j.id for j in polled}
        assert await _receipt_rows(db_engine, user.id) == []
        compensator.cancel.assert_not_awaited()
    finally:
        await _cleanup(db_engine, user.id, account.id)


@pytest.mark.parametrize("kind", ["wan", "grok"])
async def test_a_successful_request_leaves_no_receipt(
    db_engine: AsyncEngine, receipts: AcceptedTaskReceipts, kind: str
) -> None:
    user, account = await _seed(db_engine)
    provider = Provider.ALIBABA if kind == "wan" else Provider.GROK
    compensator = _fake_compensator(provider=provider)
    service, upstream = (_wan_stack if kind == "wan" else _grok_stack)(receipts, compensator)
    try:
        await _generate(db_engine, service, _request(kind), user, expect_error=False)

        assert upstream.await_count == 1
        assert await _count(db_engine, GenerationJob, GenerationJob.user_id == user.id) == 1
        assert await _receipt_rows(db_engine, user.id) == []
        compensator.cancel.assert_not_awaited()
    finally:
        await _cleanup(db_engine, user.id, account.id)


async def test_a_lost_commit_acknowledgement_leaves_no_receipt(
    db_engine: AsyncEngine, receipts: AcceptedTaskReceipts
) -> None:
    """The commit lands, then the ack raises: the delete committed with the job."""
    user, account = await _seed(db_engine)
    service, _ = _wan_stack(receipts, _fake_compensator())
    try:
        await _generate(db_engine, service, _request("wan"), user, session_patch=_commit_ack_lost)

        assert await _count(db_engine, GenerationJob, GenerationJob.user_id == user.id) == 1
        assert await _receipt_rows(db_engine, user.id) == []
    finally:
        await _cleanup(db_engine, user.id, account.id)


# ---- store and lock semantics ------------------------------------------------------------


def _accepted(user: User, task_id: str = "task-lock-1") -> AcceptedTask:
    return AcceptedTask(
        provider=Provider.ALIBABA,
        external_task_id=task_id,
        job_id=new_id(),
        user_id=user.id,
        product_id="vex",
        model="wan3.0-video",
        token_cost=40,
    )


async def test_record_is_idempotent_and_uses_its_own_connection(
    db_engine: AsyncEngine, receipts: AcceptedTaskReceipts
) -> None:
    user, account = await _seed(db_engine)
    try:
        assert await receipts.record(_accepted(user)) is True
        assert await receipts.record(_accepted(user)) is True  # ON CONFLICT DO NOTHING
        assert len(await _receipt_rows(db_engine, user.id)) == 1
    finally:
        await _cleanup(db_engine, user.id, account.id)


async def test_record_succeeds_while_the_request_pool_is_exhausted(
    test_database_url: str, receipts: AcceptedTaskReceipts, db_engine: AsyncEngine
) -> None:
    user, account = await _seed(db_engine)
    main = DatabaseManager(test_database_url, pool_size=1, max_overflow=0, pool_timeout=1)
    try:
        async with main.session() as held:
            await held.execute(text("SELECT 1"))  # the only request-pool connection is out
            assert await receipts.record(_accepted(user)) is True
    finally:
        await main.close()
        await _cleanup(db_engine, user.id, account.id)


async def test_the_reconciler_skips_a_receipt_an_in_flight_request_holds(
    db_engine: AsyncEngine,
    receipts: AcceptedTaskReceipts,
    receipt_db: DatabaseManager,
    ops_bus: MagicMock,
) -> None:
    user, account = await _seed(db_engine)
    compensator = _fake_compensator()
    reconciler = AcceptedTaskReconciler(receipt_db, compensator, ops_bus, grace_seconds=0)
    task = _accepted(user)
    try:
        assert await receipts.record(task) is True
        request = AsyncSession(bind=db_engine, expire_on_commit=False)
        try:
            await receipts.claim_for_request(
                request, provider=task.provider, external_task_id=task.external_task_id
            )  # in flight: the delete lock is held

            assert await reconciler.run_once() == 0
            compensator.observe.assert_not_awaited()
            compensator.cancel.assert_not_awaited()

            await request.rollback()  # the request failed: the row reappears, unlocked
        finally:
            await request.close()

        assert await reconciler.run_once() == 1
        (row,) = await _receipt_rows(db_engine, user.id)
        assert row.status == "cancelled"
    finally:
        await _cleanup(db_engine, user.id, account.id)


async def test_a_committed_request_removes_the_receipt_with_its_job(
    db_engine: AsyncEngine, receipts: AcceptedTaskReceipts
) -> None:
    user, account = await _seed(db_engine)
    task = _accepted(user)
    try:
        await receipts.record(task)
        async with AsyncSession(bind=db_engine) as request:
            await receipts.claim_for_request(
                request, provider=task.provider, external_task_id=task.external_task_id
            )
            await request.commit()

        assert await _receipt_rows(db_engine, user.id) == []
    finally:
        await _cleanup(db_engine, user.id, account.id)


@pytest.mark.parametrize(("isolation", "deleted"), [("READ COMMITTED", 1), ("REPEATABLE READ", 0)])
async def test_claim_needs_read_committed_to_see_a_receipt_committed_mid_transaction(
    db_engine: AsyncEngine,
    receipts: AcceptedTaskReceipts,
    isolation: str,
    deleted: int,
) -> None:
    """Pins the isolation requirement documented on ``claim_for_request``."""
    user, account = await _seed(db_engine)
    task = _accepted(user)
    engine = db_engine.execution_options(isolation_level=isolation)
    try:
        async with AsyncSession(bind=engine) as request:
            await request.execute(text("SELECT 1"))  # the snapshot predates the receipt
            await receipts.record(task)

            rows = await AcceptedTaskReceiptRepository(request).delete_pending(
                provider=task.provider.value, external_task_id=task.external_task_id
            )
            await request.rollback()

        assert rows == deleted
    finally:
        await _cleanup(db_engine, user.id, account.id)


# ---- reconciler matrix --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observe", "cancel", "status", "upstream", "cancel_calls"),
    [
        (
            OrphanObservation(OrphanTaskState.PENDING, "PENDING"),
            OrphanCancelOutcome.CANCELLED,
            "cancelled",
            "PENDING",
            1,
        ),
        (
            OrphanObservation(OrphanTaskState.CANCELLED, "CANCELED"),
            OrphanCancelOutcome.CANCELLED,
            "cancelled",
            "CANCELED",
            0,
        ),
        (
            OrphanObservation(OrphanTaskState.OTHER, "RUNNING"),
            OrphanCancelOutcome.CANCELLED,
            "absorbed",
            "RUNNING",
            0,
        ),
        (
            OrphanObservation(OrphanTaskState.PENDING, "PENDING"),
            OrphanCancelOutcome.NOT_CANCELLABLE,
            "absorbed",
            "PENDING",
            1,
        ),
        (
            OrphanObservation(OrphanTaskState.PENDING, "pending"),
            OrphanCancelOutcome.UNSUPPORTED,
            "absorbed",
            "pending",
            1,
        ),
        (
            OrphanObservation(OrphanTaskState.GONE, None),
            OrphanCancelOutcome.CANCELLED,
            "absorbed",
            "unknown",
            0,
        ),
    ],
)
async def test_reconciler_observes_first_then_acts_and_never_charges(
    db_engine: AsyncEngine,
    receipts: AcceptedTaskReceipts,
    receipt_db: DatabaseManager,
    ops_bus: MagicMock,
    observe: OrphanObservation,
    cancel: OrphanCancelOutcome,
    status: str,
    upstream: str,
    cancel_calls: int,
) -> None:
    user, account = await _seed(db_engine)
    compensator = _fake_compensator(observe=observe, cancel=cancel)
    reconciler = AcceptedTaskReconciler(receipt_db, compensator, ops_bus, grace_seconds=0)
    try:
        await receipts.record(_accepted(user))
        ledger_before = await _count(db_engine, TokenTransaction)

        assert await reconciler.run_once() == 1
        assert await reconciler.run_once() == 0  # resolved once, never again

        (row,) = await _receipt_rows(db_engine, user.id)
        assert (row.status, row.upstream_status) == (status, upstream)
        assert row.resolved_at is not None
        assert compensator.cancel.await_count == cancel_calls
        orphaned = [
            c
            for c in ops_bus.publish.await_args_list
            if c.kwargs["event_type"] is OpsEventType.VIDEO_TASK_ORPHANED
        ]
        assert len(orphaned) == 1
        assert orphaned[0].kwargs["payload"].resolution == status
        assert await _count(db_engine, TokenTransaction) == ledger_before
    finally:
        await _cleanup(db_engine, user.id, account.id)


async def test_a_transient_error_keeps_the_receipt_pending_for_the_next_tick(
    db_engine: AsyncEngine,
    receipts: AcceptedTaskReceipts,
    receipt_db: DatabaseManager,
    ops_bus: MagicMock,
) -> None:
    user, account = await _seed(db_engine)
    compensator = _fake_compensator(observe_error=_Transient("upstream 503"))
    reconciler = AcceptedTaskReconciler(receipt_db, compensator, ops_bus, grace_seconds=0)
    try:
        await receipts.record(_accepted(user))

        assert await reconciler.run_once() == 0

        (row,) = await _receipt_rows(db_engine, user.id)
        assert row.status == "pending"
        assert row.last_checked_at is not None
        assert row.resolved_at is None
        ops_bus.publish.assert_not_awaited()
    finally:
        await _cleanup(db_engine, user.id, account.id)


async def test_a_permanent_compensation_error_is_absorbed_for_manual_audit(
    db_engine: AsyncEngine,
    receipts: AcceptedTaskReceipts,
    receipt_db: DatabaseManager,
    ops_bus: MagicMock,
) -> None:
    user, account = await _seed(db_engine)
    compensator = _fake_compensator(observe_error=RuntimeError("malformed"))
    reconciler = AcceptedTaskReconciler(receipt_db, compensator, ops_bus, grace_seconds=0)
    try:
        await receipts.record(_accepted(user))

        assert await reconciler.run_once() == 1

        (row,) = await _receipt_rows(db_engine, user.id)
        assert (row.status, row.upstream_status) == ("absorbed", "unknown")
    finally:
        await _cleanup(db_engine, user.id, account.id)


async def test_the_grace_period_protects_a_fresh_receipt(
    db_engine: AsyncEngine,
    receipts: AcceptedTaskReceipts,
    receipt_db: DatabaseManager,
    ops_bus: MagicMock,
) -> None:
    user, account = await _seed(db_engine)
    compensator = _fake_compensator()
    reconciler = AcceptedTaskReconciler(receipt_db, compensator, ops_bus, grace_seconds=3600)
    try:
        await receipts.record(_accepted(user))
        await asyncio.sleep(0)

        assert await reconciler.run_once() == 0
        compensator.observe.assert_not_awaited()
    finally:
        await _cleanup(db_engine, user.id, account.id)
