"""Integration test for the post-commit-only balance-event invariant (C6/D3).

BillingService returns a pending ``BalanceEvent`` instead of publishing it —
publishing is the caller's job, strictly after the transaction that wrote the
ledger row commits. This proves the invariant against a real PostgreSQL
session, not a mock: a rolled-back reservation must produce zero publishes,
and a committed one must produce exactly one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.api.schemas.unified_generation import UnifiedGenerationRequest
from src.api.services.billing import BalanceEvent, BillingService
from src.api.services.generation.base import ProviderSubmitResult
from src.api.services.generation.rate_limiter import ModelRateLimiter
from src.api.services.generation.service import GenerationError, GenerationService
from src.core.enums import AspectRatio, GenerationType, JobStatus, ModelType, Provider
from src.core.product_registry import VEX_CONFIG
from src.core.uid import new_id
from src.db.models.billing import TokenAccount, TokenTransaction
from src.db.models.gpu_session import GpuSession
from src.db.models.storage import GenerationJob
from src.db.models.user import User
from src.db.repositories.job import JobRepository

pytestmark = pytest.mark.asyncio


class _SpyEventBus:
    """Records ``publish_balance`` calls without touching Redis.

    GenerationService only calls ``event_bus.publish_balance(event)`` — it
    doesn't require a real ``EventBus`` instance, so a minimal duck-typed spy
    is enough to observe publish ordering against a real DB transaction.
    """

    def __init__(self) -> None:
        self.published: list[BalanceEvent | None] = []

    async def publish_balance(self, event: BalanceEvent | None) -> None:
        self.published.append(event)

    async def publish(self, **_kwargs: object) -> None:
        """No-op — GenerationService also emits a JOB_STATUS_CHANGED event."""


class _WriteThenFailProvider:
    """Writes a job row and reserves a debit, then raises before returning —
    mirrors AishaGenerationProvider's real "unexpected response" path."""

    supports_image_sizing: ClassVar[bool] = True

    def __init__(self, gpu_session_id: UUID) -> None:
        self.written_job_id: UUID | None = None
        self._gpu_session_id = gpu_session_id

    def validate(self, request: UnifiedGenerationRequest) -> None:
        pass

    async def refresh_job(
        self,
        session: AsyncSession,  # noqa: ARG002
        job: GenerationJob,  # noqa: ARG002
    ) -> GenerationJob | None:
        return None

    async def submit(
        self,
        request: UnifiedGenerationRequest,
        *,
        user_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
        account_id: UUID,
        token_cost: int,
        product_id: str,
        source_job_id: UUID | None = None,  # noqa: ARG002
        source_output_id: UUID | None = None,  # noqa: ARG002
    ) -> ProviderSubmitResult:
        job_id = new_id()
        self.written_job_id = job_id
        await JobRepository(session).create(
            id=job_id,
            user_id=user_id,
            name="Test job",
            prompt=request.prompt,
            generation_type=request.generation_type,
            status=JobStatus.PENDING,
            provider=Provider.AISHA,
            model=request.model.value,
            aspect_ratio=request.aspect_ratio.value,
            product_id=product_id,
            gpu_session_id=self._gpu_session_id,
        )
        reserve_result = await billing_service.check_and_reserve(
            account_id,
            token_cost,
            job_id,
            metadata={"type": "generation", "provider": "aisha"},
            description="Generation charge",
            session=session,
            product_id=product_id,
            user_id=user_id,
        )
        await session.flush()
        # This event describes a debit that is about to be rolled back — it
        # must never reach the spy bus.
        assert reserve_result.event is not None
        raise RuntimeError("Simulated backend failure after job+debit write")


class _WriteAndSucceedProvider:
    """Writes a job row, reserves a debit, and returns normally."""

    supports_image_sizing: ClassVar[bool] = True

    def __init__(self, gpu_session_id: UUID) -> None:
        self._gpu_session_id = gpu_session_id

    def validate(self, request: UnifiedGenerationRequest) -> None:
        pass

    async def refresh_job(
        self,
        session: AsyncSession,  # noqa: ARG002
        job: GenerationJob,  # noqa: ARG002
    ) -> GenerationJob | None:
        return None

    async def submit(
        self,
        request: UnifiedGenerationRequest,
        *,
        user_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
        account_id: UUID,
        token_cost: int,
        product_id: str,
        source_job_id: UUID | None = None,  # noqa: ARG002
        source_output_id: UUID | None = None,  # noqa: ARG002
    ) -> ProviderSubmitResult:
        job_id = new_id()
        db_job = await JobRepository(session).create(
            id=job_id,
            user_id=user_id,
            name="Test job",
            prompt=request.prompt,
            generation_type=request.generation_type,
            status=JobStatus.COMPLETED,
            provider=Provider.AISHA,
            model=request.model.value,
            aspect_ratio=request.aspect_ratio.value,
            product_id=product_id,
            gpu_session_id=self._gpu_session_id,
        )
        reserve_result = await billing_service.check_and_reserve(
            account_id,
            token_cost,
            job_id,
            metadata={"type": "generation", "provider": "aisha"},
            description="Generation charge",
            session=session,
            product_id=product_id,
            user_id=user_id,
        )
        db_job.token_cost = token_cost
        db_job.debit_transaction_id = reserve_result.txn.id
        await session.flush()
        return ProviderSubmitResult(job=db_job, balance_event=reserve_result.event)


def _make_enabled_model_mock() -> MagicMock:
    record = MagicMock()
    record.is_enabled = True
    return record


async def _seed_user_and_account(
    engine: AsyncEngine, *, balance: int
) -> tuple[User, TokenAccount, GpuSession]:
    user = User(
        id=new_id(),
        email=f"event-order-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id="vex",
        is_active=True,
    )
    account = TokenAccount(
        id=new_id(),
        user_id=user.id,
        account_type="personal",
        product_id="vex",
    )
    credit = TokenTransaction(
        id=new_id(),
        account_id=account.id,
        transaction_type="credit",
        amount=balance,
        balance_after=balance,
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        bundle_name="wan_2.2_i2v",
        model_type="aisha-image",
        status="active",
    )
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add(user)
        await session.flush()
        session.add_all([account, credit, gpu_session])
        await session.commit()
        await session.refresh(user)
        await session.refresh(account)
        await session.refresh(gpu_session)
    return user, account, gpu_session


async def _cleanup(engine: AsyncEngine, *, account_id: UUID, user_id: UUID) -> None:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(text("SET LOCAL session_replication_role = 'replica'"))
        await session.execute(
            delete(TokenTransaction).where(TokenTransaction.account_id == account_id)
        )
        await session.execute(delete(GenerationJob).where(GenerationJob.user_id == user_id))
        await session.execute(delete(TokenAccount).where(TokenAccount.id == account_id))
        await session.execute(delete(GpuSession).where(GpuSession.user_id == user_id))
        await session.execute(delete(User).where(User.id == user_id))
        await session.commit()


def _make_request() -> UnifiedGenerationRequest:
    return UnifiedGenerationRequest(
        prompt="A cat",
        generation_type=GenerationType.T2I,
        model=ModelType.AISHA_IMAGE,
        aspect_ratio=AspectRatio.RATIO_1_1,
    )


async def test_no_balance_event_on_rollback(db_engine: AsyncEngine) -> None:
    """provider.submit() writes a debit, then raises → the whole transaction
    (including the debit) rolls back. Zero balance events must reach the
    event bus — the debit never became durable."""
    user, account, gpu_session = await _seed_user_and_account(db_engine, balance=1000)

    provider = _WriteThenFailProvider(gpu_session_id=gpu_session.id)
    pricing = AsyncMock()
    pricing.get_price = AsyncMock(return_value=50)
    billing = BillingService()
    spy_bus = _SpyEventBus()

    service = GenerationService(
        providers={Provider.AISHA: provider},
        billing_service=billing,
        pricing_service=pricing,
        rate_limiter=MagicMock(spec=ModelRateLimiter),
        event_bus=spy_bus,  # type: ignore[arg-type]
    )

    try:
        session = AsyncSession(bind=db_engine, expire_on_commit=False)
        try:
            with (
                patch(
                    "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                    new=AsyncMock(return_value=_make_enabled_model_mock()),
                ),
                patch(
                    "src.db.repositories.user.UserRepository.get_user",
                    new=AsyncMock(return_value=MagicMock(age_verified_at=datetime.now(UTC))),
                ),
                pytest.raises(GenerationError),
            ):
                await service.generate(
                    _make_request(),
                    user_id=user.id,
                    session=session,
                    product_config=VEX_CONFIG,
                )
        finally:
            await session.close()

        assert spy_bus.published == [], (
            f"Expected zero balance-event publishes on rollback, got {spy_bus.published}"
        )

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify_session:
            assert provider.written_job_id is not None
            job_row = await verify_session.get(GenerationJob, provider.written_job_id)
            assert job_row is None, "Job row survived the rollback"
            final_balance = await billing.get_balance(account.id, session=verify_session)
            assert final_balance == 1000
    finally:
        await _cleanup(db_engine, account_id=account.id, user_id=user.id)


async def test_balance_event_after_commit(db_engine: AsyncEngine) -> None:
    """provider.submit() succeeds → the reserve's BalanceEvent is published
    exactly once, strictly after the commit that persisted the debit."""
    user, account, gpu_session = await _seed_user_and_account(db_engine, balance=1000)

    provider = _WriteAndSucceedProvider(gpu_session_id=gpu_session.id)
    pricing = AsyncMock()
    pricing.get_price = AsyncMock(return_value=50)
    billing = BillingService()
    spy_bus = _SpyEventBus()

    service = GenerationService(
        providers={Provider.AISHA: provider},
        billing_service=billing,
        pricing_service=pricing,
        rate_limiter=MagicMock(spec=ModelRateLimiter),
        event_bus=spy_bus,  # type: ignore[arg-type]
    )

    try:
        session = AsyncSession(bind=db_engine, expire_on_commit=False)
        try:
            with (
                patch(
                    "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                    new=AsyncMock(return_value=_make_enabled_model_mock()),
                ),
                patch(
                    "src.db.repositories.user.UserRepository.get_user",
                    new=AsyncMock(return_value=MagicMock(age_verified_at=datetime.now(UTC))),
                ),
            ):
                await service.generate(
                    _make_request(),
                    user_id=user.id,
                    session=session,
                    product_config=VEX_CONFIG,
                )
        finally:
            await session.close()

        assert len(spy_bus.published) == 1, (
            f"Expected exactly one balance-event publish, got {spy_bus.published}"
        )
        event = spy_bus.published[0]
        assert event is not None
        assert event.user_ids == [user.id]
        assert event.account_id == account.id
        assert event.transaction_type == "debit"

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify_session:
            # The debit the event describes is durable — proving the publish
            # happened for a row that actually committed.
            balance = await billing.get_balance(account.id, session=verify_session)
            assert balance == 950  # 1000 - 50 (token_cost)
    finally:
        await _cleanup(db_engine, account_id=account.id, user_id=user.id)
