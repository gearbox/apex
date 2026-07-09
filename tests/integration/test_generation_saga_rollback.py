"""Integration test for GenerationService billing-saga rollback (C5).

Reproduces the realistic failure shape: a provider's submit() writes a
GenerationJob row and reserves a debit *within the session*, then raises
before returning — exactly what AishaGenerationProvider does today when
ComfyUI accepts a request but the tunnel/queue response can't be parsed. Since
`job = await provider.submit(...)` never completes when submit() raises, the
orchestrator's local `job` stays None regardless of what was written inside
the (uncommitted) transaction.

Before the fix, the route's exception handling still returned a Response
normally on some paths, so the DI teardown's commit-on-clean-exit persisted
the orphaned job + debit with no refund. After the fix, `job is None` on
failure means the whole transaction is rolled back — this test proves that
against a real PostgreSQL session, not a mock.

Self-contained — GenerationService.generate() manages the session's
transaction lifecycle itself (commit on success, rollback/refund+commit on
failure), exactly like the production DI-provided session. We therefore bind
a bare session directly to the engine rather than wrapping it in the
SAVEPOINT-based `db_session` fixture (whose isolation depends on the *test*
owning commit/rollback, not the code under test) — same rationale as
test_partial_refund_concurrency.py.
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
from src.api.services.billing import BillingService
from src.api.services.generation.base import ProviderSubmitResult
from src.api.services.generation.rate_limiter import ModelRateLimiter
from src.api.services.generation.service import (
    GenerationError,
    GenerationService,
    ProviderResponseError,
)
from src.core.enums import AspectRatio, GenerationType, JobStatus, ModelType, Provider
from src.core.product_registry import VEX_CONFIG
from src.core.uid import new_id
from src.db.models.billing import TokenAccount, TokenTransaction
from src.db.models.gpu_session import GpuSession
from src.db.models.storage import GenerationJob
from src.db.models.user import User
from src.db.repositories.job import JobRepository

pytestmark = pytest.mark.asyncio


class _WriteThenFailProvider:
    """Mimics AishaGenerationProvider.submit(): writes a job row and reserves
    a debit *before* the failure, then raises rather than returning."""

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
            # ck_generation_jobs_aisha_has_session requires a gpu_session_id
            # whenever provider='aisha'.
            gpu_session_id=self._gpu_session_id,
        )
        await billing_service.check_and_reserve(
            account_id,
            token_cost,
            job_id,
            metadata={"type": "generation", "provider": "aisha"},
            description="Generation charge",
            session=session,
            product_id=product_id,
        )
        await session.flush()
        # Simulate ComfyUI accepting the request but returning something the
        # provider can't parse — exactly AishaGenerationProvider's real
        # "no prompt_id in response" path.
        raise ProviderResponseError("The generation backend returned an unexpected response.")


def _make_enabled_model_mock() -> MagicMock:
    record = MagicMock()
    record.is_enabled = True
    return record


async def _seed_user_and_account(
    engine: AsyncEngine, *, balance: int
) -> tuple[User, TokenAccount, GpuSession]:
    user = User(
        id=new_id(),
        email=f"saga-{uuid4().hex[:8]}@example.com",
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
        # Flush the user first and explicitly — GpuSession has no ORM
        # relationship() back to User (bare FK column only), and mixing it
        # into the same add_all() as the TokenAccount/TokenTransaction chain
        # was observed to insert gpu_sessions before users, violating the FK.
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


async def test_provider_raising_after_job_and_debit_write_rolls_back_fully(
    db_engine: AsyncEngine,
) -> None:
    """job is None on failure (submit() raised) ⇒ the whole transaction —
    including the job row and debit the provider wrote — must be rolled
    back. Assert zero net ledger charge and no orphaned job row afterward."""
    user, account, gpu_session = await _seed_user_and_account(db_engine, balance=1000)

    provider = _WriteThenFailProvider(gpu_session_id=gpu_session.id)
    pricing = AsyncMock()
    pricing.quote = AsyncMock(return_value=50)
    billing = BillingService()

    service = GenerationService(
        providers={Provider.AISHA: provider},
        billing_service=billing,
        pricing_service=pricing,
        rate_limiter=MagicMock(spec=ModelRateLimiter),
    )

    request = UnifiedGenerationRequest(
        prompt="A cat",
        generation_type=GenerationType.T2I,
        model=ModelType.AISHA_IMAGE,
        aspect_ratio=AspectRatio.RATIO_1_1,
    )

    try:
        # Bare session bound to the engine — GenerationService.generate()
        # manages its transaction lifecycle itself (commit/rollback), just
        # like the production DI-provided session. No outer session.begin();
        # that would fight with the internal rollback the code under test
        # performs.
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
                pytest.raises(GenerationError) as exc_info,
            ):
                await service.generate(
                    request,
                    user_id=user.id,
                    session=session,
                    product_config=VEX_CONFIG,
                )
        finally:
            await session.close()

        # Domain error message is re-raised as-is (safe by contract) — not wrapped.
        assert "unexpected response" in str(exc_info.value)
        assert provider.written_job_id is not None

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify_session:
            # The job row the provider wrote must not be visible — rolled
            # back, not left dangling in PENDING (or any) status.
            job_row = await verify_session.get(GenerationJob, provider.written_job_id)
            assert job_row is None, (
                "Job row survived the rollback — orphaned PENDING job bug is back"
            )

            # Zero net charge: rollback discarded the debit entirely (no
            # refund transaction needed because nothing was ever committed).
            final_balance = await billing.get_balance(account.id, session=verify_session)
            assert final_balance == 1000
    finally:
        await _cleanup(db_engine, account_id=account.id, user_id=user.id)
