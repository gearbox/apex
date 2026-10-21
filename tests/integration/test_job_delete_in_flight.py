"""Z1: an in-flight job cannot be soft-deleted (real PostgreSQL).

``DELETE /v1/jobs/{id}`` on a ``pending``/``queued``/``running`` job is refused
(409 ``job_in_progress``) and the database forbids the state outright
(``ck_generation_jobs_not_deleted_in_flight``), so no lifecycle query can ever
lose track of a job whose upstream work is accepted and charged.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.jobs import UnifiedJobController
from src.api.services.unified_jobs import JOB_IN_PROGRESS_MESSAGE, JobInProgressError
from src.api.services.video_jobs.worker import AsyncVideoJobWorker
from src.core.enums import JobStatus, Provider, SoftDeleteOutcome
from src.db.models.storage import GenerationJob
from src.db.repositories.job import JobRepository
from tests.integration.test_remote_video_materialization import _cleanup, _seed

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine

_IN_FLIGHT = ["pending", "queued", "running"]
_FINISHED = ["completed", "failed", "cancelled", "moderated"]


class _DB:
    """Minimal ``DatabaseManager.session()`` over an engine."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with AsyncSession(bind=self._engine, expire_on_commit=False) as session:
            yield session


async def _row(engine: AsyncEngine, job_id: UUID) -> tuple[str, bool]:
    async with AsyncSession(bind=engine) as session:
        result = await session.execute(
            select(GenerationJob.status, GenerationJob.is_deleted).where(GenerationJob.id == job_id)
        )
        status, is_deleted = result.one()
        return str(status), bool(is_deleted)


async def _set_status(engine: AsyncEngine, job_id: UUID, status: str) -> None:
    async with AsyncSession(bind=engine) as session:
        await session.execute(
            update(GenerationJob).where(GenerationJob.id == job_id).values(status=status)
        )
        await session.commit()


async def _delete(engine: AsyncEngine, job_id: UUID, user_id: UUID) -> SoftDeleteOutcome:
    async with AsyncSession(bind=engine) as session:
        outcome = await JobRepository(session).soft_delete(job_id, user_id=user_id)
        await session.commit()
        return outcome


@pytest.mark.parametrize("status", _IN_FLIGHT)
async def test_deleting_an_in_flight_job_is_refused_and_leaves_it_untouched(
    db_engine: AsyncEngine, status: str
) -> None:
    seed = await _seed(db_engine, Provider.ALIBABA)
    try:
        await _set_status(db_engine, seed.job_id, status)

        assert await _delete(db_engine, seed.job_id, seed.user_id) is SoftDeleteOutcome.IN_FLIGHT

        assert await _row(db_engine, seed.job_id) == (status, False)
    finally:
        await _cleanup(db_engine, seed)


@pytest.mark.parametrize("status", _FINISHED)
async def test_deleting_a_finished_job_hides_it_and_is_idempotent(
    db_engine: AsyncEngine, status: str
) -> None:
    seed = await _seed(db_engine, Provider.ALIBABA)
    try:
        await _set_status(db_engine, seed.job_id, status)

        assert await _delete(db_engine, seed.job_id, seed.user_id) is SoftDeleteOutcome.DELETED
        assert await _delete(db_engine, seed.job_id, seed.user_id) is SoftDeleteOutcome.DELETED

        assert await _row(db_engine, seed.job_id) == (status, True)
        async with AsyncSession(bind=db_engine) as session:
            assert await JobRepository(session).get(seed.job_id, user_id=seed.user_id) is None
    finally:
        await _cleanup(db_engine, seed)


async def test_another_users_job_is_not_found(db_engine: AsyncEngine) -> None:
    seed = await _seed(db_engine, Provider.ALIBABA)
    other = await _seed(db_engine, Provider.ALIBABA)
    try:
        assert await _delete(db_engine, seed.job_id, other.user_id) is SoftDeleteOutcome.NOT_FOUND
        assert await _row(db_engine, seed.job_id) == ("running", False)
    finally:
        await _cleanup(db_engine, seed)
        await _cleanup(db_engine, other)


async def test_the_route_refuses_an_in_flight_job_with_job_in_progress(
    db_engine: AsyncEngine,
) -> None:
    seed = await _seed(db_engine, Provider.ALIBABA)
    try:
        async with AsyncSession(bind=db_engine) as session:
            with pytest.raises(JobInProgressError) as exc_info:
                await UnifiedJobController.delete_job.fn(
                    MagicMock(),
                    current_user_id=seed.user_id,
                    job_id=seed.job_id,
                    session=session,
                )
        assert str(exc_info.value) == JOB_IN_PROGRESS_MESSAGE
        assert exc_info.value.status == "running"
        assert await _row(db_engine, seed.job_id) == ("running", False)
    finally:
        await _cleanup(db_engine, seed)


async def test_the_database_forbids_a_deleted_in_flight_job(db_engine: AsyncEngine) -> None:
    seed = await _seed(db_engine, Provider.ALIBABA)
    try:
        async with AsyncSession(bind=db_engine) as session:
            with pytest.raises(IntegrityError, match="ck_generation_jobs_not_deleted_in_flight"):
                await session.execute(
                    text("UPDATE generation_jobs SET is_deleted = true WHERE id = :id"),
                    {"id": seed.job_id},
                )
    finally:
        await _cleanup(db_engine, seed)


async def test_a_delete_racing_a_completion_leaves_one_consistent_state(
    db_engine: AsyncEngine,
) -> None:
    """DELETE against a row whose ``running -> completed`` is still uncommitted.

    The conditional ``UPDATE`` evaluates the committed (``running``) version, so
    the delete is refused in one statement: there is no read-then-write window in
    which a ``running`` job could be marked deleted. After the completion commits,
    the same delete succeeds. The only reachable end states are consistent.
    """
    seed = await _seed(db_engine, Provider.ALIBABA)
    try:
        async with AsyncSession(bind=db_engine) as completing:
            await completing.execute(
                update(GenerationJob)
                .where(GenerationJob.id == seed.job_id)
                .values(status=JobStatus.COMPLETED.value)
            )
            raced = await _delete(db_engine, seed.job_id, seed.user_id)
            await completing.commit()

        assert raced is SoftDeleteOutcome.IN_FLIGHT
        assert await _row(db_engine, seed.job_id) == ("completed", False)

        assert await _delete(db_engine, seed.job_id, seed.user_id) is SoftDeleteOutcome.DELETED
        assert await _row(db_engine, seed.job_id) == ("completed", True)
    finally:
        await _cleanup(db_engine, seed)


@pytest.mark.parametrize("provider", [Provider.ALIBABA, Provider.GROK])
async def test_an_in_flight_job_keeps_being_polled_after_a_refused_delete(
    db_engine: AsyncEngine, provider: Provider
) -> None:
    """Before Z1 a deleted running job was never polled again and its debit stayed forever."""
    seed = await _seed(db_engine, provider, with_debit=True)
    try:
        service = MagicMock(provider=provider, VIDEO_POLLER_NAME="probe", transient_poll_errors=())
        service.poll_video_job_for_worker = AsyncMock()
        worker = AsyncVideoJobWorker(
            job_service=service,
            interval_seconds=5,
            max_concurrent_polls=2,
            db_manager=_DB(db_engine),  # type: ignore[arg-type]
            redis_client_factory=MagicMock(),
        )
        await worker.run_once()
        polls_before = service.poll_video_job_for_worker.await_count
        assert polls_before == 1

        assert await _delete(db_engine, seed.job_id, seed.user_id) is SoftDeleteOutcome.IN_FLIGHT
        await worker.run_once()

        assert service.poll_video_job_for_worker.await_count == polls_before + 1
        async with AsyncSession(bind=db_engine) as session:
            job = await JobRepository(session).get(seed.job_id, user_id=seed.user_id)
            assert job is not None  # poll-on-read still reaches it
            assert job.status == "running"
            assert job.debit_transaction_id is not None
    finally:
        await _cleanup(db_engine, seed)
