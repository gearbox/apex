"""Integration tests for JobRepository against a real PostgreSQL database."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from src.core.enums import GenerationType, JobStatus
from src.db.repositories.job import JobRepository


async def test_create_job(job_repo: JobRepository, make_user) -> None:
    """create persists and returns a GenerationJob in pending status."""
    user = await make_user(email=f"jobcreate-{uuid4().hex[:6]}@example.com")
    job = await job_repo.create(
        id=uuid4(),
        user_id=user.id,
        name="My Job",
        prompt="A beautiful landscape",
        generation_type=GenerationType.T2I,
        product_id="vex",
    )
    assert job.status == "pending"
    assert job.prompt == "A beautiful landscape"


async def test_get_job_found(job_repo: JobRepository, make_job) -> None:
    """get returns the job by PK."""
    job = await make_job()
    found = await job_repo.get(job.id)
    assert found is not None
    assert found.id == job.id


async def test_get_job_not_found(job_repo: JobRepository) -> None:
    """get returns None for unknown UUID."""
    assert await job_repo.get(uuid4()) is None


async def test_get_job_ownership_enforced(job_repo: JobRepository, make_job, make_user) -> None:
    """get with wrong user_id returns None."""
    job = await make_job()
    other = await make_user(email=f"otherjob-{uuid4().hex[:6]}@example.com")
    found = await job_repo.get(job.id, user_id=other.id)
    assert found is None


async def test_update_job_status(job_repo: JobRepository, make_job) -> None:
    """update_status changes the job status."""
    job = await make_job(status="pending")
    updated = await job_repo.update_status(job.id, JobStatus.RUNNING)
    assert updated is not None
    assert updated.status == JobStatus.RUNNING


async def test_update_job_status_unknown_returns_none(
    job_repo: JobRepository,
) -> None:
    """update_status returns None for an unknown job ID."""
    assert await job_repo.update_status(uuid4(), JobStatus.RUNNING) is None


async def test_update_job_status_with_started_at(job_repo: JobRepository, make_job) -> None:
    """update_status persists started_at timestamp."""
    job = await make_job()
    started = datetime.now(UTC)
    updated = await job_repo.update_status(job.id, JobStatus.RUNNING, started_at=started)
    assert updated is not None
    assert updated.started_at is not None


async def test_update_job_status_with_completed_at(job_repo: JobRepository, make_job) -> None:
    """update_status persists completed_at timestamp."""
    job = await make_job()
    completed = datetime.now(UTC)
    updated = await job_repo.update_status(job.id, JobStatus.COMPLETED, completed_at=completed)
    assert updated is not None
    assert updated.completed_at is not None


async def test_list_by_user_returns_owned_jobs(
    job_repo: JobRepository, make_user, make_job
) -> None:
    """list_by_user returns only jobs for the specified user."""
    user = await make_user(email=f"listjobs-{uuid4().hex[:6]}@example.com")
    for _ in range(3):
        await make_job(user=user)
    jobs = await job_repo.list_by_user(user.id)
    assert len(jobs) >= 3
    assert all(j.user_id == user.id for j in jobs)


async def test_list_by_user_filter_by_status(job_repo: JobRepository, make_user, make_job) -> None:
    """list_by_user filters by status when provided."""
    user = await make_user(email=f"statusjobs-{uuid4().hex[:6]}@example.com")
    await make_job(user=user, status="pending")
    await make_job(user=user, status="completed")
    pending_jobs = await job_repo.list_by_user(user.id, status=JobStatus.PENDING)
    assert all(j.status == "pending" for j in pending_jobs)


async def test_list_by_user_offset_past_end(job_repo: JobRepository, make_user) -> None:
    """list_by_user with offset beyond count returns empty list."""
    user = await make_user(email=f"nojobsoff-{uuid4().hex[:6]}@example.com")
    jobs = await job_repo.list_by_user(user.id, offset=1000)
    assert not list(jobs)
