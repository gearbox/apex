"""Integration tests for JobRepository against a real PostgreSQL database."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from src.core.enums import GenerationType, GpuSessionStatus, JobStatus, Provider
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


# -------------------------------------------------------------------------
# list_by_user (cursor pagination)
# -------------------------------------------------------------------------


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


async def test_list_by_user_cursor_pagination(job_repo: JobRepository, make_user, make_job) -> None:
    """list_by_user supports cursor-based pagination with limit+1 pattern."""
    user = await make_user(email=f"cursorjobs-{uuid4().hex[:6]}@example.com")
    # Create 5 jobs
    for i in range(5):
        await make_job(user=user, name=f"Job {i}")

    # First page: limit=2 → expect 3 rows (limit+1 pattern)
    page1 = await job_repo.list_by_user(user.id, limit=2)
    assert len(page1) == 3  # limit+1

    # Trim to page size and use last item as cursor
    items = list(page1)[:2]
    last = items[-1]

    # Second page using cursor
    page2 = await job_repo.list_by_user(
        user.id,
        limit=2,
        cursor_ts=last.created_at,
        cursor_id=last.id,
    )
    assert len(page2) >= 1

    # Verify no overlap between pages
    page1_ids = {j.id for j in items}
    page2_ids = {j.id for j in page2}
    assert page1_ids.isdisjoint(page2_ids)


async def test_list_by_user_empty(job_repo: JobRepository, make_user) -> None:
    """list_by_user returns empty list for user with no jobs."""
    user = await make_user(email=f"nojobs-{uuid4().hex[:6]}@example.com")
    jobs = await job_repo.list_by_user(user.id)
    assert not list(jobs)


async def test_list_by_user_filter_by_generation_type(
    job_repo: JobRepository, make_user, make_job
) -> None:
    """list_by_user filters by generation_type when provided."""
    user = await make_user(email=f"gentypejobs-{uuid4().hex[:6]}@example.com")
    await make_job(user=user, generation_type="t2i")
    await make_job(user=user, generation_type="t2v")

    t2i_jobs = await job_repo.list_by_user(user.id, generation_type=GenerationType.T2I)
    assert all(j.generation_type == "t2i" for j in t2i_jobs)
    assert len(list(t2i_jobs)) == 1


# -------------------------------------------------------------------------
# list_pending_video_jobs
# -------------------------------------------------------------------------


async def test_list_pending_video_jobs_returns_matching(
    job_repo: JobRepository, make_user, make_job, make_gpu_session
) -> None:
    """list_pending_video_jobs returns queued/running video jobs with external_request_id."""
    user = await make_user(email=f"vidpoll-{uuid4().hex[:6]}@example.com")

    # Matching: queued t2v job with external_request_id
    matching_t2v = await make_job(
        user=user,
        status="queued",
        generation_type="t2v",
        provider=Provider.GROK.value,
        external_request_id="grok-req-001",
    )

    # Matching: running i2v job
    matching_i2v = await make_job(
        user=user,
        status="running",
        generation_type="i2v",
        provider=Provider.GROK.value,
        external_request_id="grok-req-002",
    )

    # Non-matching: wrong provider (aisha requires a gpu_session)
    gpu_session = await make_gpu_session(user=user)
    await make_job(
        user=user,
        status="queued",
        generation_type="t2v",
        provider=Provider.AISHA.value,
        external_request_id="aisha-req-001",
        gpu_session_id=gpu_session.id,
    )

    # Non-matching: completed status
    await make_job(
        user=user,
        status="completed",
        generation_type="t2v",
        provider=Provider.GROK.value,
        external_request_id="grok-req-done",
    )

    # Non-matching: image generation type (t2i)
    await make_job(
        user=user,
        status="queued",
        generation_type="t2i",
        provider=Provider.GROK.value,
        external_request_id="grok-req-img",
    )

    # Non-matching: no external_request_id
    await make_job(
        user=user,
        status="queued",
        generation_type="t2v",
        provider=Provider.GROK.value,
        external_request_id=None,
    )

    jobs = await job_repo.list_pending_video_jobs()
    job_ids = {j.id for j in jobs}

    assert matching_t2v.id in job_ids
    assert matching_i2v.id in job_ids
    assert len(job_ids) == 2


async def test_list_pending_video_jobs_empty_when_no_matches(
    job_repo: JobRepository, make_user, make_job
) -> None:
    """list_pending_video_jobs returns empty when no jobs match criteria."""
    user = await make_user(email=f"vidpoll-empty-{uuid4().hex[:6]}@example.com")

    await make_job(
        user=user,
        status="completed",
        generation_type="t2v",
        provider=Provider.GROK.value,
        external_request_id="grok-done",
    )
    await make_job(
        user=user,
        status="queued",
        generation_type="t2i",
        provider=Provider.GROK.value,
        external_request_id="grok-img",
    )
    await make_job(
        user=user,
        status="queued",
        generation_type="t2v",
        provider=Provider.GROK.value,
        external_request_id=None,
    )

    jobs = await job_repo.list_pending_video_jobs()
    assert not list(jobs)


# -------------------------------------------------------------------------
# soft-delete (is_deleted) filtering
# -------------------------------------------------------------------------


async def test_list_by_user_excludes_soft_deleted_jobs(
    job_repo: JobRepository, make_user, make_job
) -> None:
    """list_by_user excludes jobs with is_deleted=True."""
    user = await make_user(email=f"deleted-{uuid4().hex[:6]}@example.com")
    visible = await make_job(user=user)
    deleted = await make_job(user=user)

    await job_repo.soft_delete(deleted.id, user_id=user.id)

    jobs = await job_repo.list_by_user(user.id)
    job_ids = {j.id for j in jobs}

    assert visible.id in job_ids
    assert deleted.id not in job_ids


async def test_get_with_user_id_excludes_soft_deleted_job(
    job_repo: JobRepository, make_user, make_job
) -> None:
    """get() with user_id returns None for a soft-deleted job."""
    user = await make_user(email=f"getdel-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)

    await job_repo.soft_delete(job.id, user_id=user.id)

    result = await job_repo.get(job.id, user_id=user.id)
    assert result is None


async def test_get_without_user_id_includes_soft_deleted_job(
    job_repo: JobRepository, make_user, make_job
) -> None:
    """get() without user_id (internal) returns soft-deleted jobs."""
    user = await make_user(email=f"getdelint-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)

    await job_repo.soft_delete(job.id, user_id=user.id)

    result = await job_repo.get(job.id)
    assert result is not None
    assert result.is_deleted is True


async def test_soft_delete_sets_is_deleted(job_repo: JobRepository, make_user, make_job) -> None:
    """soft_delete sets is_deleted=True on the job."""
    user = await make_user(email=f"softdel-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)

    result = await job_repo.soft_delete(job.id, user_id=user.id)

    assert result is not None
    assert result.is_deleted is True


async def test_soft_delete_preserves_error_message(
    job_repo: JobRepository, make_user, make_job, db_session
) -> None:
    """soft_delete does not clobber an existing error_message."""
    user = await make_user(email=f"softdelerr-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user, status="failed")
    job.error_message = "GPU OOM"
    await db_session.flush()

    result = await job_repo.soft_delete(job.id, user_id=user.id)

    assert result is not None
    assert result.is_deleted is True
    assert result.error_message == "GPU OOM"


async def test_soft_delete_wrong_user_returns_none(
    job_repo: JobRepository, make_user, make_job
) -> None:
    """soft_delete returns None when user doesn't own the job."""
    user = await make_user(email=f"softdelowner-{uuid4().hex[:6]}@example.com")
    other = await make_user(email=f"softdelother-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)

    result = await job_repo.soft_delete(job.id, user_id=other.id)

    assert result is None


async def test_soft_delete_unknown_returns_none(job_repo: JobRepository) -> None:
    """soft_delete returns None for unknown job ID."""
    assert await job_repo.soft_delete(uuid4(), user_id=uuid4()) is None


async def test_soft_delete_idempotent(job_repo: JobRepository, make_user, make_job) -> None:
    """soft_delete on already-deleted job succeeds (idempotent)."""
    user = await make_user(email=f"softdelidem-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)

    first = await job_repo.soft_delete(job.id, user_id=user.id)
    second = await job_repo.soft_delete(job.id, user_id=user.id)

    assert first is not None
    assert second is not None
    assert second.is_deleted is True


# -------------------------------------------------------------------------
# list_aisha_jobs_for_polling
# -------------------------------------------------------------------------


async def test_list_aisha_jobs_for_polling_filters_correctly(
    job_repo: JobRepository,
    make_user,
    make_job,
    make_gpu_session,
) -> None:
    """Returns QUEUED/RUNNING Aisha jobs whose session is active and not deleted."""
    user = await make_user(email=f"aisha-poll-{uuid4().hex[:6]}@example.com")
    active = await make_gpu_session(user=user, status=GpuSessionStatus.active)
    stopping = await make_gpu_session(
        user=user, status=GpuSessionStatus.stopping, model_type="aisha-video"
    )

    matching_queued = await make_job(
        user=user,
        provider=Provider.AISHA.value,
        status=JobStatus.QUEUED.value,
        gpu_session_id=active.id,
    )
    matching_running = await make_job(
        user=user,
        provider=Provider.AISHA.value,
        status=JobStatus.RUNNING.value,
        gpu_session_id=active.id,
    )

    # Non-matching: terminal status
    await make_job(
        user=user,
        provider=Provider.AISHA.value,
        status=JobStatus.COMPLETED.value,
        gpu_session_id=active.id,
    )
    # Non-matching: non-active session
    await make_job(
        user=user,
        provider=Provider.AISHA.value,
        status=JobStatus.QUEUED.value,
        gpu_session_id=stopping.id,
    )
    # Non-matching: soft-deleted
    await make_job(
        user=user,
        provider=Provider.AISHA.value,
        status=JobStatus.QUEUED.value,
        gpu_session_id=active.id,
        is_deleted=True,
    )
    # Non-matching: Grok job (no gpu_session)
    await make_job(
        user=user,
        provider=Provider.GROK.value,
        status=JobStatus.QUEUED.value,
        external_request_id="grok-1",
    )

    result = await job_repo.list_aisha_jobs_for_polling()
    ids = {j.id for j in result}

    assert matching_queued.id in ids
    assert matching_running.id in ids
    assert len(ids) == 2


async def test_list_aisha_jobs_for_polling_eager_loads_gpu_session(
    job_repo: JobRepository,
    make_user,
    make_job,
    make_gpu_session,
) -> None:
    """gpu_session relationship is eagerly loaded; access does not lazy-load."""
    user = await make_user(email=f"aisha-eager-{uuid4().hex[:6]}@example.com")
    gpu_session = await make_gpu_session(
        user=user,
        status=GpuSessionStatus.active,
        tunnel_hostname="node1.gpu-domain.com",
    )
    job = await make_job(
        user=user,
        provider=Provider.AISHA.value,
        status=JobStatus.QUEUED.value,
        gpu_session_id=gpu_session.id,
    )

    result = await job_repo.list_aisha_jobs_for_polling()
    [polled] = [j for j in result if j.id == job.id]

    # If selectinload is missing, this raises in async context because
    # the relationship is configured with lazy="raise_on_sql".
    assert polled.gpu_session is not None
    assert polled.gpu_session.id == gpu_session.id
    assert polled.gpu_session.tunnel_hostname == gpu_session.tunnel_hostname


async def test_list_aisha_jobs_for_polling_empty(
    job_repo: JobRepository,
    make_user,
) -> None:
    """Returns empty list when no jobs match."""
    await make_user(email=f"aisha-empty-{uuid4().hex[:6]}@example.com")
    result = await job_repo.list_aisha_jobs_for_polling()
    assert not list(result)


async def test_list_pending_video_jobs_custom_provider(
    job_repo: JobRepository, make_user, make_job, make_gpu_session
) -> None:
    """list_pending_video_jobs filters by the given provider enum."""
    user = await make_user(email=f"vidpoll-prov-{uuid4().hex[:6]}@example.com")

    gpu_session = await make_gpu_session(user=user)
    aisha_job = await make_job(
        user=user,
        status="queued",
        generation_type="t2v",
        provider=Provider.AISHA.value,
        external_request_id="aisha-vid-001",
        gpu_session_id=gpu_session.id,
    )

    # Grok job — should not appear when filtering by AISHA
    await make_job(
        user=user,
        status="queued",
        generation_type="t2v",
        provider=Provider.GROK.value,
        external_request_id="grok-vid-001",
    )

    jobs = await job_repo.list_pending_video_jobs(provider=Provider.AISHA)
    job_ids = {j.id for j in jobs}

    assert aisha_job.id in job_ids
    assert len(job_ids) == 1
