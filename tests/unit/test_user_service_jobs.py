"""Unit tests for UserService.get_jobs — thumbnail_url population."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

from src.api.schemas.user import JobSummaryResponse, UserJobsResponse
from src.api.services.user import UserService
from src.core.enums import GenerationType, JobStatus
from src.db.repositories import UserRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(
    *,
    job_id: UUID | None = None,
    user_id: UUID | None = None,
    status: str = JobStatus.COMPLETED.value,
    generation_type: str = GenerationType.T2I.value,
    prompt: str = "a cat",
    name: str = "Test Job",
) -> MagicMock:
    job = MagicMock()
    job.id = job_id or uuid4()
    job.user_id = user_id or uuid4()
    job.status = status
    job.generation_type = generation_type
    job.prompt = prompt
    job.name = name
    job.created_at = datetime.now(UTC)
    job.completed_at = None
    return job


def _make_output(*, job_id: UUID, storage_key: str = "users/uid/outputs/job/file.jpg") -> MagicMock:
    out = MagicMock()
    out.id = uuid4()
    out.job_id = job_id
    out.storage_key = storage_key
    return out


def _make_repo(
    *,
    user_id: UUID,
    jobs: list,
    total: int | None = None,
    output_counts: dict | None = None,
    first_outputs: dict | None = None,
) -> AsyncMock:
    repo = AsyncMock(spec=UserRepository)
    repo.get_user.return_value = MagicMock(id=user_id)
    repo.list_user_jobs.return_value = jobs
    repo.count_user_jobs.return_value = total if total is not None else len(jobs)
    repo.count_outputs_for_jobs.return_value = output_counts or {j.id: 1 for j in jobs}
    repo.get_first_outputs_for_jobs.return_value = first_outputs or {}
    return repo


def _service(*, repo: AsyncMock, r2: AsyncMock | None = None) -> UserService:
    from src.api.security import PasswordService

    return UserService(repository=repo, password_service=PasswordService(), r2_storage=r2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetJobsThumbnailUrl:
    async def test_thumbnail_url_populated_from_sign_key(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        output = _make_output(job_id=job.id, storage_key="users/uid/outputs/j/img.jpg")

        r2 = AsyncMock()
        r2.sign_key = AsyncMock(return_value="https://r2.example.com/thumb.jpg")

        repo = _make_repo(
            user_id=user_id,
            jobs=[job],
            first_outputs={job.id: output},
        )

        result = await _service(repo=repo, r2=r2).get_jobs(user_id)

        assert isinstance(result, UserJobsResponse)
        assert len(result.items) == 1
        item = result.items[0]
        assert isinstance(item, JobSummaryResponse)
        assert item.thumbnail_url == "https://r2.example.com/thumb.jpg"
        r2.sign_key.assert_awaited_once_with(output.storage_key)

    async def test_thumbnail_url_none_when_no_outputs(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)

        r2 = AsyncMock()
        r2.sign_key = AsyncMock(return_value="https://r2.example.com/thumb.jpg")

        repo = _make_repo(
            user_id=user_id,
            jobs=[job],
            output_counts={job.id: 0},
            first_outputs={},  # no outputs for this job
        )

        result = await _service(repo=repo, r2=r2).get_jobs(user_id)

        assert result.items[0].thumbnail_url is None
        r2.sign_key.assert_not_awaited()

    async def test_thumbnail_url_none_when_r2_not_configured(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        output = _make_output(job_id=job.id)

        repo = _make_repo(
            user_id=user_id,
            jobs=[job],
            first_outputs={job.id: output},
        )

        result = await _service(repo=repo, r2=None).get_jobs(user_id)

        assert result.items[0].thumbnail_url is None

    async def test_thumbnail_url_none_when_sign_key_raises(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        output = _make_output(job_id=job.id)

        r2 = AsyncMock()
        r2.sign_key = AsyncMock(side_effect=Exception("R2 unavailable"))

        repo = _make_repo(
            user_id=user_id,
            jobs=[job],
            first_outputs={job.id: output},
        )

        result = await _service(repo=repo, r2=r2).get_jobs(user_id)

        # sign failure is swallowed; job still returned with thumbnail_url=None
        assert len(result.items) == 1
        assert result.items[0].thumbnail_url is None

    async def test_multiple_jobs_all_get_thumbnail_urls(self) -> None:
        user_id = uuid4()
        jobs = [_make_job(user_id=user_id) for _ in range(3)]
        outputs = {job.id: _make_output(job_id=job.id) for job in jobs}
        signed_url = "https://r2.example.com/img.jpg"

        r2 = AsyncMock()
        r2.sign_key = AsyncMock(return_value=signed_url)

        repo = _make_repo(
            user_id=user_id,
            jobs=jobs,
            first_outputs=outputs,
        )

        result = await _service(repo=repo, r2=r2).get_jobs(user_id)

        assert len(result.items) == 3
        for item in result.items:
            assert item.thumbnail_url == signed_url
        assert r2.sign_key.await_count == 3

    async def test_partial_sign_failure_only_nulls_failed_job(self) -> None:
        user_id = uuid4()
        job_good = _make_job(user_id=user_id)
        job_bad = _make_job(user_id=user_id)
        good_key = "users/uid/outputs/good/img.jpg"
        bad_key = "users/uid/outputs/bad/img.jpg"

        r2 = AsyncMock()

        async def sign_side_effect(key: str) -> str:
            if key == bad_key:
                raise Exception("R2 failure")
            return "https://r2.example.com/good.jpg"

        r2.sign_key = AsyncMock(side_effect=sign_side_effect)

        repo = _make_repo(
            user_id=user_id,
            jobs=[job_good, job_bad],
            first_outputs={
                job_good.id: _make_output(job_id=job_good.id, storage_key=good_key),
                job_bad.id: _make_output(job_id=job_bad.id, storage_key=bad_key),
            },
        )

        result = await _service(repo=repo, r2=r2).get_jobs(user_id)

        assert len(result.items) == 2
        items_by_id = {UUID(item.id): item for item in result.items}
        assert items_by_id[job_good.id].thumbnail_url == "https://r2.example.com/good.jpg"
        assert items_by_id[job_bad.id].thumbnail_url is None

    async def test_get_jobs_returns_correct_output_count(self) -> None:
        user_id = uuid4()
        job = _make_job(user_id=user_id)
        output = _make_output(job_id=job.id)

        repo = _make_repo(
            user_id=user_id,
            jobs=[job],
            output_counts={job.id: 4},
            first_outputs={job.id: output},
        )

        result = await _service(repo=repo, r2=None).get_jobs(user_id)

        assert result.items[0].output_count == 4

    async def test_get_jobs_total_matches_count_query(self) -> None:
        user_id = uuid4()
        repo = _make_repo(user_id=user_id, jobs=[], total=42)

        result = await _service(repo=repo, r2=None).get_jobs(user_id)

        assert result.total == 42

    async def test_prompt_truncated_to_200_chars(self) -> None:
        user_id = uuid4()
        long_prompt = "x" * 300
        job = _make_job(user_id=user_id, prompt=long_prompt)

        repo = _make_repo(user_id=user_id, jobs=[job], first_outputs={})

        result = await _service(repo=repo, r2=None).get_jobs(user_id)

        assert result.items[0].prompt == f"{'x' * 200}..."

    async def test_prompt_under_limit_not_truncated(self) -> None:
        user_id = uuid4()
        short_prompt = "a cat"
        job = _make_job(user_id=user_id, prompt=short_prompt)

        repo = _make_repo(user_id=user_id, jobs=[job], first_outputs={})

        result = await _service(repo=repo, r2=None).get_jobs(user_id)

        assert result.items[0].prompt == short_prompt
