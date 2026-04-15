"""Integration tests for GalleryRepository against a real database."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import GenerationType, OutputMediaType
from src.db.models.storage import GenerationJob, GenerationOutput
from src.db.models.user import User
from src.db.repositories.gallery import GalleryRepository

OutputFactory = Callable[..., Coroutine[Any, Any, GenerationOutput]]


@pytest_asyncio.fixture
async def gallery_repo(db_session: AsyncSession) -> GalleryRepository:
    """GalleryRepository bound to the test session."""
    return GalleryRepository(db_session)


@pytest_asyncio.fixture
async def make_output(
    db_session: AsyncSession,
    make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    make_user: Callable[..., Coroutine[Any, Any, User]],
) -> OutputFactory:
    """Factory fixture: create a GenerationOutput row and flush it."""

    async def _factory(
        *,
        job: GenerationJob | None = None,
        user: User | None = None,
        content_type: str = "image/jpeg",
        is_thumbnail: bool = False,
        output_index: int = 0,
        size_bytes: int = 1000,
        product_id: str = "vex",
        output_id: object = None,
    ) -> GenerationOutput:
        if user is None:
            user = await make_user(email=f"out-{uuid4().hex[:8]}@example.com")
        if job is None:
            job = await make_job(user=user, status="completed", product_id=product_id)
        file_ext = "jpeg" if content_type == "image/jpeg" else "mp4"
        oid = output_id or uuid4()
        out = GenerationOutput(
            id=oid,
            user_id=user.id,
            job_id=job.id,
            storage_key=f"users/{user.id}/outputs/{job.id}/{oid}.{file_ext}",
            content_type=content_type,
            size_bytes=size_bytes,
            format=file_ext,
            output_index=output_index,
            expires_at=datetime.now(UTC) + timedelta(days=7),
            is_thumbnail=is_thumbnail,
            product_id=product_id,
        )
        db_session.add(out)
        await db_session.flush()
        return out

    return _factory


# ---------------------------------------------------------------------------
# list_gallery_jobs
# ---------------------------------------------------------------------------


class TestListGalleryJobs:
    async def test_returns_completed_jobs(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        user = await make_user(email=f"gal-{uuid4().hex[:8]}@example.com")
        job = await make_job(user=user, status="completed", generation_type="t2i")

        rows = await gallery_repo.list_gallery_jobs(user.id, "vex")
        job_ids = [r.id for r in rows]
        assert job.id in job_ids

    async def test_excludes_pending_jobs(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        user = await make_user(email=f"gal-{uuid4().hex[:8]}@example.com")
        pending = await make_job(user=user, status="pending", generation_type="t2i")

        rows = await gallery_repo.list_gallery_jobs(user.id, "vex")
        job_ids = [r.id for r in rows]
        assert pending.id not in job_ids

    async def test_excludes_other_user_jobs(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        user_a = await make_user(email=f"a-{uuid4().hex[:8]}@example.com")
        user_b = await make_user(email=f"b-{uuid4().hex[:8]}@example.com")
        await make_job(user=user_b, status="completed", generation_type="t2i")

        rows = await gallery_repo.list_gallery_jobs(user_a.id, "vex")
        # user_a should see zero rows (no jobs for user_a yet)
        for r in rows:
            assert r.user_id == user_a.id

    async def test_excludes_other_product(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        user = await make_user(
            email=f"prod-{uuid4().hex[:8]}@example.com",
            product_id="synthara",
        )
        await make_job(user=user, status="completed", product_id="synthara")

        rows = await gallery_repo.list_gallery_jobs(user.id, "vex")
        assert all(r.product_id == "vex" for r in rows)

    async def test_media_type_video_filter(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        user = await make_user(email=f"vid-{uuid4().hex[:8]}@example.com")
        await make_job(user=user, status="completed", generation_type="t2i")
        vid_job = await make_job(user=user, status="completed", generation_type="t2v")

        rows = await gallery_repo.list_gallery_jobs(
            user.id, "vex", media_type=OutputMediaType.VIDEO
        )
        job_ids = [r.id for r in rows]
        assert vid_job.id in job_ids
        for row in rows:
            assert GenerationType(row.generation_type).is_video

    async def test_media_type_image_filter(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        user = await make_user(email=f"img-{uuid4().hex[:8]}@example.com")
        img_job = await make_job(user=user, status="completed", generation_type="t2i")
        await make_job(user=user, status="completed", generation_type="t2v")

        rows = await gallery_repo.list_gallery_jobs(
            user.id, "vex", media_type=OutputMediaType.IMAGE
        )
        job_ids = [r.id for r in rows]
        assert img_job.id in job_ids
        for row in rows:
            assert not GenerationType(row.generation_type).is_video

    async def test_generation_type_filter(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        user = await make_user(email=f"gt-{uuid4().hex[:8]}@example.com")
        i2i = await make_job(user=user, status="completed", generation_type="i2i")
        await make_job(user=user, status="completed", generation_type="t2i")

        rows = await gallery_repo.list_gallery_jobs(
            user.id, "vex", generation_type=GenerationType.I2I
        )
        for row in rows:
            assert row.generation_type == "i2i"
        assert i2i.id in [r.id for r in rows]

    async def test_model_filter(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        user = await make_user(email=f"mdl-{uuid4().hex[:8]}@example.com")
        target = await make_job(user=user, status="completed", model="grok-imagine-image")
        await make_job(user=user, status="completed", model="grok-2-image-1212")

        rows = await gallery_repo.list_gallery_jobs(user.id, "vex", model="grok-imagine-image")
        for row in rows:
            assert row.model == "grok-imagine-image"
        assert target.id in [r.id for r in rows]

    async def test_limit_plus_one_pattern(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        user = await make_user(email=f"lim-{uuid4().hex[:8]}@example.com")
        for _ in range(3):
            await make_job(user=user, status="completed", generation_type="t2i")

        rows = await gallery_repo.list_gallery_jobs(user.id, "vex", limit=2)
        # With limit=2 and 3 matching rows, should get 3 (limit+1)
        assert len(rows) == 3

    async def test_cursor_pagination(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        user = await make_user(email=f"cur-{uuid4().hex[:8]}@example.com")
        jobs = []
        for _ in range(3):
            j = await make_job(user=user, status="completed", generation_type="t2i")
            jobs.append(j)

        page1 = await gallery_repo.list_gallery_jobs(user.id, "vex", limit=2)
        assert len(page1) <= 3  # at most limit+1

        if len(page1) >= 2:
            last = page1[1]
            page2 = await gallery_repo.list_gallery_jobs(
                user.id,
                "vex",
                limit=2,
                cursor_ts=last.created_at,
                cursor_id=last.id,
            )
            page1_ids = {r.id for r in page1[:2]}
            for row in page2[:2]:
                assert row.id not in page1_ids


# ---------------------------------------------------------------------------
# get_gallery_job
# ---------------------------------------------------------------------------


class TestGetGalleryJob:
    async def test_returns_completed_job(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        user = await make_user(email=f"detail-{uuid4().hex[:8]}@example.com")
        job = await make_job(user=user, status="completed", generation_type="t2i")

        result = await gallery_repo.get_gallery_job(job.id, user.id, "vex")
        assert result is not None
        assert result.id == job.id

    async def test_returns_none_for_pending(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        user = await make_user(email=f"pend-{uuid4().hex[:8]}@example.com")
        job = await make_job(user=user, status="pending", generation_type="t2i")

        result = await gallery_repo.get_gallery_job(job.id, user.id, "vex")
        assert result is None

    async def test_returns_none_for_wrong_user(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        owner = await make_user(email=f"own-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"oth-{uuid4().hex[:8]}@example.com")
        job = await make_job(user=owner, status="completed")

        result = await gallery_repo.get_gallery_job(job.id, other.id, "vex")
        assert result is None

    async def test_returns_none_for_wrong_product(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    ) -> None:
        user = await make_user(
            email=f"wprod-{uuid4().hex[:8]}@example.com",
            product_id="synthara",
        )
        job = await make_job(user=user, status="completed", product_id="synthara")

        result = await gallery_repo.get_gallery_job(job.id, user.id, "vex")
        assert result is None


# ---------------------------------------------------------------------------
# batch_cover_data
# ---------------------------------------------------------------------------


class TestBatchCoverData:
    async def test_empty_input(self, gallery_repo: GalleryRepository) -> None:
        result = await gallery_repo.batch_cover_data([])
        assert result == {}

    async def test_counts_non_thumbnail_outputs(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
        make_output: OutputFactory,
    ) -> None:
        user = await make_user(email=f"bcd-{uuid4().hex[:8]}@example.com")
        job = await make_job(user=user, status="completed")
        await make_output(job=job, user=user, is_thumbnail=False, output_index=0)
        await make_output(job=job, user=user, is_thumbnail=False, output_index=1)
        await make_output(job=job, user=user, is_thumbnail=True, output_index=2)

        result = await gallery_repo.batch_cover_data([job.id])
        cover = result[job.id]
        assert cover.output_count == 2
        assert cover.thumbnail_output_id is not None

    async def test_detects_video_output(
        self,
        gallery_repo: GalleryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
        make_output: OutputFactory,
    ) -> None:
        user = await make_user(email=f"vid-{uuid4().hex[:8]}@example.com")
        job = await make_job(user=user, status="completed")
        vid = await make_output(job=job, user=user, content_type="video/mp4", is_thumbnail=False)

        result = await gallery_repo.batch_cover_data([job.id])
        cover = result[job.id]
        assert cover.video_output_id == vid.id

    async def test_missing_job_gets_empty_cover(
        self,
        gallery_repo: GalleryRepository,
    ) -> None:
        fake_id = uuid4()
        result = await gallery_repo.batch_cover_data([fake_id])
        cover = result[fake_id]
        assert cover.output_count == 0
        assert cover.cover_output_id is None


# ---------------------------------------------------------------------------
# soft-delete filtering in gallery queries
# ---------------------------------------------------------------------------


async def test_list_gallery_jobs_excludes_soft_deleted(
    gallery_repo: GalleryRepository,
    make_user: Callable[..., Coroutine[Any, Any, User]],
    make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    db_session: AsyncSession,
) -> None:
    """list_gallery_jobs excludes soft-deleted completed jobs."""
    from src.db.repositories.job import JobRepository

    user = await make_user(email=f"galdel-{uuid4().hex[:6]}@example.com")
    visible = await make_job(user=user, status="completed")
    deleted = await make_job(user=user, status="completed")

    await JobRepository(db_session).soft_delete(deleted.id, user_id=user.id)

    jobs = await gallery_repo.list_gallery_jobs(user.id, "vex")
    job_ids = {j.id for j in jobs}

    assert visible.id in job_ids
    assert deleted.id not in job_ids


async def test_get_gallery_job_returns_none_for_soft_deleted(
    gallery_repo: GalleryRepository,
    make_user: Callable[..., Coroutine[Any, Any, User]],
    make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    db_session: AsyncSession,
) -> None:
    """get_gallery_job returns None for a soft-deleted job."""
    from src.db.repositories.job import JobRepository

    user = await make_user(email=f"galdetaildel-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user, status="completed")

    await JobRepository(db_session).soft_delete(job.id, user_id=user.id)

    result = await gallery_repo.get_gallery_job(job.id, user.id, "vex")
    assert result is None
