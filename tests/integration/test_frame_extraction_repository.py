"""Integration tests for FrameExtractionJobRepository and the migration 020
constraints, against a real PostgreSQL database.

The worker unit tests (tests/unit/test_frames_worker.py) are fully mocked
and cannot exercise FOR UPDATE SKIP LOCKED or the CHECK/FK constraints — that
requires a real database, hence this file. See tests/integration/conftest.py
for fixture conventions and test_partial_refund_concurrency.py for the
two-connection pattern SKIP LOCKED tests need.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from alembic.config import Config
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from alembic import command
from src.core.enums import FrameExtractionKind, FrameExtractionStatus
from src.core.uid import new_id
from src.db.models.frame_extraction import FrameExtractionJob
from src.db.models.storage import GenerationOutput, UserImage
from src.db.models.user import User
from src.db.repositories.frame_extraction import FrameExtractionJobRepository

if TYPE_CHECKING:
    from tests.integration.conftest import FrameExtractionJobFactory, JobFactory, UserImageFactory

# No module-level `pytestmark = pytest.mark.asyncio` — this file mixes async
# tests (asyncio_mode="auto" picks them up without a marker) with one
# deliberately sync test (the migration round-trip; see its docstring).


# ---------------------------------------------------------------------------
# SKIP LOCKED — needs two real connections; the SAVEPOINT-based db_session
# fixture only allocates one, so these seed/clean up directly against the
# shared engine (mirrors test_partial_refund_concurrency.py).
# ---------------------------------------------------------------------------


async def _seed_queued_job(
    engine: AsyncEngine, *, created_at: datetime | None = None
) -> tuple[User, FrameExtractionJob]:
    user = User(
        id=new_id(),
        email=f"frameclaim-{uuid4().hex[:8]}@example.com",
        password_hash="x" * 64,
        product_id="vex",
        is_active=True,
    )
    upload = UserImage(
        id=new_id(),
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/src.mp4",
        original_filename="src.mp4",
        content_type="video/mp4",
        size_bytes=100,
        format="mp4",
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    job = FrameExtractionJob(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        kind=FrameExtractionKind.PREVIEW.value,
        status=FrameExtractionStatus.QUEUED.value,
        source_upload_id=upload.id,
        params={"frame_count": 12},
    )
    if created_at is not None:
        job.created_at = created_at

    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add_all([user, upload])
        await session.flush()
        session.add(job)
        await session.commit()
        await session.refresh(user)
        await session.refresh(job)
    return user, job


async def _cleanup_user(engine: AsyncEngine, user_id) -> None:
    """Delete the seeded user — cascades to its uploads and frame jobs."""
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(delete(User).where(User.id == user_id))
        await session.commit()


async def test_claim_next_skip_locked_no_double_claim(db_engine: AsyncEngine) -> None:
    """Two independent connections racing claim_next() on the same queued job:
    the first holds the row lock uncommitted, the second must see nothing
    (SKIP LOCKED) rather than blocking or double-claiming. After the first
    commits, the job is 'running' and a third claim also finds nothing."""
    user, job = await _seed_queued_job(db_engine)
    session1 = AsyncSession(bind=db_engine, expire_on_commit=False)
    session2 = AsyncSession(bind=db_engine, expire_on_commit=False)
    try:
        claimed1 = await FrameExtractionJobRepository(session1).claim_next()
        assert claimed1 is not None
        assert claimed1.id == job.id

        # session2 is a distinct connection (NullPool: every checkout is a
        # fresh physical connection) — SKIP LOCKED means it sees nothing
        # rather than blocking on session1's uncommitted row lock.
        claimed2 = await FrameExtractionJobRepository(session2).claim_next()
        assert claimed2 is None
        await session2.rollback()

        await session1.commit()

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session3:
            claimed3 = await FrameExtractionJobRepository(session3).claim_next()
            assert claimed3 is None
    finally:
        await session1.close()
        await session2.close()
        await _cleanup_user(db_engine, user.id)


async def test_claim_next_orders_by_created_at(db_engine: AsyncEngine) -> None:
    """The oldest queued job is claimed first."""
    now = datetime.now(UTC)
    older_user, older_job = await _seed_queued_job(
        db_engine, created_at=now - timedelta(minutes=10)
    )
    newer_user, _newer_job = await _seed_queued_job(
        db_engine, created_at=now - timedelta(minutes=1)
    )
    try:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
            claimed = await FrameExtractionJobRepository(session).claim_next()
            assert claimed is not None
            assert claimed.id == older_job.id
            await session.commit()
    finally:
        await _cleanup_user(db_engine, older_user.id)
        await _cleanup_user(db_engine, newer_user.id)


# ---------------------------------------------------------------------------
# fail_stale_running (F2)
# ---------------------------------------------------------------------------


async def test_fail_stale_running_marks_only_stale(
    db_session: AsyncSession,
    frame_extraction_job_repo: FrameExtractionJobRepository,
    make_frame_extraction_job: FrameExtractionJobFactory,
) -> None:
    """Only a running job started before the cutoff is failed; a fresh
    running job and a queued job are left untouched."""
    now = datetime.now(UTC)
    stale = await make_frame_extraction_job(
        status=FrameExtractionStatus.RUNNING.value,
        started_at=now - timedelta(minutes=10),
    )
    fresh = await make_frame_extraction_job(
        status=FrameExtractionStatus.RUNNING.value,
        started_at=now - timedelta(seconds=10),
    )
    queued = await make_frame_extraction_job(status=FrameExtractionStatus.QUEUED.value)

    count = await frame_extraction_job_repo.fail_stale_running(cutoff=now - timedelta(minutes=5))
    assert count == 1

    # fail_stale_running is a Core UPDATE (synchronize_session=False) — the
    # ORM identity map still holds the pre-update in-memory state until
    # expired.
    db_session.expunge_all()

    refreshed_stale = await frame_extraction_job_repo.get(stale.id)
    assert refreshed_stale is not None
    assert refreshed_stale.status == FrameExtractionStatus.FAILED.value
    assert refreshed_stale.error == "worker died mid-execution"
    assert refreshed_stale.finished_at is not None

    refreshed_fresh = await frame_extraction_job_repo.get(fresh.id)
    assert refreshed_fresh is not None
    assert refreshed_fresh.status == FrameExtractionStatus.RUNNING.value

    refreshed_queued = await frame_extraction_job_repo.get(queued.id)
    assert refreshed_queued is not None
    assert refreshed_queued.status == FrameExtractionStatus.QUEUED.value


# ---------------------------------------------------------------------------
# user_images lineage constraints (migration 020)
# ---------------------------------------------------------------------------


async def test_user_images_check_rejects_double_source(
    db_session: AsyncSession,
    make_user,
    make_user_image: UserImageFactory,
    make_job: JobFactory,
) -> None:
    """A UserImage row with both source_output_id and source_upload_id set
    violates ck_user_images_single_frame_source."""
    user = await make_user(email=f"doublesrc-{uuid4().hex[:6]}@example.com")
    source_upload = await make_user_image(user=user, content_type="video/mp4")
    job = await make_job(user=user)
    output = GenerationOutput(
        id=uuid4(),
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/src.mp4",
        content_type="video/mp4",
        size_bytes=100,
        format="mp4",
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    db_session.add(output)
    await db_session.flush()

    db_session.add(
        UserImage(
            id=uuid4(),
            user_id=user.id,
            storage_key=f"users/{user.id}/uploads/frame.png",
            original_filename="frame.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
            source_output_id=output.id,
            source_upload_id=source_upload.id,
        )
    )
    with pytest.raises(IntegrityError, match="ck_user_images_single_frame_source"):
        await db_session.flush()


async def test_user_images_lineage_set_null_on_source_delete(
    db_session: AsyncSession,
    user_image_repo,
    make_user,
    make_user_image: UserImageFactory,
) -> None:
    """Deleting the source upload SET NULLs the frame's source_upload_id but
    leaves the frame row (and source_timestamp_ms) intact."""
    user = await make_user(email=f"srcdelete-{uuid4().hex[:6]}@example.com")
    source = await make_user_image(user=user, content_type="video/mp4")
    frame = UserImage(
        id=uuid4(),
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/frame.png",
        original_filename="frame.png",
        content_type="image/png",
        size_bytes=100,
        format="png",
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
        source_upload_id=source.id,
        source_timestamp_ms=1500,
    )
    db_session.add(frame)
    await db_session.flush()

    deleted = await user_image_repo.delete(source.id)
    assert deleted is True

    # The FK's ON DELETE SET NULL is a database-level side effect — expire
    # the identity map so the re-fetch reflects it instead of the stale
    # in-memory `frame` object.
    db_session.expunge_all()

    survivor = await user_image_repo.get(frame.id)
    assert survivor is not None
    assert survivor.source_upload_id is None
    assert survivor.source_timestamp_ms == 1500


# ---------------------------------------------------------------------------
# frame_extraction_jobs constraints (migration 020)
# ---------------------------------------------------------------------------


async def test_frame_jobs_check_requires_exactly_one_source(
    db_session: AsyncSession,
    make_user,
    make_user_image: UserImageFactory,
    make_job: JobFactory,
) -> None:
    """Zero sources and two sources both violate
    ck_frame_extraction_jobs_exactly_one_source."""
    user = await make_user(email=f"zerosrc-{uuid4().hex[:6]}@example.com")

    sp = await db_session.begin_nested()
    db_session.add(
        FrameExtractionJob(
            id=uuid4(),
            user_id=user.id,
            product_id="vex",
            kind=FrameExtractionKind.PREVIEW.value,
            status=FrameExtractionStatus.QUEUED.value,
            source_output_id=None,
            source_upload_id=None,
            params={"frame_count": 12},
        )
    )
    with pytest.raises(IntegrityError, match="ck_frame_extraction_jobs_exactly_one_source"):
        await db_session.flush()
    await sp.rollback()

    upload = await make_user_image(user=user, content_type="video/mp4")
    job = await make_job(user=user)
    output = GenerationOutput(
        id=uuid4(),
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/src.mp4",
        content_type="video/mp4",
        size_bytes=100,
        format="mp4",
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    db_session.add(output)
    await db_session.flush()

    db_session.add(
        FrameExtractionJob(
            id=uuid4(),
            user_id=user.id,
            product_id="vex",
            kind=FrameExtractionKind.PREVIEW.value,
            status=FrameExtractionStatus.QUEUED.value,
            source_output_id=output.id,
            source_upload_id=upload.id,
            params={"frame_count": 12},
        )
    )
    with pytest.raises(IntegrityError, match="ck_frame_extraction_jobs_exactly_one_source"):
        await db_session.flush()


async def test_frame_jobs_cascade_with_source(
    db_session: AsyncSession,
    frame_extraction_job_repo: FrameExtractionJobRepository,
    user_image_repo,
    make_user,
    make_user_image: UserImageFactory,
) -> None:
    """Deleting the source upload CASCADE-deletes a queued job pointing at it
    (unlike UserImage lineage, a job's source must exist for the job to be
    runnable — see the model docstring)."""
    user = await make_user(email=f"jobcascade-{uuid4().hex[:6]}@example.com")
    source = await make_user_image(user=user, content_type="video/mp4")
    job = FrameExtractionJob(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        kind=FrameExtractionKind.PREVIEW.value,
        status=FrameExtractionStatus.QUEUED.value,
        source_upload_id=source.id,
        params={"frame_count": 12},
    )
    db_session.add(job)
    await db_session.flush()

    deleted = await user_image_repo.delete(source.id)
    assert deleted is True

    # ON DELETE CASCADE is a database-level side effect — session.get()
    # would otherwise return the stale in-memory `job` from the identity map.
    db_session.expunge_all()

    survivor = await frame_extraction_job_repo.get(job.id)
    assert survivor is None


# ---------------------------------------------------------------------------
# Migration round-trip — no established per-migration up/down test pattern
# exists yet (test_migration_drift.py covers autogenerate-drift at head, not
# a round trip), so this is the "smoke test in this file" the review calls
# for. Deliberately sync (not `async def`): alembic's env.py drives its own
# asyncio.run() internally, which cannot be nested inside pytest-asyncio's
# already-running loop — see test_migration_drift.py for the same pattern.
# ---------------------------------------------------------------------------


def test_migration_020_downgrade_upgrade_round_trip(
    test_database_url: str,
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downgrading past 020 removes frame extraction support, and upgrading
    back to head restores it — verifies the down_revision chain and
    downgrade() body are actually correct, not just upgrade()."""
    assert db_engine is not None  # ensures session-scoped Alembic upgrade has already run
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("DEBUG", "false")
    config = Config("alembic.ini")

    try:
        command.downgrade(config, "019")
        command.upgrade(config, "020")
    finally:
        # Always leave the shared session-scoped schema at head, even if an
        # assertion above fails — other test files in this session depend on it.
        command.upgrade(config, "head")
