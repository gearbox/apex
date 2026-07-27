"""Integration tests for the backfill_thumbnails CLI command.

Uses a real Postgres session (via db_session fixture) and a mock R2 service
so that thumbnail generation runs through real PIL but no actual S3 calls are made.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from PIL import Image
from sqlalchemy import select

from src.api.services.storage import StorageNotFoundError, StorageType, UploadResult
from src.cli.backfill_thumbnails import _Only, run_backfill
from src.core.thumbnails import THUMBNAIL_SPECS
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage
from src.db.models.user import User

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_png() -> bytes:
    """Return a minimal valid 10x10 PNG suitable for thumbnail generation."""
    buf = io.BytesIO()
    img = Image.new("RGB", (10, 10), color=(200, 100, 50))
    img.save(buf, format="PNG")
    return buf.getvalue()


TINY_PNG = _make_tiny_png()


def _make_mock_r2(
    *,
    fail_keys: set[str] | None = None,
) -> AsyncMock:
    """Build a mock R2StorageService for testing.

    Args:
        fail_keys: Storage keys that should raise StorageNotFoundError on download.
    """
    failing = fail_keys or set()
    upload_log: list[dict[str, Any]] = []

    async def _download(storage_key: str) -> bytes:
        if storage_key in failing:
            raise StorageNotFoundError(f"mock: not found: {storage_key}")
        return TINY_PNG

    async def _upload(
        *,
        user_id: UUID,
        data: bytes,
        content_type: str,
        storage_type: StorageType,
        job_id: UUID | None = None,
    ) -> UploadResult:
        file_id = uuid4()
        if storage_type == StorageType.OUTPUT and job_id is not None:
            key = f"users/{user_id}/outputs/{job_id}/{file_id}.webp"
        else:
            key = f"users/{user_id}/uploads/{file_id}.webp"
        upload_log.append({"key": key, "size": len(data), "content_type": content_type})
        return UploadResult(
            id=file_id,
            storage_key=key,
            expires_at=datetime.now(UTC) + timedelta(days=7),
        )

    mock = AsyncMock()
    mock.download.side_effect = _download
    mock.upload.side_effect = _upload
    mock.delete.return_value = True
    mock._upload_log = upload_log  # expose for assertions
    return mock


def _expires() -> datetime:
    return datetime.now(UTC) + timedelta(days=7)


# ---------------------------------------------------------------------------
# Fixtures: seed rows
# ---------------------------------------------------------------------------


async def _make_user(session: AsyncSession, *, suffix: str = "") -> User:
    user = User(
        id=uuid4(),
        email=f"backfill{suffix}@test.com",
        password_hash="x",
        product_id="vex",
    )
    session.add(user)
    await session.flush()
    return user


async def _make_job(
    session: AsyncSession,
    user: User,
    *,
    generation_type: str = "t2i",
    provider: str = "grok",
) -> GenerationJob:
    job = GenerationJob(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        generation_type=generation_type,
        provider=provider,
        prompt="test",
        status="completed",
    )
    session.add(job)
    await session.flush()
    return job


async def _make_output(
    session: AsyncSession,
    user: User,
    job: GenerationJob,
    *,
    content_type: str = "image/png",
    fmt: str = "png",
    is_thumbnail: bool = False,
    parent_output_id: UUID | None = None,
    thumbnail_max_edge: int | None = None,
    storage_key: str | None = None,
) -> GenerationOutput:
    oid = uuid4()
    key = storage_key or f"users/{user.id}/outputs/{job.id}/{oid}.{fmt}"
    output = GenerationOutput(
        id=oid,
        user_id=user.id,
        job_id=job.id,
        product_id="vex",
        storage_key=key,
        content_type=content_type,
        size_bytes=1024,
        format=fmt,
        output_index=0,
        expires_at=_expires(),
        is_thumbnail=is_thumbnail,
        parent_output_id=parent_output_id,
        thumbnail_max_edge=thumbnail_max_edge,
    )
    session.add(output)
    await session.flush()
    return output


async def _make_upload(
    session: AsyncSession,
    user: User,
    *,
    content_type: str = "image/png",
    fmt: str = "png",
    original_filename: str = "test.png",
    is_thumbnail: bool = False,
    parent_image_id: UUID | None = None,
    thumbnail_max_edge: int | None = None,
    storage_key: str | None = None,
) -> UserImage:
    uid = uuid4()
    key = storage_key or f"users/{user.id}/uploads/{uid}.{fmt}"
    upload = UserImage(
        id=uid,
        user_id=user.id,
        product_id="vex",
        storage_key=key,
        original_filename=original_filename,
        content_type=content_type,
        size_bytes=1024,
        format=fmt,
        expires_at=_expires(),
        is_thumbnail=is_thumbnail,
        parent_image_id=parent_image_id,
        thumbnail_max_edge=thumbnail_max_edge,
    )
    session.add(upload)
    await session.flush()
    return upload


# ---------------------------------------------------------------------------
# Helper query
# ---------------------------------------------------------------------------


async def _get_output_derivatives(
    session: AsyncSession,
    parent_output_id: UUID,
) -> list[GenerationOutput]:
    result = await session.execute(
        select(GenerationOutput).where(GenerationOutput.parent_output_id == parent_output_id)
    )
    return list(result.scalars().all())


async def _get_upload_derivatives(
    session: AsyncSession,
    parent_image_id: UUID,
) -> list[UserImage]:
    result = await session.execute(
        select(UserImage).where(UserImage.parent_image_id == parent_image_id)
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Tests: happy-path scenarios
# ---------------------------------------------------------------------------


async def test_upload_no_derivatives_gets_sm_and_md(db_session: AsyncSession) -> None:
    """(a) An upload with no derivatives gains sm and md WEBP thumbnails."""
    user = await _make_user(db_session, suffix="-a")
    upload = await _make_upload(db_session, user)
    r2 = _make_mock_r2()

    _output_stats, upload_stats = await run_backfill(
        db_session,
        r2,
        product=None,
        only=_Only.uploads,
        dry_run=False,
        batch_size=100,
        limit=None,
        include_video=False,
    )

    derivatives = await _get_upload_derivatives(db_session, upload.id)
    labels = {d.thumbnail_max_edge for d in derivatives}
    assert len(derivatives) == 2
    assert labels == {150, 512}

    for d in derivatives:
        assert d.is_thumbnail is True
        assert d.parent_image_id == upload.id
        assert d.format == "webp"
        assert d.content_type == "image/webp"
        assert d.width is not None
        assert d.height is not None
        assert d.product_id == "vex"
        assert d.user_id == user.id
        assert d.expires_at == upload.expires_at

    assert upload_stats.updated == 1
    assert upload_stats.variants_created == 2
    assert upload_stats.skipped_complete == 0


async def test_image_output_with_md_webp_gains_sm_only(db_session: AsyncSession) -> None:
    """(b) An image output with only a md WEBP derivative gains only sm."""
    user = await _make_user(db_session, suffix="-b")
    job = await _make_job(db_session, user)
    full = await _make_output(db_session, user, job)
    _existing_md = await _make_output(
        db_session,
        user,
        job,
        content_type="image/webp",
        fmt="webp",
        is_thumbnail=True,
        parent_output_id=full.id,
        thumbnail_max_edge=512,
    )
    r2 = _make_mock_r2()

    output_stats, _ = await run_backfill(
        db_session,
        r2,
        product=None,
        only=_Only.outputs,
        dry_run=False,
        batch_size=100,
        limit=None,
        include_video=False,
    )

    derivatives = await _get_output_derivatives(db_session, full.id)
    sm_rows = [d for d in derivatives if d.thumbnail_max_edge == 150]
    md_rows = [d for d in derivatives if d.thumbnail_max_edge == 512]
    assert len(sm_rows) == 1, "exactly one sm derivative should exist"
    assert len(md_rows) == 1, "original md should be unchanged"
    assert sm_rows[0].format == "webp"
    assert md_rows[0].format == "webp"  # original was already webp

    assert output_stats.updated == 1
    assert output_stats.variants_created == 1


async def test_video_output_with_jpeg_poster_gains_sm_only(db_session: AsyncSession) -> None:
    """(c) A video output with a JPEG md poster gains sm only; the JPEG md is untouched."""
    user = await _make_user(db_session, suffix="-c")
    job = await _make_job(db_session, user, generation_type="t2v")
    video_full = await _make_output(
        db_session,
        user,
        job,
        content_type="video/mp4",
        fmt="mp4",
    )
    jpeg_md = await _make_output(
        db_session,
        user,
        job,
        content_type="image/jpeg",
        fmt="jpeg",
        is_thumbnail=True,
        parent_output_id=video_full.id,
        thumbnail_max_edge=512,
    )
    r2 = _make_mock_r2()

    output_stats, _ = await run_backfill(
        db_session,
        r2,
        product=None,
        only=_Only.outputs,
        dry_run=False,
        batch_size=100,
        limit=None,
        include_video=False,
    )

    derivatives = await _get_output_derivatives(db_session, video_full.id)
    sm_rows = [d for d in derivatives if d.thumbnail_max_edge == 150]
    md_rows = [d for d in derivatives if d.thumbnail_max_edge == 512]

    assert len(sm_rows) == 1, "one sm WEBP should have been created"
    assert len(md_rows) == 1, "the JPEG md should not be duplicated"
    assert sm_rows[0].format == "webp"
    assert sm_rows[0].content_type == "image/webp"
    assert md_rows[0].id == jpeg_md.id, "the original JPEG md is unchanged"
    assert md_rows[0].format == "jpeg"

    assert output_stats.updated == 1
    assert output_stats.variants_created == 1


async def test_complete_upload_is_skipped(db_session: AsyncSession) -> None:
    """(d) An upload that already has sm + md WEBP is left unchanged."""
    user = await _make_user(db_session, suffix="-d")
    upload = await _make_upload(db_session, user)
    for spec in THUMBNAIL_SPECS:
        await _make_upload(
            db_session,
            user,
            fmt="webp",
            content_type="image/webp",
            is_thumbnail=True,
            parent_image_id=upload.id,
            thumbnail_max_edge=spec.max_edge,
        )
    r2 = _make_mock_r2()

    _output_stats, upload_stats = await run_backfill(
        db_session,
        r2,
        product=None,
        only=_Only.uploads,
        dry_run=False,
        batch_size=100,
        limit=None,
        include_video=False,
    )

    # R2 upload should never have been called
    r2.upload.assert_not_called()
    assert upload_stats.skipped_complete == 1
    assert upload_stats.updated == 0
    assert upload_stats.variants_created == 0


# ---------------------------------------------------------------------------
# No duplicate labels
# ---------------------------------------------------------------------------


async def test_no_duplicate_labels_after_backfill(db_session: AsyncSession) -> None:
    """No parent ends up with two derivatives sharing the same thumbnail_max_edge."""
    user = await _make_user(db_session, suffix="-nodup")
    job = await _make_job(db_session, user)
    full = await _make_output(db_session, user, job)
    r2 = _make_mock_r2()

    await run_backfill(
        db_session,
        r2,
        product=None,
        only=_Only.all,
        dry_run=False,
        batch_size=100,
        limit=None,
        include_video=False,
    )

    derivatives = await _get_output_derivatives(db_session, full.id)
    edges = [d.thumbnail_max_edge for d in derivatives]
    assert len(edges) == len(set(edges)), "duplicate thumbnail_max_edge detected"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


async def test_second_run_creates_zero_new_rows(db_session: AsyncSession) -> None:
    """A second run after a completed backfill creates no new rows."""
    user = await _make_user(db_session, suffix="-idem")
    upload = await _make_upload(db_session, user)
    r2 = _make_mock_r2()

    await run_backfill(
        db_session,
        r2,
        product=None,
        only=_Only.uploads,
        dry_run=False,
        batch_size=100,
        limit=None,
        include_video=False,
    )

    before = await _get_upload_derivatives(db_session, upload.id)
    assert len(before) == 2

    # Second run
    _, upload_stats2 = await run_backfill(
        db_session,
        r2,
        product=None,
        only=_Only.uploads,
        dry_run=False,
        batch_size=100,
        limit=None,
        include_video=False,
    )

    after = await _get_upload_derivatives(db_session, upload.id)
    assert len(after) == 2, "second run must not add new rows"
    assert upload_stats2.updated == 0
    assert upload_stats2.skipped_complete == 1


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


async def test_dry_run_creates_nothing(db_session: AsyncSession) -> None:
    """--dry-run reports correct would-create counts but writes nothing."""
    user = await _make_user(db_session, suffix="-dry")
    upload = await _make_upload(db_session, user)
    r2 = _make_mock_r2()

    _, upload_stats = await run_backfill(
        db_session,
        r2,
        product=None,
        only=_Only.uploads,
        dry_run=True,
        batch_size=100,
        limit=None,
        include_video=False,
    )

    # No DB rows created
    derivatives = await _get_upload_derivatives(db_session, upload.id)
    assert derivatives == []

    # No R2 uploads
    r2.upload.assert_not_called()

    # But stats reflect what would have been created
    assert upload_stats.updated == 1
    assert upload_stats.variants_created == 2


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


async def test_r2_download_failure_isolates_one_row(db_session: AsyncSession) -> None:
    """If r2.download raises for one upload, that row is FAILED and others succeed."""
    user = await _make_user(db_session, suffix="-fail")
    upload_ok = await _make_upload(db_session, user, storage_key=f"users/{user.id}/uploads/ok.png")
    upload_bad = await _make_upload(
        db_session, user, storage_key=f"users/{user.id}/uploads/bad.png"
    )
    r2 = _make_mock_r2(fail_keys={upload_bad.storage_key})

    _, upload_stats = await run_backfill(
        db_session,
        r2,
        product=None,
        only=_Only.uploads,
        dry_run=False,
        batch_size=100,
        limit=None,
        include_video=False,
    )

    ok_derivatives = await _get_upload_derivatives(db_session, upload_ok.id)
    bad_derivatives = await _get_upload_derivatives(db_session, upload_bad.id)

    assert len(ok_derivatives) == 2, "good upload should have been backfilled"
    assert bad_derivatives == [], "failed upload should have no derivatives"
    assert upload_stats.failed == 1
    assert upload_stats.updated == 1


# ---------------------------------------------------------------------------
# Video without --include-video
# ---------------------------------------------------------------------------


async def test_video_without_poster_is_skipped_no_poster(db_session: AsyncSession) -> None:
    """A video output with no poster derivative is SKIPPED_NO_POSTER (not FAILED)."""
    user = await _make_user(db_session, suffix="-noposter")
    job = await _make_job(db_session, user, generation_type="t2v")
    video_full = await _make_output(
        db_session,
        user,
        job,
        content_type="video/mp4",
        fmt="mp4",
    )
    r2 = _make_mock_r2()

    output_stats, _ = await run_backfill(
        db_session,
        r2,
        product=None,
        only=_Only.outputs,
        dry_run=False,
        batch_size=100,
        limit=None,
        include_video=False,
    )

    derivatives = await _get_output_derivatives(db_session, video_full.id)
    assert derivatives == []
    assert output_stats.skipped_no_poster == 1
    assert output_stats.failed == 0


# ---------------------------------------------------------------------------
# Product filter
# ---------------------------------------------------------------------------


async def test_product_filter_isolates_target_product(db_session: AsyncSession) -> None:
    """--product limits backfill to rows with that product_id."""
    user = await _make_user(db_session, suffix="-prod")
    upload_vex = await _make_upload(db_session, user)  # product_id = "vex"
    # Create a synthara user + upload
    user_syn = User(
        id=uuid4(),
        email="backfill-syn@test.com",
        password_hash="x",
        product_id="synthara",
    )
    db_session.add(user_syn)
    await db_session.flush()
    upload_syn = UserImage(
        id=uuid4(),
        user_id=user_syn.id,
        product_id="synthara",
        storage_key=f"users/{user_syn.id}/uploads/{uuid4()}.png",
        original_filename="syn.png",
        content_type="image/png",
        size_bytes=1024,
        format="png",
        expires_at=_expires(),
    )
    db_session.add(upload_syn)
    await db_session.flush()

    r2 = _make_mock_r2()

    _, upload_stats = await run_backfill(
        db_session,
        r2,
        product="vex",
        only=_Only.uploads,
        dry_run=False,
        batch_size=100,
        limit=None,
        include_video=False,
    )

    vex_derivatives = await _get_upload_derivatives(db_session, upload_vex.id)
    syn_derivatives = await _get_upload_derivatives(db_session, upload_syn.id)

    assert len(vex_derivatives) == 2, "vex upload should be backfilled"
    assert syn_derivatives == [], "synthara upload must not be touched"
    assert upload_stats.scanned == 1
