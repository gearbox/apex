"""Integration tests for the content retention sweeper against a real PostgreSQL database.

ContentRetentionService commits internally across independent sessions, so
setup here uses a real (committing) session factory bound directly to the
test engine rather than the SAVEPOINT-scoped ``db_session`` fixture used
elsewhere — that fixture never commits, so writes made through it would be
invisible to the sweeper's own sessions.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.services.content_retention import ContentRetentionService
from src.api.services.storage.r2 import R2StorageService, R2StorageSettings
from src.core.enums import GenerationType, JobStatus
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage
from src.db.models.user import User
from src.db.repositories.library import LibraryRepository

if TYPE_CHECKING:
    from collections.abc import Generator

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


class FakeR2Storage:
    """Records delete_many calls instead of touching real R2.

    Used by every integration test here except the end-to-end one below,
    which routes through a real ``R2StorageService`` (patched only at the
    boto client boundary) to exercise the actual ``delete_many`` batching /
    response-parsing logic. The other tests assert DB/cascade behaviour and
    which keys are passed — routing them through the real client would add
    nothing beyond what the one real-`delete_many` test already proves.
    """

    def __init__(self) -> None:
        self.deleted_keys: list[str] = []

    async def delete_many(self, storage_keys: list[str]) -> int:
        self.deleted_keys.extend(storage_keys)
        return len(storage_keys)


@pytest.fixture
def recording_r2() -> Generator[tuple[R2StorageService, list[str]]]:
    """Real R2StorageService running its true delete_many logic against a
    mock S3 client that records every deleted key. No moto, no live server."""
    deleted: list[str] = []
    mock_client = AsyncMock()

    async def _delete_objects(*, Bucket: str, Delete: dict) -> dict:  # noqa: ARG001
        keys = [o["Key"] for o in Delete["Objects"]]
        deleted.extend(keys)
        return {"Deleted": [{"Key": k} for k in keys]}

    mock_client.delete_objects = AsyncMock(side_effect=_delete_objects)

    settings = R2StorageSettings(
        account_id="test",
        access_key_id="test",
        secret_access_key="test",
        bucket_name="apex-user-content",
    )
    storage = R2StorageService(settings)

    @asynccontextmanager
    async def _fake_get_client():
        yield mock_client

    with patch.object(storage, "_get_client", _fake_get_client):
        yield storage, deleted


@pytest.fixture
def retention_session_factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Real, committing session factory bound directly to the test engine."""
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


def _past(hours: int = 1) -> datetime:
    return datetime.now(UTC) - timedelta(hours=hours)


def _future(days: int = 7) -> datetime:
    return datetime.now(UTC) + timedelta(days=days)


async def _create_user(session_factory: async_sessionmaker[AsyncSession]) -> User:
    async with session_factory() as session:
        user = User(
            id=uuid4(),
            email=f"retention-{uuid4().hex[:8]}@example.com",
            password_hash="hashed",
            product_id="vex",
        )
        session.add(user)
        await session.commit()
        return user


async def _create_job(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user: User,
    status: JobStatus = JobStatus.COMPLETED,
    source_output_id: object = None,
) -> GenerationJob:
    async with session_factory() as session:
        job = GenerationJob(
            id=uuid4(),
            user_id=user.id,
            product_id=user.product_id,
            name="Retention Test Job",
            status=status.value,
            generation_type=GenerationType.T2I.value,
            prompt="a test image",
            provider="grok",
            source_output_id=source_output_id,
        )
        session.add(job)
        await session.commit()
        return job


async def _create_output(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user: User,
    job: GenerationJob,
    expires_at: datetime,
    with_thumbnail: bool = False,
) -> tuple[GenerationOutput, GenerationOutput | None]:
    async with session_factory() as session:
        out_id = uuid4()
        output = GenerationOutput(
            id=out_id,
            user_id=user.id,
            job_id=job.id,
            product_id=user.product_id,
            storage_key=f"users/{user.id}/outputs/{job.id}/{out_id}.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            output_index=0,
            expires_at=expires_at,
        )
        session.add(output)
        await session.flush()

        thumbnail: GenerationOutput | None = None
        if with_thumbnail:
            thumb_id = uuid4()
            thumbnail = GenerationOutput(
                id=thumb_id,
                user_id=user.id,
                job_id=job.id,
                product_id=user.product_id,
                storage_key=f"users/{user.id}/outputs/{job.id}/{thumb_id}_sm.webp",
                content_type="image/webp",
                size_bytes=20,
                format="webp",
                output_index=0,
                is_thumbnail=True,
                parent_output_id=out_id,
                thumbnail_max_edge=150,
                expires_at=expires_at,
            )
            session.add(thumbnail)

        await session.commit()
        return output, thumbnail


async def _create_upload(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user: User,
    expires_at: datetime,
    with_thumbnail: bool = False,
) -> tuple[UserImage, UserImage | None]:
    async with session_factory() as session:
        img_id = uuid4()
        image = UserImage(
            id=img_id,
            user_id=user.id,
            product_id=user.product_id,
            storage_key=f"users/{user.id}/uploads/{img_id}.png",
            original_filename="photo.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            expires_at=expires_at,
        )
        session.add(image)
        await session.flush()

        thumbnail: UserImage | None = None
        if with_thumbnail:
            thumb_id = uuid4()
            thumbnail = UserImage(
                id=thumb_id,
                user_id=user.id,
                product_id=user.product_id,
                storage_key=f"users/{user.id}/uploads/{thumb_id}_sm.webp",
                original_filename="photo_sm.webp",
                content_type="image/webp",
                size_bytes=20,
                format="webp",
                is_thumbnail=True,
                parent_image_id=img_id,
                thumbnail_max_edge=150,
                expires_at=expires_at,
            )
            session.add(thumbnail)

        await session.commit()
        return image, thumbnail


async def test_end_to_end_sweep_removes_rows_thumbnails_and_r2_objects(
    retention_session_factory: async_sessionmaker[AsyncSession],
    db_session: AsyncSession,
    recording_r2: tuple[R2StorageService, list[str]],
) -> None:
    """A full sweep deletes expired output + upload rows, cascades to their
    thumbnails, and best-effort deletes every collected R2 key — driven
    through the real R2StorageService.delete_many (batching + Deleted-count
    parsing), not FakeR2Storage."""
    user = await _create_user(retention_session_factory)
    job = await _create_job(retention_session_factory, user=user)
    output, output_thumb = await _create_output(
        retention_session_factory, user=user, job=job, expires_at=_past(), with_thumbnail=True
    )
    upload, upload_thumb = await _create_upload(
        retention_session_factory, user=user, expires_at=_past(), with_thumbnail=True
    )
    assert output_thumb is not None
    assert upload_thumb is not None

    storage, deleted_keys = recording_r2
    service = ContentRetentionService(
        session_factory=retention_session_factory,
        storage=storage,
        batch_size=500,
        max_batches_per_run=20,
    )

    result = await service.sweep()

    expected_keys = {
        output.storage_key,
        output_thumb.storage_key,
        upload.storage_key,
        upload_thumb.storage_key,
    }

    assert result.outputs_deleted == 1
    assert result.uploads_deleted == 1
    assert result.r2_delete_failed is False
    assert result.r2_keys_deleted == len(expected_keys)
    assert set(deleted_keys) == expected_keys

    assert await db_session.get(GenerationOutput, output.id) is None
    assert await db_session.get(GenerationOutput, output_thumb.id) is None
    assert await db_session.get(UserImage, upload.id) is None
    assert await db_session.get(UserImage, upload_thumb.id) is None


async def test_sweep_nulls_lineage_fks_on_downstream_jobs(
    retention_session_factory: async_sessionmaker[AsyncSession],
    db_session: AsyncSession,
) -> None:
    """Downstream job's source_output_id is nulled (ON DELETE SET NULL);
    the downstream job row itself survives."""
    user = await _create_user(retention_session_factory)
    source_job = await _create_job(retention_session_factory, user=user)
    source_output, _ = await _create_output(
        retention_session_factory, user=user, job=source_job, expires_at=_past()
    )
    downstream_job = await _create_job(
        retention_session_factory, user=user, source_output_id=source_output.id
    )

    storage = FakeR2Storage()
    service = ContentRetentionService(
        session_factory=retention_session_factory,
        storage=storage,  # type: ignore[arg-type]
        batch_size=500,
        max_batches_per_run=20,
    )
    result = await service.sweep()

    assert result.outputs_deleted == 1

    refreshed_downstream = await db_session.get(GenerationJob, downstream_job.id)
    assert refreshed_downstream is not None
    assert refreshed_downstream.source_output_id is None

    # source_job had exactly one output, now gone -> soft-deleted.
    refreshed_source = await db_session.get(GenerationJob, source_job.id)
    assert refreshed_source is not None
    assert refreshed_source.is_deleted is True


async def test_library_excludes_soft_deleted_swept_jobs(
    retention_session_factory: async_sessionmaker[AsyncSession],
    db_session: AsyncSession,
) -> None:
    """A job whose only output gets swept disappears from the library listing."""
    user = await _create_user(retention_session_factory)
    job = await _create_job(retention_session_factory, user=user, status=JobStatus.COMPLETED)
    await _create_output(retention_session_factory, user=user, job=job, expires_at=_past())

    storage = FakeR2Storage()
    service = ContentRetentionService(
        session_factory=retention_session_factory,
        storage=storage,  # type: ignore[arg-type]
        batch_size=500,
        max_batches_per_run=20,
    )
    await service.sweep()

    library_repo = LibraryRepository(db_session)
    rows = await library_repo.list_assets(user.id, user.product_id, limit=20)
    assert job.id not in {r.job_id for r in rows if r.job_id is not None}


async def test_sweep_keeps_job_with_live_output(
    retention_session_factory: async_sessionmaker[AsyncSession],
    db_session: AsyncSession,
) -> None:
    """A job with one expired and one still-live output is not soft-deleted."""
    user = await _create_user(retention_session_factory)
    job = await _create_job(retention_session_factory, user=user)
    await _create_output(retention_session_factory, user=user, job=job, expires_at=_past())
    await _create_output(retention_session_factory, user=user, job=job, expires_at=_future())

    storage = FakeR2Storage()
    service = ContentRetentionService(
        session_factory=retention_session_factory,
        storage=storage,  # type: ignore[arg-type]
        batch_size=500,
        max_batches_per_run=20,
    )
    result = await service.sweep()

    assert result.outputs_deleted == 1
    assert result.jobs_soft_deleted == 0

    refreshed_job = await db_session.get(GenerationJob, job.id)
    assert refreshed_job is not None
    assert refreshed_job.is_deleted is False


async def test_sweep_ignores_unexpired_content(
    retention_session_factory: async_sessionmaker[AsyncSession],
    db_session: AsyncSession,
) -> None:
    """Content with a future expires_at is left untouched."""
    user = await _create_user(retention_session_factory)
    job = await _create_job(retention_session_factory, user=user)
    output, _ = await _create_output(
        retention_session_factory, user=user, job=job, expires_at=_future()
    )
    upload, _ = await _create_upload(retention_session_factory, user=user, expires_at=_future())

    storage = FakeR2Storage()
    service = ContentRetentionService(
        session_factory=retention_session_factory,
        storage=storage,  # type: ignore[arg-type]
        batch_size=500,
        max_batches_per_run=20,
    )
    result = await service.sweep()

    assert result.outputs_deleted == 0
    assert result.uploads_deleted == 0
    assert storage.deleted_keys == []
    assert await db_session.get(GenerationOutput, output.id) is not None
    assert await db_session.get(UserImage, upload.id) is not None


async def test_sweep_thumbnail_alone_is_not_selected(
    retention_session_factory: async_sessionmaker[AsyncSession],
    db_session: AsyncSession,
) -> None:
    """get_expired excludes thumbnail rows even if independently expired —
    they are only ever removed via cascade from their parent."""
    user = await _create_user(retention_session_factory)
    job = await _create_job(retention_session_factory, user=user)
    # Full output not expired, but simulate an (unusual) independently
    # expired thumbnail row to confirm it's never selected directly.
    output, thumbnail = await _create_output(
        retention_session_factory, user=user, job=job, expires_at=_future(), with_thumbnail=True
    )
    assert thumbnail is not None
    async with retention_session_factory() as session:
        thumb_row = await session.get(GenerationOutput, thumbnail.id)
        assert thumb_row is not None
        thumb_row.expires_at = _past()
        await session.commit()

    storage = FakeR2Storage()
    service = ContentRetentionService(
        session_factory=retention_session_factory,
        storage=storage,  # type: ignore[arg-type]
        batch_size=500,
        max_batches_per_run=20,
    )
    result = await service.sweep()

    assert result.outputs_deleted == 0
    assert storage.deleted_keys == []
    assert await db_session.get(GenerationOutput, output.id) is not None
    assert await db_session.get(GenerationOutput, thumbnail.id) is not None
