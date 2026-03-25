"""Integration tests for StorageRepository against a real PostgreSQL database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import GenerationType, JobStatus
from src.db.repositories.storage import StorageRepository

# ---------------------------------------------------------------------------
# UserImage — create / get / list / delete
# ---------------------------------------------------------------------------


async def test_create_user_image(storage_repo: StorageRepository, make_user) -> None:
    """create_user_image persists and returns a UserImage."""
    user = await make_user(email=f"imgcreate-{uuid4().hex[:6]}@example.com")
    img_id = uuid4()
    image = await storage_repo.create_user_image(
        id=img_id,
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/{img_id}.png",
        original_filename="photo.png",
        content_type="image/png",
        size_bytes=2048,
        format="png",
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    assert image.id == img_id
    assert image.user_id == user.id
    assert image.size_bytes == 2048


async def test_create_user_image_duplicate_key_raises(
    storage_repo: StorageRepository, make_user
) -> None:
    """Creating two UserImages with the same storage_key raises IntegrityError."""
    user = await make_user(email=f"imgdup-{uuid4().hex[:6]}@example.com")
    key = f"users/{user.id}/uploads/dup.png"
    await storage_repo.create_user_image(
        id=uuid4(),
        user_id=user.id,
        storage_key=key,
        original_filename="dup.png",
        content_type="image/png",
        size_bytes=100,
        format="png",
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    with pytest.raises(IntegrityError):
        await storage_repo.create_user_image(
            id=uuid4(),
            user_id=user.id,
            storage_key=key,
            original_filename="dup.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
        )


async def test_get_user_image_found(storage_repo: StorageRepository, make_user_image) -> None:
    """get_user_image returns the image by PK."""
    image = await make_user_image()
    found = await storage_repo.get_user_image(image.id)
    assert found is not None
    assert found.id == image.id


async def test_get_user_image_not_found(storage_repo: StorageRepository) -> None:
    """get_user_image returns None for an unknown UUID."""
    assert await storage_repo.get_user_image(uuid4()) is None


async def test_get_user_image_ownership_enforced(
    storage_repo: StorageRepository, make_user_image, make_user
) -> None:
    """get_user_image with wrong user_id returns None."""
    image = await make_user_image()
    other_user = await make_user(email=f"other-{uuid4().hex[:6]}@example.com")
    found = await storage_repo.get_user_image(image.id, user_id=other_user.id)
    assert found is None


async def test_get_user_image_by_key(storage_repo: StorageRepository, make_user) -> None:
    """get_user_image_by_key returns image by storage key."""
    user = await make_user(email=f"imgkey-{uuid4().hex[:6]}@example.com")
    img_id = uuid4()
    key = f"users/{user.id}/uploads/{img_id}.png"
    await storage_repo.create_user_image(
        id=img_id,
        user_id=user.id,
        storage_key=key,
        original_filename="x.png",
        content_type="image/png",
        size_bytes=512,
        format="png",
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    found = await storage_repo.get_user_image_by_key(key)
    assert found is not None
    assert found.storage_key == key


async def test_list_user_images_paginated(
    storage_repo: StorageRepository, make_user, make_user_image
) -> None:
    """list_user_images returns paginated results using limit+1 fetch pattern."""
    user = await make_user(email=f"imglist-{uuid4().hex[:6]}@example.com")
    for i in range(5):
        await make_user_image(user=user, storage_key=f"users/{user.id}/uploads/{i}.png")
    # limit+1 pattern: fetching limit=3 returns up to 4 items (limit+1)
    images = await storage_repo.list_user_images(user.id, limit=3)
    assert len(images) == 4  # 3+1 since 5 > 3, has_more=True


async def test_list_user_images_empty_for_new_user(
    storage_repo: StorageRepository, make_user
) -> None:
    """list_user_images returns empty list for a user with no uploads."""
    user = await make_user(email=f"imglistoff-{uuid4().hex[:6]}@example.com")
    images = await storage_repo.list_user_images(user.id)
    assert not list(images)


async def test_delete_user_image(storage_repo: StorageRepository, make_user_image) -> None:
    """delete_user_image removes the row and returns True."""
    image = await make_user_image()
    result = await storage_repo.delete_user_image(image.id)
    assert result is True
    assert await storage_repo.get_user_image(image.id) is None


async def test_delete_user_image_not_found_returns_false(
    storage_repo: StorageRepository,
) -> None:
    """delete_user_image returns False for an unknown image."""
    assert await storage_repo.delete_user_image(uuid4()) is False


async def test_get_expired_images(storage_repo: StorageRepository, make_user) -> None:
    """get_expired_images returns images past their expires_at."""
    user = await make_user(email=f"expiredimg-{uuid4().hex[:6]}@example.com")
    past = datetime.now(UTC) - timedelta(hours=1)
    img_id = uuid4()
    await storage_repo.create_user_image(
        id=img_id,
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/{img_id}.png",
        original_filename="old.png",
        content_type="image/png",
        size_bytes=100,
        format="png",
        expires_at=past,
        product_id="vex",
    )
    expired = await storage_repo.get_expired_images()
    assert any(img.id == img_id for img in expired)


# ---------------------------------------------------------------------------
# GenerationJob — create / get / update / list
# ---------------------------------------------------------------------------


async def test_create_job(storage_repo: StorageRepository, make_user) -> None:
    """create_job persists and returns a GenerationJob in pending status."""
    user = await make_user(email=f"jobcreate-{uuid4().hex[:6]}@example.com")
    job = await storage_repo.create_job(
        id=uuid4(),
        user_id=user.id,
        name="My Job",
        prompt="A beautiful landscape",
        generation_type=GenerationType.T2I,
        product_id="vex",
    )
    assert job.status == "pending"
    assert job.prompt == "A beautiful landscape"


async def test_get_job_found(storage_repo: StorageRepository, make_job) -> None:
    """get_job returns the job by PK."""
    job = await make_job()
    found = await storage_repo.get_job(job.id)
    assert found is not None
    assert found.id == job.id


async def test_get_job_not_found(storage_repo: StorageRepository) -> None:
    """get_job returns None for unknown UUID."""
    assert await storage_repo.get_job(uuid4()) is None


async def test_get_job_ownership_enforced(
    storage_repo: StorageRepository, make_job, make_user
) -> None:
    """get_job with wrong user_id returns None."""
    job = await make_job()
    other = await make_user(email=f"otherjob-{uuid4().hex[:6]}@example.com")
    found = await storage_repo.get_job(job.id, user_id=other.id)
    assert found is None


async def test_update_job_status(storage_repo: StorageRepository, make_job) -> None:
    """update_job_status changes the job status."""
    job = await make_job(status="pending")
    updated = await storage_repo.update_job_status(job.id, JobStatus.RUNNING)
    assert updated is not None
    assert updated.status == JobStatus.RUNNING


async def test_update_job_status_unknown_returns_none(
    storage_repo: StorageRepository,
) -> None:
    """update_job_status returns None for an unknown job ID."""
    assert await storage_repo.update_job_status(uuid4(), JobStatus.RUNNING) is None


async def test_update_job_status_with_started_at(storage_repo: StorageRepository, make_job) -> None:
    """update_job_status persists started_at timestamp."""
    job = await make_job()
    started = datetime.now(UTC)
    updated = await storage_repo.update_job_status(job.id, JobStatus.RUNNING, started_at=started)
    assert updated is not None
    assert updated.started_at is not None


async def test_update_job_status_with_completed_at(
    storage_repo: StorageRepository, make_job
) -> None:
    """update_job_status persists completed_at timestamp."""
    job = await make_job()
    completed = datetime.now(UTC)
    updated = await storage_repo.update_job_status(
        job.id, JobStatus.COMPLETED, completed_at=completed
    )
    assert updated is not None
    assert updated.completed_at is not None


async def test_list_user_jobs_returns_owned_jobs(
    storage_repo: StorageRepository, make_user, make_job
) -> None:
    """list_user_jobs returns only jobs for the specified user."""
    user = await make_user(email=f"listjobs-{uuid4().hex[:6]}@example.com")
    for _ in range(3):
        await make_job(user=user)
    jobs = await storage_repo.list_user_jobs(user.id)
    assert len(jobs) >= 3
    assert all(j.user_id == user.id for j in jobs)


async def test_list_user_jobs_filter_by_status(
    storage_repo: StorageRepository, make_user, make_job
) -> None:
    """list_user_jobs filters by status when provided."""
    user = await make_user(email=f"statusjobs-{uuid4().hex[:6]}@example.com")
    await make_job(user=user, status="pending")
    await make_job(user=user, status="completed")
    pending_jobs = await storage_repo.list_user_jobs(user.id, status=JobStatus.PENDING)
    assert all(j.status == "pending" for j in pending_jobs)


async def test_list_user_jobs_offset_past_end(storage_repo: StorageRepository, make_user) -> None:
    """list_user_jobs with offset beyond count returns empty list."""
    user = await make_user(email=f"nojobsoff-{uuid4().hex[:6]}@example.com")
    jobs = await storage_repo.list_user_jobs(user.id, offset=1000)
    assert not list(jobs)


# ---------------------------------------------------------------------------
# GenerationOutput — create / get / list / delete
# ---------------------------------------------------------------------------


async def test_create_output(storage_repo: StorageRepository, make_user, make_job) -> None:
    """create_output persists and returns a GenerationOutput."""
    user = await make_user(email=f"outcreate-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    out_id = uuid4()
    output = await storage_repo.create_output(
        id=out_id,
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{out_id}.png",
        content_type="image/png",
        size_bytes=8192,
        format="png",
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    assert output.id == out_id
    assert output.job_id == job.id


async def test_create_output_invalid_job_raises(storage_repo: StorageRepository, make_user) -> None:
    """create_output with a non-existent job_id raises IntegrityError (FK violation)."""
    user = await make_user(email=f"outfk-{uuid4().hex[:6]}@example.com")
    fake_job_id = uuid4()
    out_id = uuid4()
    with pytest.raises(IntegrityError):
        await storage_repo.create_output(
            id=out_id,
            user_id=user.id,
            job_id=fake_job_id,
            storage_key=f"users/{user.id}/outputs/{fake_job_id}/{out_id}.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            output_index=0,
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
        )


async def test_get_output_found(storage_repo: StorageRepository, make_user, make_job) -> None:
    """get_output returns the output by PK."""
    user = await make_user(email=f"outget-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    out_id = uuid4()
    output = await storage_repo.create_output(
        id=out_id,
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{out_id}.png",
        content_type="image/png",
        size_bytes=100,
        format="png",
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    found = await storage_repo.get_output(output.id)
    assert found is not None
    assert found.id == output.id


async def test_get_output_not_found(storage_repo: StorageRepository) -> None:
    """get_output returns None for unknown UUID."""
    assert await storage_repo.get_output(uuid4()) is None


async def test_get_output_ownership_enforced(
    storage_repo: StorageRepository, make_user, make_job
) -> None:
    """get_output with wrong user_id returns None."""
    user = await make_user(email=f"outowner-{uuid4().hex[:6]}@example.com")
    other = await make_user(email=f"outother-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    out_id = uuid4()
    output = await storage_repo.create_output(
        id=out_id,
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{out_id}.png",
        content_type="image/png",
        size_bytes=100,
        format="png",
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    assert await storage_repo.get_output(output.id, user_id=other.id) is None


async def test_list_job_outputs(storage_repo: StorageRepository, make_user, make_job) -> None:
    """list_job_outputs returns all outputs for a job ordered by output_index."""
    user = await make_user(email=f"outlist-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    for i in range(3):
        out_id = uuid4()
        await storage_repo.create_output(
            id=out_id,
            user_id=user.id,
            job_id=job.id,
            storage_key=f"users/{user.id}/outputs/{job.id}/{out_id}.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            output_index=i,
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
        )
    outputs = await storage_repo.list_job_outputs(job.id)
    assert len(outputs) == 3
    assert [o.output_index for o in outputs] == [0, 1, 2]


async def test_list_job_outputs_no_outputs(storage_repo: StorageRepository, make_job) -> None:
    """list_job_outputs returns empty list for a job with no outputs."""
    job = await make_job()
    outputs = await storage_repo.list_job_outputs(job.id)
    assert not list(outputs)


async def test_delete_output(storage_repo: StorageRepository, make_user, make_job) -> None:
    """delete_output removes the row and returns True."""
    user = await make_user(email=f"outdel-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    out_id = uuid4()
    output = await storage_repo.create_output(
        id=out_id,
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{out_id}.png",
        content_type="image/png",
        size_bytes=100,
        format="png",
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    assert await storage_repo.delete_output(output.id) is True
    assert await storage_repo.get_output(output.id) is None


async def test_delete_output_not_found_returns_false(
    storage_repo: StorageRepository,
) -> None:
    """delete_output returns False for an unknown output."""
    assert await storage_repo.delete_output(uuid4()) is False


async def test_delete_outputs_batch(storage_repo: StorageRepository, make_user, make_job) -> None:
    """delete_outputs_batch deletes multiple outputs and returns the count."""
    user = await make_user(email=f"batchdel-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    output_ids = []
    for i in range(3):
        out_id = uuid4()
        await storage_repo.create_output(
            id=out_id,
            user_id=user.id,
            job_id=job.id,
            storage_key=f"users/{user.id}/outputs/{job.id}/{out_id}.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            output_index=i,
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
        )
        output_ids.append(out_id)
    deleted = await storage_repo.delete_outputs_batch(output_ids)
    assert deleted == 3


async def test_get_expired_outputs(storage_repo: StorageRepository, make_user, make_job) -> None:
    """get_expired_outputs returns outputs past their expires_at."""
    user = await make_user(email=f"expout-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    past = datetime.now(UTC) - timedelta(hours=1)
    out_id = uuid4()
    await storage_repo.create_output(
        id=out_id,
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{out_id}.png",
        content_type="image/png",
        size_bytes=100,
        format="png",
        output_index=0,
        expires_at=past,
        product_id="vex",
    )
    expired = await storage_repo.get_expired_outputs()
    assert any(o.id == out_id for o in expired)


# ---------------------------------------------------------------------------
# Storage statistics
# ---------------------------------------------------------------------------


async def test_get_user_storage_stats_empty(storage_repo: StorageRepository, make_user) -> None:
    """get_user_storage_stats returns zeros for a user with no files."""
    user = await make_user(email=f"statszero-{uuid4().hex[:6]}@example.com")
    stats = await storage_repo.get_user_storage_stats(user.id)
    assert stats == {"upload_count": 0, "output_count": 0, "total_bytes": 0}


async def test_get_user_storage_stats_counts_files(
    storage_repo: StorageRepository, make_user, make_user_image, make_job
) -> None:
    """get_user_storage_stats reflects upload and output counts."""
    user = await make_user(email=f"statsfull-{uuid4().hex[:6]}@example.com")
    await make_user_image(user=user, size_bytes=1000)
    job = await make_job(user=user)
    out_id = uuid4()
    await storage_repo.create_output(
        id=out_id,
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{out_id}.png",
        content_type="image/png",
        size_bytes=2000,
        format="png",
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    stats = await storage_repo.get_user_storage_stats(user.id)
    assert stats["upload_count"] >= 1
    assert stats["output_count"] >= 1
    assert stats["total_bytes"] >= 3000


# ---------------------------------------------------------------------------
# Cascade: delete job cascades to outputs
# ---------------------------------------------------------------------------


async def test_cascade_delete_job_removes_outputs(
    storage_repo: StorageRepository, make_user, make_job, db_session: AsyncSession
) -> None:
    """Deleting a GenerationJob cascades to its GenerationOutput rows."""
    user = await make_user(email=f"cascade-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    out_id = uuid4()
    await storage_repo.create_output(
        id=out_id,
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{out_id}.png",
        content_type="image/png",
        size_bytes=100,
        format="png",
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )

    job_obj = await storage_repo.get_job(job.id)
    assert job_obj is not None
    await db_session.delete(job_obj)
    await db_session.flush()

    outputs = await storage_repo.list_job_outputs(job.id)
    assert not list(outputs)


# ---------------------------------------------------------------------------
# Cleanup count accuracy
# ---------------------------------------------------------------------------


async def test_expired_images_count_accuracy(storage_repo: StorageRepository, make_user) -> None:
    """Expired image query returns exactly N expired items regardless of non-expired ones."""
    user = await make_user(email=f"expcount-{uuid4().hex[:6]}@example.com")
    past = datetime.now(UTC) - timedelta(hours=2)
    future = datetime.now(UTC) + timedelta(days=7)

    # Create 3 expired and 2 non-expired
    expired_ids = set()
    for i in range(3):
        img_id = uuid4()
        await storage_repo.create_user_image(
            id=img_id,
            user_id=user.id,
            storage_key=f"users/{user.id}/uploads/exp{i}.png",
            original_filename=f"exp{i}.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            expires_at=past,
            product_id="vex",
        )
        expired_ids.add(img_id)

    for i in range(2):
        img_id = uuid4()
        await storage_repo.create_user_image(
            id=img_id,
            user_id=user.id,
            storage_key=f"users/{user.id}/uploads/fresh{i}.png",
            original_filename=f"fresh{i}.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            expires_at=future,
            product_id="vex",
        )

    expired = await storage_repo.get_expired_images()
    expired_from_this_user = {img.id for img in expired if img.user_id == user.id}
    assert expired_from_this_user == expired_ids
