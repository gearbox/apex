"""Integration tests for OutputRepository against a real PostgreSQL database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.repositories.output import OutputRepository


async def test_create_output(output_repo: OutputRepository, make_user, make_job) -> None:
    """create persists and returns a GenerationOutput."""
    user = await make_user(email=f"outcreate-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    out_id = uuid4()
    output = await output_repo.create(
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


async def test_create_output_invalid_job_raises(output_repo: OutputRepository, make_user) -> None:
    """create with a non-existent job_id raises IntegrityError (FK violation)."""
    user = await make_user(email=f"outfk-{uuid4().hex[:6]}@example.com")
    fake_job_id = uuid4()
    out_id = uuid4()
    with pytest.raises(IntegrityError):
        await output_repo.create(
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


async def test_get_output_found(output_repo: OutputRepository, make_user, make_job) -> None:
    """get returns the output by PK."""
    user = await make_user(email=f"outget-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    out_id = uuid4()
    output = await output_repo.create(
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
    found = await output_repo.get(output.id)
    assert found is not None
    assert found.id == output.id


async def test_get_output_not_found(output_repo: OutputRepository) -> None:
    """get returns None for unknown UUID."""
    assert await output_repo.get(uuid4()) is None


async def test_get_output_ownership_enforced(
    output_repo: OutputRepository, make_user, make_job
) -> None:
    """get with wrong user_id returns None."""
    user = await make_user(email=f"outowner-{uuid4().hex[:6]}@example.com")
    other = await make_user(email=f"outother-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    out_id = uuid4()
    output = await output_repo.create(
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
    assert await output_repo.get(output.id, user_id=other.id) is None


async def test_list_by_job(output_repo: OutputRepository, make_user, make_job) -> None:
    """list_by_job returns all outputs for a job ordered by output_index."""
    user = await make_user(email=f"outlist-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    for i in range(3):
        out_id = uuid4()
        await output_repo.create(
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
    outputs = await output_repo.list_by_job(job.id)
    assert len(outputs) == 3
    assert [o.output_index for o in outputs] == [0, 1, 2]


async def test_list_by_job_no_outputs(output_repo: OutputRepository, make_job) -> None:
    """list_by_job returns empty list for a job with no outputs."""
    job = await make_job()
    outputs = await output_repo.list_by_job(job.id)
    assert not list(outputs)


async def test_delete_output(output_repo: OutputRepository, make_user, make_job) -> None:
    """delete removes the row and returns True."""
    user = await make_user(email=f"outdel-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    out_id = uuid4()
    output = await output_repo.create(
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
    assert await output_repo.delete(output.id) is True
    assert await output_repo.get(output.id) is None


async def test_delete_output_not_found_returns_false(
    output_repo: OutputRepository,
) -> None:
    """delete returns False for an unknown output."""
    assert await output_repo.delete(uuid4()) is False


async def test_delete_batch(output_repo: OutputRepository, make_user, make_job) -> None:
    """delete_batch deletes multiple outputs and returns the count."""
    user = await make_user(email=f"batchdel-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    output_ids = []
    for i in range(3):
        out_id = uuid4()
        await output_repo.create(
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
    deleted = await output_repo.delete_batch(output_ids)
    assert deleted == 3


async def test_get_expired(output_repo: OutputRepository, make_user, make_job) -> None:
    """get_expired returns outputs past their expires_at."""
    user = await make_user(email=f"expout-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    past = datetime.now(UTC) - timedelta(hours=1)
    out_id = uuid4()
    await output_repo.create(
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
    expired = await output_repo.get_expired()
    assert any(o.id == out_id for o in expired)


async def test_count_and_sum_by_user_with_data(
    output_repo: OutputRepository, make_user, make_job
) -> None:
    """count_and_sum_by_user returns correct count and byte total."""
    from datetime import UTC, datetime, timedelta
    from uuid import uuid4 as _uuid4

    user = await make_user(email=f"outstats-{_uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)

    await output_repo.create(
        id=_uuid4(),
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{_uuid4()}.png",
        content_type="image/png",
        size_bytes=5000,
        format="png",
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    await output_repo.create(
        id=_uuid4(),
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{_uuid4()}.png",
        content_type="image/png",
        size_bytes=3000,
        format="png",
        output_index=1,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )

    count, total_bytes = await output_repo.count_and_sum_by_user(user.id)
    assert count == 2
    assert total_bytes == 8000


async def test_count_and_sum_by_user_empty(output_repo: OutputRepository, make_user) -> None:
    """count_and_sum_by_user returns zeros for a user with no outputs."""
    user = await make_user(email=f"outstatsempty-{uuid4().hex[:6]}@example.com")
    count, total_bytes = await output_repo.count_and_sum_by_user(user.id)
    assert count == 0
    assert total_bytes == 0


async def test_cascade_delete_job_removes_outputs(
    output_repo: OutputRepository, make_user, make_job, db_session: AsyncSession
) -> None:
    """Deleting a GenerationJob cascades to its GenerationOutput rows."""
    from src.db.repositories.job import JobRepository

    user = await make_user(email=f"cascade-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    out_id = uuid4()
    await output_repo.create(
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

    job_repo = JobRepository(db_session)
    job_obj = await job_repo.get(job.id)
    assert job_obj is not None
    await db_session.delete(job_obj)
    await db_session.flush()

    outputs = await output_repo.list_by_job(job.id)
    assert not list(outputs)
