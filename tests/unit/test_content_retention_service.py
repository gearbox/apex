"""Unit tests for ContentRetentionService with mocked sessions/repositories."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.content_retention import ContentRetentionService
from src.api.services.storage.exceptions import StorageDeleteError

pytestmark = pytest.mark.unit


def _make_session() -> AsyncMock:
    session = AsyncMock()
    session.commit = AsyncMock()
    return session


def _exec_result(rowcount: int) -> MagicMock:
    result = MagicMock()
    result.rowcount = rowcount
    return result


def _cm(session: AsyncMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_session_factory(sessions: list[AsyncMock]) -> MagicMock:
    """Factory whose successive calls hand out each session in order, as async CMs."""
    iterator = iter(sessions)
    return MagicMock(side_effect=lambda: _cm(next(iterator)))


def _make_auto_session_factory(rowcount: int = 1) -> MagicMock:
    """Factory that manufactures a fresh session on every call.

    Every ``session.execute()`` call on any produced session returns the
    same fixed rowcount — useful when only batch counts/draining behavior
    matter, not exact per-row bookkeeping.
    """

    def _factory() -> MagicMock:
        session = _make_session()
        session.execute = AsyncMock(return_value=_exec_result(rowcount))
        return _cm(session)

    return MagicMock(side_effect=_factory)


def _make_default_storage() -> AsyncMock:
    """Default storage mock: delete_many succeeds, confirming every key attempted."""

    async def _delete_many(keys: list[str]) -> int:
        return len(keys)

    storage = AsyncMock()
    storage.delete_many = AsyncMock(side_effect=_delete_many)
    return storage


def _make_service(
    *,
    batch_size: int = 500,
    max_batches_per_run: int = 20,
    storage: AsyncMock | None = None,
    session_factory: MagicMock | None = None,
) -> ContentRetentionService:
    return ContentRetentionService(
        session_factory=session_factory or _make_auto_session_factory(),
        storage=storage or _make_default_storage(),
        batch_size=batch_size,
        max_batches_per_run=max_batches_per_run,
    )


def _patch_repos(
    *,
    output_expired: list[object] | None = None,
    output_derivatives: dict[object, list[object]] | None = None,
    upload_expired: list[object] | None = None,
    upload_derivatives: dict[object, list[object]] | None = None,
):
    output_repo = AsyncMock()
    output_repo.get_expired = AsyncMock(return_value=output_expired or [])
    output_repo.batch_derivatives = AsyncMock(return_value=output_derivatives or {})

    image_repo = AsyncMock()
    image_repo.get_expired = AsyncMock(return_value=upload_expired or [])
    image_repo.batch_derivatives = AsyncMock(return_value=upload_derivatives or {})

    return patch.multiple(
        "src.api.services.content_retention",
        OutputRepository=MagicMock(return_value=output_repo),
        UserImageRepository=MagicMock(return_value=image_repo),
    )


async def test_sweep_noop_when_nothing_expired() -> None:
    storage = AsyncMock()

    with _patch_repos():
        service = _make_service(storage=storage)
        result = await service.sweep()

    assert result.outputs_deleted == 0
    assert result.uploads_deleted == 0
    assert result.jobs_soft_deleted == 0
    assert result.r2_keys_deleted == 0
    assert result.r2_delete_failed is False
    assert result.batches == 0
    storage.delete_many.assert_not_awaited()


async def test_sweep_deletes_expired_outputs_and_collects_r2_keys() -> None:
    output_id = uuid4()
    job_id = uuid4()
    parent = MagicMock(id=output_id, job_id=job_id, storage_key="users/u/outputs/j/out.png")
    derivative = MagicMock(storage_key="users/u/outputs/j/out_sm.webp")

    session = _make_session()
    # delete, soft-delete UPDATE, metadata purge DELETE (D14), tags purge DELETE
    session.execute = AsyncMock(
        side_effect=[_exec_result(1), _exec_result(0), _exec_result(0), _exec_result(0)]
    )
    uploads_session = _make_session()

    storage = AsyncMock()
    storage.delete_many = AsyncMock(return_value=2)
    factory = _make_session_factory([session, uploads_session])

    with _patch_repos(
        output_expired=[parent],
        output_derivatives={output_id: [derivative]},
    ):
        service = _make_service(storage=storage, session_factory=factory)
        result = await service.sweep()

    assert result.outputs_deleted == 1
    assert result.r2_keys_deleted == 2
    assert result.batches == 1
    storage.delete_many.assert_awaited_once()
    (keys,), _ = storage.delete_many.call_args
    assert set(keys) == {parent.storage_key, derivative.storage_key}


async def test_sweep_soft_deletes_jobs_with_no_remaining_outputs() -> None:
    output_id = uuid4()
    job_id = uuid4()
    parent = MagicMock(id=output_id, job_id=job_id, storage_key="users/u/outputs/j/out.png")

    session = _make_session()
    # delete rowcount=1, soft-delete UPDATE rowcount=1 -> job had no remaining outputs,
    # metadata purge DELETE (D14), tags purge DELETE
    session.execute = AsyncMock(
        side_effect=[_exec_result(1), _exec_result(1), _exec_result(0), _exec_result(0)]
    )
    uploads_session = _make_session()
    factory = _make_session_factory([session, uploads_session])

    with _patch_repos(output_expired=[parent]):
        service = _make_service(session_factory=factory)
        result = await service.sweep()

    assert result.outputs_deleted == 1
    assert result.jobs_soft_deleted == 1


async def test_sweep_keeps_jobs_with_remaining_outputs() -> None:
    """When the UPDATE affects 0 rows, jobs_soft_deleted must reflect that —
    not just "how many jobs were touched by this batch" — because a job
    with one expired output and one still-live output must stay
    is_deleted=False (enforced by the NOT EXISTS predicate at the SQL level;
    the mocked rowcount here stands in for "the job still has an output").
    """
    output_id = uuid4()
    job_id = uuid4()
    parent = MagicMock(id=output_id, job_id=job_id, storage_key="users/u/outputs/j/out.png")

    session = _make_session()
    session.execute = AsyncMock(
        side_effect=[_exec_result(1), _exec_result(0), _exec_result(0), _exec_result(0)]
    )
    uploads_session = _make_session()
    factory = _make_session_factory([session, uploads_session])

    with _patch_repos(output_expired=[parent]):
        service = _make_service(session_factory=factory)
        result = await service.sweep()

    assert result.outputs_deleted == 1
    assert result.jobs_soft_deleted == 0


async def test_sweep_deletes_expired_uploads_with_derivatives() -> None:
    image_id = uuid4()
    parent = MagicMock(id=image_id, storage_key="users/u/uploads/img.png")
    derivative = MagicMock(storage_key="users/u/uploads/img_sm.webp")

    outputs_session = _make_session()
    uploads_session = _make_session()
    uploads_session.execute = AsyncMock(return_value=_exec_result(1))
    factory = _make_session_factory([outputs_session, uploads_session])

    storage = AsyncMock()
    storage.delete_many = AsyncMock(return_value=2)

    with _patch_repos(
        upload_expired=[parent],
        upload_derivatives={image_id: [derivative]},
    ):
        service = _make_service(storage=storage, session_factory=factory)
        result = await service.sweep()

    assert result.uploads_deleted == 1
    assert result.r2_keys_deleted == 2
    assert result.batches == 1
    storage.delete_many.assert_awaited_once()
    (keys,), _ = storage.delete_many.call_args
    assert set(keys) == {parent.storage_key, derivative.storage_key}


async def test_sweep_r2_failure_does_not_raise() -> None:
    output_id = uuid4()
    job_id = uuid4()
    parent = MagicMock(id=output_id, job_id=job_id, storage_key="users/u/outputs/j/out.png")

    session = _make_session()
    session.execute = AsyncMock(
        side_effect=[_exec_result(1), _exec_result(0), _exec_result(0), _exec_result(0)]
    )
    uploads_session = _make_session()
    factory = _make_session_factory([session, uploads_session])

    storage = AsyncMock()
    storage.delete_many = AsyncMock(side_effect=StorageDeleteError("boom"))

    with _patch_repos(output_expired=[parent]):
        service = _make_service(storage=storage, session_factory=factory)
        result = await service.sweep()  # must not raise

    assert result.outputs_deleted == 1
    assert result.r2_keys_deleted == 0
    assert result.r2_delete_failed is True
    session.commit.assert_awaited_once()


async def test_sweep_drains_backlog_across_batches() -> None:
    batches = [
        [MagicMock(id=uuid4(), job_id=uuid4(), storage_key=f"k{i}") for i in range(2)],
        [MagicMock(id=uuid4(), job_id=uuid4(), storage_key=f"k{i}") for i in range(2)],
        [MagicMock(id=uuid4(), job_id=uuid4(), storage_key=f"k{i}") for i in range(2)],
        [MagicMock(id=uuid4(), job_id=uuid4(), storage_key="k_last")],
    ]

    output_repo = AsyncMock()
    output_repo.get_expired = AsyncMock(side_effect=[*batches, []])
    output_repo.batch_derivatives = AsyncMock(return_value={})

    image_repo = AsyncMock()
    image_repo.get_expired = AsyncMock(return_value=[])

    factory = _make_auto_session_factory(rowcount=1)

    with patch.multiple(
        "src.api.services.content_retention",
        OutputRepository=MagicMock(return_value=output_repo),
        UserImageRepository=MagicMock(return_value=image_repo),
    ):
        service = _make_service(batch_size=2, max_batches_per_run=10, session_factory=factory)
        result = await service.sweep()

    assert result.batches == 4
    assert result.outputs_deleted == 4  # 1 rowcount per batch x 4 batches


async def test_sweep_respects_max_batches_per_run() -> None:
    full_batch = [MagicMock(id=uuid4(), job_id=uuid4(), storage_key="k") for _ in range(2)]

    output_repo = AsyncMock()
    output_repo.get_expired = AsyncMock(return_value=full_batch)  # always a full batch
    output_repo.batch_derivatives = AsyncMock(return_value={})

    image_repo = AsyncMock()
    image_repo.get_expired = AsyncMock(return_value=[])

    factory = _make_auto_session_factory(rowcount=1)

    with patch.multiple(
        "src.api.services.content_retention",
        OutputRepository=MagicMock(return_value=output_repo),
        UserImageRepository=MagicMock(return_value=image_repo),
    ):
        service = _make_service(batch_size=2, max_batches_per_run=3, session_factory=factory)
        result = await service.sweep()

    assert result.batches == 3
    assert output_repo.get_expired.await_count == 3
