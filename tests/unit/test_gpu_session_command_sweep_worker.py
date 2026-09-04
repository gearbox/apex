"""Unit tests for GpuSessionCommandSweepWorker (P6, invariant #14).

Session factory is mocked; GpuSessionCommandRepository/GpuSessionOperationRepository
are patched at the worker module level.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.api.services.gpu_session.command_sweep_worker import GpuSessionCommandSweepWorker
from src.core.enums import CommandStatus
from src.db.models.gpu_session_command import GpuSessionCommand

_COMMAND_REPO = "src.api.services.gpu_session.command_sweep_worker.GpuSessionCommandRepository"
_OPERATION_REPO = "src.api.services.gpu_session.command_sweep_worker.GpuSessionOperationRepository"


class _StubSettings:
    gpu_command_sweep_interval_seconds = 60


def _session_factory() -> MagicMock:
    db = MagicMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=None)
    begin = MagicMock()
    begin.__aenter__ = AsyncMock(return_value=None)
    begin.__aexit__ = AsyncMock(return_value=None)
    db.begin.return_value = begin
    return MagicMock(return_value=db)


def _make_worker() -> GpuSessionCommandSweepWorker:
    return GpuSessionCommandSweepWorker(
        session_factory=_session_factory(),  # type: ignore[arg-type]
        settings=_StubSettings(),  # type: ignore[arg-type]
        redis_enabled=False,
        redis_client_factory=MagicMock(),
    )


def _expired_command(*, claimed_at: datetime, deadline_at: datetime) -> GpuSessionCommand:
    return GpuSessionCommand(
        id=uuid4(),
        session_id=uuid4(),
        product_id="vex",
        operation_id=uuid4(),
        kind="bundle_provision",
        payload={},
        status=CommandStatus.expired,
        agent_id="agent-a",
        claimed_at=claimed_at,
        deadline_at=deadline_at,
    )


async def test_run_once_fails_operations_for_every_expired_command() -> None:
    worker = _make_worker()
    now = datetime.now(UTC)
    claimed_at = now - timedelta(minutes=70)
    deadline_at = now - timedelta(minutes=10)
    command = _expired_command(claimed_at=claimed_at, deadline_at=deadline_at)

    with patch(_COMMAND_REPO) as CommandRepo, patch(_OPERATION_REPO) as OperationRepo:
        command_repo = AsyncMock()
        CommandRepo.return_value = command_repo
        command_repo.expire_overdue.return_value = [command]
        operation_repo = AsyncMock()
        OperationRepo.return_value = operation_repo

        await worker.run_once()

    command_repo.expire_overdue.assert_awaited_once()
    operation_repo.close_failed.assert_awaited_once()
    call = operation_repo.close_failed.await_args
    assert call.args[0] == command.operation_id
    assert "4200" in call.kwargs["error"]  # 70 minutes, in seconds
    assert "bundle_provision" in call.kwargs["error"]


async def test_run_once_is_a_no_op_when_nothing_expired() -> None:
    worker = _make_worker()

    with patch(_COMMAND_REPO) as CommandRepo, patch(_OPERATION_REPO) as OperationRepo:
        command_repo = AsyncMock()
        CommandRepo.return_value = command_repo
        command_repo.expire_overdue.return_value = []
        operation_repo = AsyncMock()
        OperationRepo.return_value = operation_repo

        await worker.run_once()

    operation_repo.close_failed.assert_not_awaited()
