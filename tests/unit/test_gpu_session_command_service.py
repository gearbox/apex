"""Unit tests for GpuSessionCommandService: enqueue/enqueue_batch validation and
atomicity (D24/D25), and the claim() auth/session-state logic (D29).

The session factory is mocked so no real DB is required; GpuSessionRepository/
GpuSessionOperationRepository/GpuSessionCommandRepository are patched at the
service module level. Real SKIP LOCKED/unique-index behavior of claim() itself is
covered by tests/integration/test_gpu_session_command_repository.py against Postgres.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from src.api.services.gpu_session.command_payload import (
    CommandBuildError,
    ProvisionCommand,
    RemovalCommand,
    RestartCommand,
)
from src.api.services.gpu_session.command_service import (
    CommandEnqueueSessionError,
    GpuSessionCommandService,
)
from src.core.enums import CommandStatus, GpuSessionStatus
from src.db.models.gpu_session import GpuSession
from src.db.models.gpu_session_command import GpuSessionCommand

_SESSION_REPO = "src.api.services.gpu_session.command_service.GpuSessionRepository"
_OPERATION_REPO = "src.api.services.gpu_session.command_service.GpuSessionOperationRepository"
_COMMAND_REPO = "src.api.services.gpu_session.command_service.GpuSessionCommandRepository"
_TOKEN = "callback-token"


class _StubSettings:
    gpu_command_provision_timeout_seconds = 3600
    gpu_command_removal_timeout_seconds = 600
    gpu_command_restart_timeout_seconds = 300


def _session_factory() -> MagicMock:
    db = MagicMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=None)
    db.get = AsyncMock(return_value=_gpu_session(session_id=uuid4()))
    begin = MagicMock()
    begin.__aenter__ = AsyncMock(return_value=None)
    begin.__aexit__ = AsyncMock(return_value=None)
    db.begin.return_value = begin
    return MagicMock(return_value=db)


def _make_service(factory: MagicMock | None = None) -> GpuSessionCommandService:
    return GpuSessionCommandService(
        session_factory=factory or _session_factory(),  # type: ignore[arg-type]
        settings=_StubSettings(),  # type: ignore[arg-type]
    )


def _gpu_session(
    *, session_id: object, status: GpuSessionStatus | str = GpuSessionStatus.active
) -> GpuSession:
    return GpuSession(
        id=session_id,
        user_id=uuid4(),
        product_id="vex",
        status=status,
        bundle_name="wan_2.2_i2v",
        model_type="aisha-image",
        callback_token_hash=hashlib.sha256(_TOKEN.encode()).hexdigest(),
    )


# ---------------------------------------------------------------------------
# enqueue / enqueue_batch — D24 (reject before write) / D25 (atomic)
# ---------------------------------------------------------------------------


class TestEnqueue:
    async def test_malformed_command_never_touches_the_db(self) -> None:
        factory = _session_factory()
        service = _make_service(factory)

        with pytest.raises(CommandBuildError):
            await service.enqueue(
                session_id=uuid4(),
                product_id="vex",
                command=ProvisionCommand(bundle="", mode="full"),
            )

        factory.assert_not_called()

    async def test_valid_provision_creates_operation_then_command_atomically(self) -> None:
        session_id = uuid4()
        service = _make_service()

        with patch(_OPERATION_REPO) as OperationRepo, patch(_COMMAND_REPO) as CommandRepo:
            operation_repo = AsyncMock()
            OperationRepo.return_value = operation_repo
            command_repo = AsyncMock()
            CommandRepo.return_value = command_repo
            command_repo.create.return_value = MagicMock(spec=GpuSessionCommand)

            await service.enqueue(
                session_id=session_id,
                product_id="vex",
                command=ProvisionCommand(bundle="wan_2.2_i2v:260105-01", mode="additive"),
            )

        operation_repo.create.assert_awaited_once()
        command_repo.create.assert_awaited_once()
        op_kwargs = operation_repo.create.await_args.kwargs
        cmd_kwargs = command_repo.create.await_args.kwargs
        # Same operation_id/command_id link both rows (D25's forward slot).
        assert op_kwargs["command_id"] == cmd_kwargs["id"]
        assert op_kwargs["id"] == cmd_kwargs["operation_id"]
        assert op_kwargs["target_bundle"] == "wan_2.2_i2v"
        assert op_kwargs["target_bundle_version"] == "260105-01"
        assert op_kwargs["target_mode"] == "additive"
        assert cmd_kwargs["payload"]["bundle"] == "wan_2.2_i2v:260105-01"

    async def test_removal_and_restart_do_not_set_target_bundle(self) -> None:
        service = _make_service()

        with patch(_OPERATION_REPO) as OperationRepo, patch(_COMMAND_REPO) as CommandRepo:
            operation_repo = AsyncMock()
            OperationRepo.return_value = operation_repo
            CommandRepo.return_value = AsyncMock()

            await service.enqueue(
                session_id=uuid4(), product_id="vex", command=RemovalCommand(bundle="old_bundle")
            )

        op_kwargs = operation_repo.create.await_args.kwargs
        assert op_kwargs["target_bundle"] is None
        assert op_kwargs["target_mode"] is None

    async def test_missing_session_is_rejected_before_creating_an_operation(self) -> None:
        service = _make_service()
        with patch(_SESSION_REPO) as SessionRepo, patch(_OPERATION_REPO) as OperationRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            session_repo.get_by_id.return_value = None
            operation_repo = AsyncMock()
            OperationRepo.return_value = operation_repo

            with pytest.raises(CommandEnqueueSessionError, match="does not exist"):
                await service.enqueue(
                    session_id=uuid4(),
                    product_id="vex",
                    command=ProvisionCommand(bundle="wan_2.2_i2v", mode="full"),
                )

        operation_repo.create.assert_not_awaited()


class TestEnqueueBatch:
    async def test_empty_batch_is_rejected(self) -> None:
        service = _make_service()
        with pytest.raises(CommandBuildError, match="at least one command"):
            await service.enqueue_batch(session_id=uuid4(), product_id="vex", commands=[])

    async def test_a_malformed_member_leaves_nothing_written(self) -> None:
        """Invariant #12: enqueue_batch validates every member before writing any."""
        factory = _session_factory()
        service = _make_service(factory)
        commands = [
            ProvisionCommand(bundle="wan_2.2_i2v", mode="full"),
            ProvisionCommand(bundle="", mode="full"),  # invalid — 2nd of 2
        ]

        with pytest.raises(CommandBuildError):
            await service.enqueue_batch(session_id=uuid4(), product_id="vex", commands=commands)

        factory.assert_not_called()

    async def test_batch_assigns_index_and_total_per_member(self) -> None:
        service = _make_service()
        commands = [
            ProvisionCommand(bundle="a", mode="full"),
            RemovalCommand(bundle="b"),
            RestartCommand(),
        ]

        with patch(_OPERATION_REPO) as OperationRepo, patch(_COMMAND_REPO) as CommandRepo:
            OperationRepo.return_value = AsyncMock()
            command_repo = AsyncMock()
            CommandRepo.return_value = command_repo
            command_repo.create.return_value = MagicMock(spec=GpuSessionCommand)

            created = await service.enqueue_batch(
                session_id=uuid4(), product_id="vex", commands=commands
            )

        assert len(created) == 3
        calls = command_repo.create.await_args_list
        batch_ids = {call.kwargs["batch_id"] for call in calls}
        assert len(batch_ids) == 1  # one batch_id shared across all members
        assert [call.kwargs["batch_index"] for call in calls] == [0, 1, 2]
        assert all(call.kwargs["batch_total"] == 3 for call in calls)

    async def test_declared_model_bytes_lands_only_on_index_0(self) -> None:
        service = _make_service()
        commands = [
            ProvisionCommand(bundle="a", mode="full"),
            ProvisionCommand(bundle="b", mode="full"),
        ]

        with patch(_OPERATION_REPO) as OperationRepo, patch(_COMMAND_REPO) as CommandRepo:
            OperationRepo.return_value = AsyncMock()
            command_repo = AsyncMock()
            CommandRepo.return_value = command_repo
            command_repo.create.return_value = MagicMock(spec=GpuSessionCommand)

            await service.enqueue_batch(
                session_id=uuid4(),
                product_id="vex",
                commands=commands,
                declared_model_bytes=1024,
            )

        calls = command_repo.create.await_args_list
        assert calls[0].kwargs["payload"]["batch_declared_bytes"] == 1024
        assert "batch_declared_bytes" not in calls[1].kwargs["payload"]

    async def test_declared_model_bytes_requires_provision_at_index_0(self) -> None:
        service = _make_service()
        commands = [RemovalCommand(bundle="a")]

        with pytest.raises(CommandBuildError, match="index 0"):
            await service.enqueue_batch(
                session_id=uuid4(),
                product_id="vex",
                commands=commands,
                declared_model_bytes=1024,
            )

    async def test_per_member_declared_model_bytes_is_rejected(self) -> None:
        """declared_model_bytes must be set via enqueue_batch's own kwarg, not
        pre-set on an individual ProvisionCommand — one source of truth (F8)."""
        service = _make_service()
        commands = [ProvisionCommand(bundle="a", mode="full", declared_model_bytes=999)]

        with pytest.raises(CommandBuildError, match="enqueue_batch's own parameter"):
            await service.enqueue_batch(session_id=uuid4(), product_id="vex", commands=commands)

    async def test_terminal_session_is_rejected_before_creating_batch_operations(self) -> None:
        session_id = uuid4()
        service = _make_service()
        with patch(_SESSION_REPO) as SessionRepo, patch(_OPERATION_REPO) as OperationRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            session_repo.get_by_id.return_value = _gpu_session(
                session_id=session_id, status=GpuSessionStatus.stopped
            )
            operation_repo = AsyncMock()
            OperationRepo.return_value = operation_repo

            with pytest.raises(CommandEnqueueSessionError, match="is terminal"):
                await service.enqueue_batch(
                    session_id=session_id,
                    product_id="vex",
                    commands=[ProvisionCommand(bundle="wan_2.2_i2v", mode="full")],
                )

        operation_repo.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# claim — D29 status codes
# ---------------------------------------------------------------------------


class TestClaim:
    async def test_session_with_no_stored_token_hash_returns_401(self) -> None:
        session_id = uuid4()
        service = _make_service()
        session = _gpu_session(session_id=session_id)
        session.callback_token_hash = None
        with patch(_SESSION_REPO) as SessionRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            session_repo.get_by_id.return_value = session

            status, envelope = await service.claim(
                session_id=session_id, bearer_token=_TOKEN, agent_id="agent-a"
            )

        assert (status, envelope) == (401, None)

    async def test_unknown_session_returns_401(self) -> None:
        service = _make_service()
        with patch(_SESSION_REPO) as SessionRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            session_repo.get_by_id.return_value = None

            status, envelope = await service.claim(
                session_id=uuid4(), bearer_token=_TOKEN, agent_id="agent-a"
            )

        assert (status, envelope) == (401, None)

    async def test_wrong_token_returns_401(self) -> None:
        session_id = uuid4()
        service = _make_service()
        with patch(_SESSION_REPO) as SessionRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            session_repo.get_by_id.return_value = _gpu_session(session_id=session_id)

            status, envelope = await service.claim(
                session_id=session_id, bearer_token="wrong", agent_id="agent-a"
            )

        assert (status, envelope) == (401, None)

    @pytest.mark.parametrize("status", [GpuSessionStatus.stopped, GpuSessionStatus.failed])
    async def test_terminal_session_returns_204_not_404_or_409(
        self, status: GpuSessionStatus
    ) -> None:
        session_id = uuid4()
        service = _make_service()
        with patch(_SESSION_REPO) as SessionRepo, patch(_COMMAND_REPO) as CommandRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            session_repo.get_by_id.return_value = _gpu_session(session_id=session_id, status=status)
            command_repo = AsyncMock()
            CommandRepo.return_value = command_repo

            result = await service.claim(
                session_id=session_id, bearer_token=_TOKEN, agent_id="agent-a"
            )

        assert result == (204, None)
        command_repo.claim.assert_not_awaited()

    async def test_no_queued_work_returns_204(self) -> None:
        session_id = uuid4()
        service = _make_service()
        with patch(_SESSION_REPO) as SessionRepo, patch(_COMMAND_REPO) as CommandRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            session_repo.get_by_id.return_value = _gpu_session(session_id=session_id)
            command_repo = AsyncMock()
            CommandRepo.return_value = command_repo
            command_repo.claim.return_value = None

            result = await service.claim(
                session_id=session_id, bearer_token=_TOKEN, agent_id="agent-a"
            )

        assert result == (204, None)

    async def test_unique_claim_race_returns_204_after_transaction_rollback(self) -> None:
        session_id = uuid4()
        service = _make_service()
        with patch(_SESSION_REPO) as SessionRepo, patch(_COMMAND_REPO) as CommandRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            session_repo.get_by_id.return_value = _gpu_session(session_id=session_id)
            command_repo = AsyncMock()
            CommandRepo.return_value = command_repo
            command_repo.claim.side_effect = IntegrityError("UPDATE", {}, Exception("unique"))

            result = await service.claim(
                session_id=session_id, bearer_token=_TOKEN, agent_id="agent-a"
            )

        assert result == (204, None)

    async def test_successful_claim_returns_200_and_envelope(self) -> None:
        session_id = uuid4()
        command_id, operation_id = uuid4(), uuid4()
        service = _make_service()
        claimed = GpuSessionCommand(
            id=command_id,
            session_id=session_id,
            product_id="vex",
            operation_id=operation_id,
            kind="bundle_provision",
            payload={"bundle": "wan_2.2_i2v", "mode": "full", "verify": True},
            status=CommandStatus.claimed,
            agent_id="agent-a",
            claimed_at=datetime.now(UTC),
        )

        with patch(_SESSION_REPO) as SessionRepo, patch(_COMMAND_REPO) as CommandRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            session_repo.get_by_id.return_value = _gpu_session(session_id=session_id)
            command_repo = AsyncMock()
            CommandRepo.return_value = command_repo
            command_repo.claim.return_value = claimed

            status, envelope = await service.claim(
                session_id=session_id, bearer_token=_TOKEN, agent_id="agent-a"
            )

        assert status == 200
        assert envelope is not None
        assert envelope["command_id"] == str(command_id)
        assert envelope["operation_id"] == str(operation_id)
        assert envelope["kind"] == "bundle_provision"
        assert envelope["batch"] is None
        assert envelope["payload"]["bundle"] == "wan_2.2_i2v"

    async def test_successful_batch_claim_serializes_batch_fields(self) -> None:
        session_id = uuid4()
        service = _make_service()
        claimed = GpuSessionCommand(
            id=uuid4(),
            session_id=session_id,
            product_id="vex",
            operation_id=uuid4(),
            kind="bundle_provision",
            payload={"bundle": "a", "mode": "full", "verify": True},
            status=CommandStatus.claimed,
            agent_id="agent-a",
            batch_id="b1",
            batch_index=1,
            batch_total=3,
        )

        with patch(_SESSION_REPO) as SessionRepo, patch(_COMMAND_REPO) as CommandRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            session_repo.get_by_id.return_value = _gpu_session(session_id=session_id)
            command_repo = AsyncMock()
            CommandRepo.return_value = command_repo
            command_repo.claim.return_value = claimed

            _status, envelope = await service.claim(
                session_id=session_id, bearer_token=_TOKEN, agent_id="agent-a"
            )

        assert envelope is not None
        assert envelope["batch"] == {"batch_id": "b1", "index": 1, "total": 3}


class TestDeadlineFor:
    """D26: per-kind deadlines are read from Settings, one entry per kind."""

    def test_maps_each_kind_to_its_own_setting(self) -> None:
        service = _make_service()
        assert service._deadline_for("bundle_provision") == 3600
        assert service._deadline_for("bundle_removal") == 600
        assert service._deadline_for("comfyui_restart") == 300
