"""Integration tests for GpuSessionCommandRepository against a real PostgreSQL database.

Covers the P3 invariants that need real Postgres semantics: D9/D22/D23's one-in-flight
claim (single guarded statement, SKIP LOCKED, self-reclaim), D26's per-kind deadlines,
FIFO-within-batch ordering (created_at ties within one transaction), the terminal
guards on mark_terminal/expire_overdue, and D31's cancel_for_session cascade.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import update

from src.core.enums import CommandStatus, OperationKind
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.gpu_session_command import GpuSessionCommand
from src.db.repositories.gpu_session_command import GpuSessionCommandRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


GpuSessionFactory = Callable[..., Coroutine[Any, Any, GpuSession]]

_DEADLINES = {
    OperationKind.bundle_provision.value: 3600,
    OperationKind.bundle_removal.value: 600,
    OperationKind.comfyui_restart.value: 300,
}


def _deadline_for(kind: str) -> int:
    return _DEADLINES[kind]


def command_repo(db_session: AsyncSession) -> GpuSessionCommandRepository:
    return GpuSessionCommandRepository(db_session)


async def _make_command(
    db_session: AsyncSession,
    *,
    session: GpuSession,
    kind: OperationKind | str = OperationKind.bundle_provision,
    payload: dict[str, Any] | None = None,
    batch_id: str | None = None,
    batch_index: int | None = None,
    batch_total: int | None = None,
) -> GpuSessionCommand:
    repo = GpuSessionCommandRepository(db_session)
    return await repo.create(
        id=new_id(),
        session_id=session.id,
        product_id=session.product_id,
        operation_id=new_id(),
        kind=kind,
        payload=payload or {"bundle": "wan_2.2_i2v", "mode": "full", "verify": True},
        batch_id=batch_id,
        batch_index=batch_index,
        batch_total=batch_total,
    )


# ---------------------------------------------------------------------------
# create / get_by_operation
# ---------------------------------------------------------------------------


async def test_create_persists_all_fields(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    session = await make_gpu_session()
    command = await _make_command(db_session, session=session)

    assert command.status == CommandStatus.queued
    assert command.session_id == session.id
    assert command.kind == OperationKind.bundle_provision
    assert command.payload["bundle"] == "wan_2.2_i2v"
    assert command.claimed_at is None
    assert command.terminal_at is None


async def test_get_by_operation_round_trips(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    session = await make_gpu_session()
    command = await _make_command(db_session, session=session)

    repo = command_repo(db_session)
    found = await repo.get_by_operation(command.operation_id)
    assert found is not None
    assert found.id == command.id


async def test_get_by_operation_unknown_returns_none(db_session: AsyncSession) -> None:
    repo = command_repo(db_session)
    assert await repo.get_by_operation(uuid4()) is None


# ---------------------------------------------------------------------------
# claim — D9/D22/D23 (invariants 5, 6, 7, 8 partial, 15)
# ---------------------------------------------------------------------------


async def test_claim_returns_none_when_queue_is_empty(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    session = await make_gpu_session()
    repo = command_repo(db_session)
    result = await repo.claim(
        session.id, "agent-a", now=datetime.now(UTC), deadline_for=_deadline_for
    )
    assert result is None


async def test_claim_returns_and_marks_the_queued_command(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    session = await make_gpu_session()
    command = await _make_command(db_session, session=session)
    now = datetime.now(UTC)

    repo = command_repo(db_session)
    claimed = await repo.claim(session.id, "agent-a", now=now, deadline_for=_deadline_for)

    assert claimed is not None
    assert claimed.id == command.id
    assert claimed.status == CommandStatus.claimed
    assert claimed.agent_id == "agent-a"
    assert claimed.claimed_at is not None
    assert claimed.deadline_at is not None
    assert claimed.deadline_at > now


async def test_claim_by_different_agent_returns_none_while_one_in_flight(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    """Invariant #5: with two queued commands, a second claim by a different agent
    returns None (204) while the first is still claimed."""
    session = await make_gpu_session()
    await _make_command(db_session, session=session)
    await _make_command(db_session, session=session)
    repo = command_repo(db_session)
    now = datetime.now(UTC)

    first = await repo.claim(session.id, "agent-a", now=now, deadline_for=_deadline_for)
    second = await repo.claim(session.id, "agent-b", now=now, deadline_for=_deadline_for)

    assert first is not None
    assert second is None


async def test_claim_is_stable_for_the_same_agent(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    """Invariant #6: the same agent_id claiming twice gets the same command_id
    both times, and claimed_at does not move."""
    session = await make_gpu_session()
    await _make_command(db_session, session=session)
    repo = command_repo(db_session)
    now = datetime.now(UTC)

    first = await repo.claim(session.id, "agent-a", now=now, deadline_for=_deadline_for)
    assert first is not None

    later = now + timedelta(seconds=5)
    second = await repo.claim(session.id, "agent-a", now=later, deadline_for=_deadline_for)

    assert second is not None
    assert second.id == first.id
    assert second.claimed_at == first.claimed_at
    assert second.deadline_at == first.deadline_at


async def test_claim_fifo_within_batch_by_index(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    """Invariant #7: batch members are handed out in index order, even though
    they share one created_at (server default is transaction-scoped)."""
    session = await make_gpu_session()
    batch_id = str(uuid4())
    # Insert out of index order to prove ordering isn't insertion-order-derived.
    third = await _make_command(
        db_session, session=session, batch_id=batch_id, batch_index=2, batch_total=3
    )
    first = await _make_command(
        db_session, session=session, batch_id=batch_id, batch_index=0, batch_total=3
    )
    second = await _make_command(
        db_session, session=session, batch_id=batch_id, batch_index=1, batch_total=3
    )
    assert third.created_at == first.created_at == second.created_at  # same transaction

    repo = command_repo(db_session)
    now = datetime.now(UTC)
    claimed1 = await repo.claim(session.id, "agent-a", now=now, deadline_for=_deadline_for)
    assert claimed1 is not None
    assert claimed1.id == first.id

    await repo.mark_terminal(claimed1.id, status=CommandStatus.succeeded, at=now)
    claimed2 = await repo.claim(session.id, "agent-a", now=now, deadline_for=_deadline_for)
    assert claimed2 is not None
    assert claimed2.id == second.id

    await repo.mark_terminal(claimed2.id, status=CommandStatus.succeeded, at=now)
    claimed3 = await repo.claim(session.id, "agent-a", now=now, deadline_for=_deadline_for)
    assert claimed3 is not None
    assert claimed3.id == third.id


async def test_claim_per_kind_deadlines_differ(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    """Invariant #15: a provision, a removal, and a restart claimed at the same
    instant get three different deadline_at values."""
    now = datetime.now(UTC)
    repo = command_repo(db_session)
    results: dict[str, datetime] = {}
    for kind in (
        OperationKind.bundle_provision,
        OperationKind.bundle_removal,
        OperationKind.comfyui_restart,
    ):
        session = await make_gpu_session()
        await _make_command(db_session, session=session, kind=kind)
        claimed = await repo.claim(session.id, "agent-a", now=now, deadline_for=_deadline_for)
        assert claimed is not None
        assert claimed.deadline_at is not None
        results[kind.value] = claimed.deadline_at

    assert results[OperationKind.bundle_provision.value] == now + timedelta(seconds=3600)
    assert results[OperationKind.bundle_removal.value] == now + timedelta(seconds=600)
    assert results[OperationKind.comfyui_restart.value] == now + timedelta(seconds=300)
    assert len(set(results.values())) == 3


# ---------------------------------------------------------------------------
# mark_terminal (invariant #13)
# ---------------------------------------------------------------------------


async def test_mark_terminal_applies_once(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    session = await make_gpu_session()
    command = await _make_command(db_session, session=session)
    at = datetime.now(UTC)

    repo = command_repo(db_session)
    applied = await repo.mark_terminal(command.id, status=CommandStatus.succeeded, at=at)
    assert applied is True
    await db_session.refresh(command)
    assert command.status == CommandStatus.succeeded
    assert command.terminal_at == at


async def test_mark_terminal_is_guarded_against_a_second_call(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    """A second terminal event does not move terminal_at (invariant #13)."""
    session = await make_gpu_session()
    command = await _make_command(db_session, session=session)
    first_at = datetime.now(UTC)

    repo = command_repo(db_session)
    await repo.mark_terminal(command.id, status=CommandStatus.succeeded, at=first_at)

    second_at = first_at + timedelta(seconds=5)
    applied = await repo.mark_terminal(
        command.id, status=CommandStatus.failed, at=second_at, error="late"
    )

    assert applied is False
    await db_session.refresh(command)
    assert command.status == CommandStatus.succeeded
    assert command.terminal_at == first_at
    assert command.error is None


# ---------------------------------------------------------------------------
# expire_overdue (invariant #14)
# ---------------------------------------------------------------------------


async def test_expire_overdue_expires_claimed_past_deadline(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    session = await make_gpu_session()
    await _make_command(db_session, session=session)
    repo = command_repo(db_session)
    now = datetime.now(UTC)
    claimed = await repo.claim(session.id, "agent-a", now=now, deadline_for=_deadline_for)
    assert claimed is not None

    # Force the deadline into the past directly — avoids waiting real time.
    await db_session.execute(
        update(GpuSessionCommand)
        .where(GpuSessionCommand.id == claimed.id)
        .values(deadline_at=now - timedelta(seconds=1))
    )
    await db_session.flush()

    expired = await repo.expire_overdue(now)

    assert [c.id for c in expired] == [claimed.id]
    await db_session.refresh(claimed)
    assert claimed.status == CommandStatus.expired
    assert claimed.terminal_at == now


async def test_expire_overdue_leaves_commands_not_yet_due(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    session = await make_gpu_session()
    await _make_command(db_session, session=session)
    repo = command_repo(db_session)
    now = datetime.now(UTC)
    claimed = await repo.claim(session.id, "agent-a", now=now, deadline_for=_deadline_for)
    assert claimed is not None

    expired = await repo.expire_overdue(now)

    assert expired == []
    await db_session.refresh(claimed)
    assert claimed.status == CommandStatus.claimed


# ---------------------------------------------------------------------------
# cancel_for_session — D31 (invariant #16)
# ---------------------------------------------------------------------------


async def test_cancel_for_session_moves_queued_and_claimed_to_cancelled(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    session = await make_gpu_session()
    c1 = await _make_command(db_session, session=session)
    c2 = await _make_command(db_session, session=session)
    c3 = await _make_command(db_session, session=session)
    repo = command_repo(db_session)
    now = datetime.now(UTC)
    claimed = await repo.claim(session.id, "agent-a", now=now, deadline_for=_deadline_for)
    assert claimed is not None
    assert claimed.id == c1.id

    cancelled = await repo.cancel_for_session(session.id, at=now, reason="session stopped")

    assert {c.id for c in cancelled} == {c1.id, c2.id, c3.id}
    for command in (c1, c2, c3):
        await db_session.refresh(command)
        assert command.status == CommandStatus.cancelled
        assert command.terminal_at == now
        assert command.error == "session stopped"


async def test_cancel_for_session_leaves_other_sessions_untouched(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    session1 = await make_gpu_session()
    session2 = await make_gpu_session()
    await _make_command(db_session, session=session1)
    other = await _make_command(db_session, session=session2)
    repo = command_repo(db_session)

    cancelled = await repo.cancel_for_session(
        session1.id, at=datetime.now(UTC), reason="session stopped"
    )

    assert other.id not in {c.id for c in cancelled}
    await db_session.refresh(other)
    assert other.status == CommandStatus.queued


async def test_cancel_for_session_leaves_terminal_commands_untouched(
    db_session: AsyncSession, make_gpu_session: GpuSessionFactory
) -> None:
    session = await make_gpu_session()
    command = await _make_command(db_session, session=session)
    repo = command_repo(db_session)
    now = datetime.now(UTC)
    await repo.mark_terminal(command.id, status=CommandStatus.succeeded, at=now)

    cancelled = await repo.cancel_for_session(session.id, at=now, reason="session stopped")

    assert cancelled == []
    await db_session.refresh(command)
    assert command.status == CommandStatus.succeeded
