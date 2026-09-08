"""Integration tests for guarded operation-event updates."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.gpu_session import GpuSessionResponse
from src.core.enums import DeploymentStatus, GpuSessionStatus, OperationKind, OperationStatus
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.gpu_session_deployment import GpuSessionDeployment
from src.db.models.user import User
from src.db.repositories.gpu_session_deployment import GpuSessionDeploymentRepository
from src.db.repositories.gpu_session_operation import EventOutcome, GpuSessionOperationRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


async def _make_session_with_deployments(
    db_session: AsyncSession, *, model_types: tuple[str, ...]
) -> tuple[GpuSession, list[GpuSessionDeployment]]:
    """Persist one session and its deployments for operation-projection tests."""
    user = User(
        id=uuid4(),
        email=f"operation-projection-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.active,
        bundle_name="qwen_rapid_aio",
        model_type=model_types[0],
    )
    deployments = [
        GpuSessionDeployment(
            id=uuid4(),
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=model_type,
            bundle_name="qwen_rapid_aio",
            status=DeploymentStatus.deploying,
            is_primary=index == 0,
            pending_restart=True,
        )
        for index, model_type in enumerate(model_types)
    ]
    db_session.add(user)
    await db_session.flush()
    db_session.add(gpu_session)
    await db_session.flush()
    db_session.add_all(deployments)
    await db_session.flush()
    return gpu_session, deployments


async def test_latest_operation_queries_break_transaction_timestamp_ties_by_uuidv7(
    db_session: AsyncSession,
) -> None:
    """CURRENT_TIMESTAMP is transaction-start time, so created_at alone is not total."""
    user = User(
        id=uuid4(),
        email=f"operation-order-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="qwen_rapid_aio",
        model_type="aisha-image",
    )
    deployment = GpuSessionDeployment(
        id=uuid4(),
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type="aisha-image",
        bundle_name="qwen_rapid_aio",
        status=DeploymentStatus.deploying,
        is_primary=True,
    )
    db_session.add(user)
    await db_session.flush()
    db_session.add(gpu_session)
    await db_session.flush()
    db_session.add(deployment)
    await db_session.flush()

    repo = GpuSessionOperationRepository(db_session)
    first = await repo.create(
        id=new_id(),
        session_id=gpu_session.id,
        deployment_id=deployment.id,
        product_id="vex",
        kind=OperationKind.bundle_removal,
    )
    second = await repo.create(
        id=new_id(),
        session_id=gpu_session.id,
        deployment_id=deployment.id,
        product_id="vex",
        kind=OperationKind.bundle_removal,
    )

    assert first.created_at == second.created_at
    assert first.id < second.id
    assert (await repo.latest_by_deployment(gpu_session.id))[deployment.id].id == second.id
    latest_removal = await repo.latest_for_deployment_and_kind(
        deployment.id, OperationKind.bundle_removal
    )
    assert latest_removal is not None
    assert latest_removal.id == second.id


async def test_latest_by_deployment_projects_cohort_restart_for_every_member(
    db_session: AsyncSession,
) -> None:
    """A restart with no singular deployment target remains current for its cohort."""
    gpu_session, deployments = await _make_session_with_deployments(
        db_session, model_types=("aisha-image", "aisha-video")
    )
    operation_repo = GpuSessionOperationRepository(db_session)
    provision = await operation_repo.create(
        id=new_id(),
        session_id=gpu_session.id,
        deployment_id=deployments[0].id,
        product_id="vex",
        kind=OperationKind.bundle_provision,
    )
    provision.status = OperationStatus.succeeded
    provision.terminal_at = datetime.now(UTC)
    restart = await operation_repo.create(
        id=new_id(),
        session_id=gpu_session.id,
        product_id="vex",
        kind=OperationKind.comfyui_restart,
    )
    updated = await GpuSessionDeploymentRepository(db_session).set_restart_pointer(
        [deployment.id for deployment in deployments], operation_id=restart.id
    )

    latest = await operation_repo.latest_by_deployment(gpu_session.id)
    response = GpuSessionResponse.from_model(
        gpu_session, deployments=deployments, current_operations=latest
    )

    assert updated == 2
    assert restart.deployment_id is None
    assert latest[deployments[0].id].id == restart.id
    assert latest[deployments[1].id].id == restart.id
    assert latest[deployments[0].id] is latest[deployments[1].id]
    for deployment in response.deployments:
        current_operation = deployment.current_operation
        assert current_operation is not None
        assert current_operation.id == restart.id
        assert current_operation.deployment_id is None


async def test_latest_by_deployment_prefers_later_removal_over_stale_restart_pointer(
    db_session: AsyncSession,
) -> None:
    """A terminal restart pointer does not hide a later deployment-scoped removal."""
    gpu_session, [deployment] = await _make_session_with_deployments(
        db_session, model_types=("aisha-image",)
    )
    operation_repo = GpuSessionOperationRepository(db_session)
    restart = await operation_repo.create(
        id=new_id(),
        session_id=gpu_session.id,
        product_id="vex",
        kind=OperationKind.comfyui_restart,
    )
    updated = await GpuSessionDeploymentRepository(db_session).set_restart_pointer(
        [deployment.id], operation_id=restart.id
    )
    restart.status = OperationStatus.succeeded
    restart.terminal_at = datetime.now(UTC)
    await db_session.flush()
    removal = await operation_repo.create(
        id=new_id(),
        session_id=gpu_session.id,
        deployment_id=deployment.id,
        product_id="vex",
        kind=OperationKind.bundle_removal,
    )

    latest = await operation_repo.latest_by_deployment(gpu_session.id)

    assert updated == 1
    assert latest[deployment.id].id == removal.id


async def test_latest_by_deployment_excludes_cross_session_restart_pointer(
    db_session: AsyncSession,
) -> None:
    """A corrupted pointer cannot expose an operation owned by another session."""
    first_session, [first_deployment] = await _make_session_with_deployments(
        db_session, model_types=("aisha-image",)
    )
    second_session, _ = await _make_session_with_deployments(
        db_session, model_types=("aisha-video",)
    )
    restart = await GpuSessionOperationRepository(db_session).create(
        id=new_id(),
        session_id=second_session.id,
        product_id="vex",
        kind=OperationKind.comfyui_restart,
    )
    updated = await GpuSessionDeploymentRepository(db_session).set_restart_pointer(
        [first_deployment.id], operation_id=restart.id
    )

    latest = await GpuSessionOperationRepository(db_session).latest_by_deployment(first_session.id)

    assert updated == 1
    assert latest == {}


async def test_apply_event_is_monotonic_and_terminal_once(db_session: AsyncSession) -> None:
    """Late, duplicate, and post-terminal events cannot overwrite durable state."""
    user = User(
        id=uuid4(),
        email=f"telemetry-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="qwen_rapid_aio",
        model_type="aisha-image",
    )
    db_session.add(user)
    await db_session.flush()
    db_session.add(gpu_session)
    await db_session.flush()

    repo = GpuSessionOperationRepository(db_session)
    operation = await repo.create(
        id=uuid4(),
        session_id=gpu_session.id,
        product_id="vex",
        kind=OperationKind.session_bootstrap,
    )
    now = datetime.now(UTC)
    first = await repo.apply_event(
        operation_id=operation.id,
        session_id=gpu_session.id,
        sequence=0,
        event_id="bash-fallback-event",
        status=OperationStatus.running,
        phase="models",
        node_started_at=now,
        event_at=now,
        message="downloading",
        progress={"work": {"completed": 1, "total": 2, "unit": "files"}},
        plan={"phases": []},
        summary=None,
        error=None,
    )
    terminal = await repo.apply_event(
        operation_id=operation.id,
        session_id=gpu_session.id,
        sequence=1,
        event_id="terminal-event",
        status=OperationStatus.succeeded,
        phase=None,
        node_started_at=now,
        event_at=now,
        message="done",
        progress=None,
        plan=None,
        summary={"total_seconds": 1.0},
        error=None,
    )
    duplicate = await repo.apply_event(
        operation_id=operation.id,
        session_id=gpu_session.id,
        sequence=1,
        event_id="terminal-event",
        status=OperationStatus.succeeded,
        phase=None,
        node_started_at=now,
        event_at=now,
        message="changed",
        progress=None,
        plan=None,
        summary=None,
        error=None,
    )
    later = await repo.apply_event(
        operation_id=operation.id,
        session_id=gpu_session.id,
        sequence=2,
        event_id="second-terminal",
        status=OperationStatus.failed,
        phase=None,
        node_started_at=now,
        event_at=now,
        message="failed",
        progress=None,
        plan=None,
        summary=None,
        error="late failure",
    )
    after_terminal = await repo.apply_event(
        operation_id=operation.id,
        session_id=gpu_session.id,
        sequence=2,
        event_id="late-running-event",
        status=OperationStatus.running,
        phase="models",
        node_started_at=now,
        event_at=now,
        message="late progress",
        progress=None,
        plan=None,
        summary=None,
        error=None,
    )

    await db_session.refresh(operation)
    assert first.applied is True
    assert terminal.applied is True
    assert duplicate.reason == "duplicate"
    assert later.reason == "terminal_after_terminal"
    assert after_terminal.reason == "after_terminal"
    assert operation.last_sequence == 1
    assert operation.status == OperationStatus.succeeded
    assert operation.phase is None
    assert operation.terminal_at is not None
    assert operation.node_started_at == now
    assert operation.progress == {"work": {"completed": 1, "total": 2, "unit": "files"}}


async def test_apply_event_records_resolved_bundle_version_without_overwriting_pin(
    db_session: AsyncSession,
) -> None:
    """Nodes may resolve ``current`` but must not override Apex's pinned choice."""
    user = User(
        id=uuid4(),
        email=f"telemetry-target-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="qwen_rapid_aio",
        model_type="aisha-image",
    )
    db_session.add(user)
    await db_session.flush()
    db_session.add(gpu_session)
    await db_session.flush()

    repo = GpuSessionOperationRepository(db_session)
    resolved = await repo.create(
        id=uuid4(),
        session_id=gpu_session.id,
        product_id="vex",
        kind=OperationKind.session_bootstrap,
    )
    pinned = await repo.create(
        id=uuid4(),
        session_id=gpu_session.id,
        product_id="vex",
        kind=OperationKind.session_bootstrap,
        target_bundle_version="260101-01",
    )
    now = datetime.now(UTC)

    for operation, reported_version in (
        (resolved, "260105-01"),
        (pinned, "260106-01"),
    ):
        outcome = await repo.apply_event(
            operation_id=operation.id,
            session_id=gpu_session.id,
            sequence=0,
            event_id=f"target-{operation.id}",
            status=OperationStatus.running,
            phase="preflight",
            node_started_at=now,
            event_at=now,
            message="starting",
            progress=None,
            plan=None,
            summary=None,
            error=None,
            target_bundle_version=reported_version,
        )
        assert outcome.applied is True

    await db_session.refresh(resolved)
    await db_session.refresh(pinned)
    assert resolved.target_bundle_version == "260105-01"
    assert pinned.target_bundle_version == "260101-01"


async def test_concurrent_events_apply_once_without_losing_the_higher_sequence(
    db_engine: AsyncEngine,
) -> None:
    """Two real connections race safely through the guarded UPDATE.

    The lower sequence may win first, but the higher event must still become
    the durable latest state.  NullPool supplies independent connections.
    """
    user = User(
        id=uuid4(),
        email=f"telemetry-race-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="qwen_rapid_aio",
        model_type="aisha-image",
    )
    operation_id = uuid4()
    now = datetime.now(UTC)

    async with AsyncSession(bind=db_engine, expire_on_commit=False) as db:
        db.add(user)
        await db.flush()
        db.add(gpu_session)
        await db.flush()
        await GpuSessionOperationRepository(db).create(
            id=operation_id,
            session_id=gpu_session.id,
            product_id="vex",
            kind=OperationKind.session_bootstrap,
        )
        await db.commit()

    async def deliver(*, sequence: int, event_id: str) -> EventOutcome:
        async with (
            AsyncSession(bind=db_engine, expire_on_commit=False) as db,
            db.begin(),
        ):
            return await GpuSessionOperationRepository(db).apply_event(
                operation_id=operation_id,
                session_id=gpu_session.id,
                sequence=sequence,
                event_id=event_id,
                status=OperationStatus.succeeded if sequence == 1 else OperationStatus.running,
                phase=None if sequence == 1 else "models",
                node_started_at=now,
                event_at=now,
                message=event_id,
                progress=None,
                plan=None,
                summary={"winner": event_id} if sequence == 1 else None,
                error=None,
            )

    try:
        lower_outcome, terminal_outcome = await asyncio.gather(
            deliver(sequence=0, event_id="concurrent-start"),
            deliver(sequence=1, event_id="concurrent-terminal"),
        )
        # The terminal event cannot be lost to a concurrently delivered lower
        # sequence. The lower delivery either wins first or is correctly
        # rejected once the terminal state is durable.
        assert terminal_outcome.applied is True
        if lower_outcome.applied:
            assert lower_outcome.reason == "applied"
        else:
            assert lower_outcome.reason in {"stale", "after_terminal"}

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as db:
            operation = await GpuSessionOperationRepository(db).get(operation_id)
            assert operation is not None
            assert operation.last_sequence == 1
            assert operation.last_event_id == "concurrent-terminal"
            assert operation.status == OperationStatus.succeeded
            assert operation.terminal_at is not None
    finally:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as db:
            await db.execute(delete(User).where(User.id == user.id))
            await db.commit()
