"""Integration tests for P4: GpuSessionDeploymentService (attach/remove) and
DeploymentOrchestrationWorker, against a real PostgreSQL database.

GpuSessionCommandService and DeploymentOrchestrationWorker both commit internally
across independent sessions (session_factory-based, matching every other P3/P4
worker), so setup here uses a real (committing) session factory bound directly to
the test engine rather than the SAVEPOINT-scoped ``db_session`` fixture used
elsewhere — see test_content_retention.py's module docstring for the same
rationale. "No real node": a node agent's claim/telemetry round trip is not
simulated — command outcomes are applied directly via
GpuSessionCommandRepository.mark_terminal, the same primitive
OperationEventService.handle_event uses on a real terminal telemetry event.

Covers: D8 (batch of one), D33 (provision success != routable), D34 (one restart
per batch, fires even on partial failure), D35 (drain + timeout), D36 (duplicate
attach), D37 (removal blocked only by same-model in-flight jobs), D11 (retain_bundles
derivation), CO5 (already-removed tolerance), invariant #13 (removal failure ->
active), invariant #14 (OperationEventService never touches a deployment), and the
end-to-end attach -> provision succeeded -> pending_restart -> drain -> restart
succeeded -> routable flow required by the P4 definition of done.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.schemas.events import EventType
from src.api.schemas.gpu_session import DeploymentResponse, OperationEventBody
from src.api.services.gpu_session.command_payload import ProvisionCommand
from src.api.services.gpu_session.command_service import (
    CommandEnqueueSessionError,
    GpuSessionCommandService,
)
from src.api.services.gpu_session.deployment_orchestration_worker import (
    _ALREADY_REMOVED_ERROR_MARKER,
    DeploymentOrchestrationWorker,
    _is_already_removed_error,
)
from src.api.services.gpu_session.deployment_service import GpuSessionDeploymentService
from src.api.services.gpu_session.exceptions import (
    DeploymentAlreadyLiveError,
    DeploymentHasInFlightJobsError,
    DeploymentNotLiveError,
    InvalidSessionStateError,
    LastDeploymentRequiresForceError,
)
from src.api.services.gpu_session.operation_event_service import OperationEventService
from src.core.enums import (
    TERMINAL_COMMAND_STATUSES,
    CommandStatus,
    DeploymentStatus,
    GpuSessionStatus,
    ModelType,
    OperationKind,
    OperationStatus,
)
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.gpu_session_command import GpuSessionCommand
from src.db.models.gpu_session_deployment import GpuSessionDeployment
from src.db.models.storage import GenerationJob
from src.db.models.user import User
from src.db.repositories.gpu_session_command import GpuSessionCommandRepository
from src.db.repositories.gpu_session_deployment import (
    RESTART_COHORT_SANITY_LIMIT,
    GpuSessionDeploymentRepository,
)
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository
from src.db.repositories.job import JobRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


class _StubSettings:
    """Just the attributes GpuSessionCommandService/DeploymentOrchestrationWorker read."""

    def __init__(self, *, restart_drain_timeout_seconds: int = 900) -> None:
        self.gpu_command_provision_timeout_seconds = 3600
        self.gpu_command_removal_timeout_seconds = 600
        self.gpu_command_restart_timeout_seconds = 300
        self.gpu_deployment_orchestration_interval_seconds = 15
        self.gpu_deployment_restart_drain_timeout_seconds = restart_drain_timeout_seconds


class _FakeBundleIndex:
    """Duck-typed stand-in for BundleIndexService — attach() only reads a few
    attributes off the resolved mapping, never the type itself."""

    def resolve_bundle(self, model_type: str) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            bundle_name=f"bundle-{model_type}",
            bundle_version=None,
            readiness_marker=SimpleNamespace(node_class=f"NodeClassFor{model_type}"),
            declared_model_bytes=1_000_000,
        )


@pytest.fixture
def orchestration_session_factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_orchestration_rows(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
):
    """Real commits here (unlike the SAVEPOINT-scoped ``db_session`` fixture used
    elsewhere) persist for the rest of the session-scoped ``db_engine``. Several of
    this file's jobs are genuinely provider='aisha', QUEUED/RUNNING, on an 'active'
    session — exactly what JobRepository.list_aisha_jobs_for_polling's unscoped query
    matches — so leftovers would leak into unrelated tests later in the same pytest
    run. _create_user's fixed 'orch-' email prefix makes a blanket, cascade-driven
    cleanup (users -> gpu_sessions/generation_jobs/gpu_session_deployments) safe and
    simple, without tracking individual ids."""
    yield
    async with orchestration_session_factory() as session:
        await session.execute(delete(User).where(User.email.like("orch-%")))
        await session.commit()


def _command_service(
    session_factory: async_sessionmaker[AsyncSession], settings: _StubSettings
) -> GpuSessionCommandService:
    return GpuSessionCommandService(session_factory=session_factory, settings=settings)  # type: ignore[arg-type]


def _deployment_service(
    session_factory: async_sessionmaker[AsyncSession],
    command_service: GpuSessionCommandService,
) -> GpuSessionDeploymentService:
    return GpuSessionDeploymentService(
        session_factory=session_factory,
        bundle_index=_FakeBundleIndex(),  # type: ignore[arg-type]
        command_service=command_service,
    )


def _worker(
    session_factory: async_sessionmaker[AsyncSession],
    command_service: GpuSessionCommandService,
    settings: _StubSettings,
    *,
    event_bus: Any = None,
) -> DeploymentOrchestrationWorker:
    return DeploymentOrchestrationWorker(
        session_factory=session_factory,
        command_service=command_service,
        settings=settings,  # type: ignore[arg-type]
        event_bus=event_bus,
        redis_enabled=False,
        redis_client_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


async def _create_user(session_factory: async_sessionmaker[AsyncSession]) -> User:
    async with session_factory() as session:
        user = User(
            id=uuid4(),
            email=f"orch-{uuid4().hex[:10]}@example.com",
            password_hash="hashed",
            is_active=True,
            product_id="vex",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def _create_gpu_session(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user: User,
    status: str = GpuSessionStatus.active,
    model_type: str = ModelType.AISHA_IMAGE.value,
    bundle_name: str = "primary-bundle",
    callback_token_hash: str | None = None,
) -> GpuSession:
    async with session_factory() as session:
        gpu_session = GpuSession(
            id=new_id(),
            user_id=user.id,
            product_id="vex",
            bundle_name=bundle_name,
            model_type=model_type,
            status=status,
            callback_token_hash=callback_token_hash,
        )
        session.add(gpu_session)
        await session.commit()
        await session.refresh(gpu_session)
        return gpu_session


async def _create_deployment(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    gpu_session: GpuSession,
    status: str = DeploymentStatus.active,
    is_primary: bool = True,
    model_type: str | None = None,
    bundle_name: str | None = None,
    pending_restart: bool = False,
    batch_id: str | None = None,
    provision_operation_id: Any = None,
    readiness_marker_node_class: str | None = None,
) -> GpuSessionDeployment:
    async with session_factory() as session:
        deployment = await GpuSessionDeploymentRepository(session).create(
            id=new_id(),
            session_id=gpu_session.id,
            user_id=gpu_session.user_id,
            product_id=gpu_session.product_id,
            model_type=model_type or gpu_session.model_type,
            bundle_name=bundle_name if bundle_name is not None else gpu_session.bundle_name,
            status=status,
            is_primary=is_primary,
            pending_restart=pending_restart,
            provision_operation_id=provision_operation_id,
            readiness_marker_node_class=readiness_marker_node_class,
        )
        await session.commit()
        if batch_id is not None:
            await GpuSessionDeploymentRepository(session).set_provision_pointer(
                deployment.id, operation_id=provision_operation_id or new_id(), batch_id=batch_id
            )
            await session.commit()
        await session.refresh(deployment)
        return deployment


async def _create_raw_batch_command(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    gpu_session: GpuSession,
    deployment: GpuSessionDeployment,
    batch_id: str,
    batch_index: int,
    batch_total: int,
    kind: str = OperationKind.bundle_provision,
) -> GpuSessionCommand:
    """Manually construct one member of a multi-deployment batch — enqueue_batch's
    API stamps a single deployment_id across the whole batch, so a genuine
    multi-deployment batch (D34's generic case) has to be built by hand."""
    async with session_factory() as session, session.begin():
        operation_id = new_id()
        command_id = new_id()
        await GpuSessionOperationRepository(session).create(
            id=operation_id,
            session_id=gpu_session.id,
            product_id=gpu_session.product_id,
            kind=kind,
            batch_id=batch_id,
            batch_index=batch_index,
            batch_total=batch_total,
            command_id=command_id,
        )
        return await GpuSessionCommandRepository(session).create(
            id=command_id,
            session_id=gpu_session.id,
            product_id=gpu_session.product_id,
            operation_id=operation_id,
            deployment_id=deployment.id,
            kind=kind,
            payload={"bundle": deployment.bundle_name, "mode": "additive", "verify": True},
            batch_id=batch_id,
            batch_index=batch_index,
            batch_total=batch_total,
        )


async def _create_job(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user: User,
    gpu_session: GpuSession,
    model: str,
    status: str = "running",
) -> GenerationJob:
    async with session_factory() as session:
        job = GenerationJob(
            id=uuid4(),
            user_id=user.id,
            product_id="vex",
            provider="aisha",
            model=model,
            name="orchestration test job",
            status=status,
            generation_type="t2i",
            prompt="a test image",
            gpu_session_id=gpu_session.id,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job


async def _get_command_by_operation(
    session_factory: async_sessionmaker[AsyncSession], operation_id: Any
) -> GpuSessionCommand:
    async with session_factory() as session:
        command = await GpuSessionCommandRepository(session).get_by_operation(operation_id)
        assert command is not None
        return command


async def _mark_terminal(
    session_factory: async_sessionmaker[AsyncSession],
    command_id: Any,
    *,
    status: CommandStatus,
    error: str | None = None,
    at: datetime | None = None,
) -> None:
    async with session_factory() as session, session.begin():
        applied = await GpuSessionCommandRepository(session).mark_terminal(
            command_id, status=status, at=at or datetime.now(UTC), error=error
        )
        assert applied is True


async def _get_deployment(
    session_factory: async_sessionmaker[AsyncSession], deployment_id: Any
) -> GpuSessionDeployment:
    async with session_factory() as session:
        deployment = await session.get(GpuSessionDeployment, deployment_id)
        assert deployment is not None
        return deployment


# ---------------------------------------------------------------------------
# End-to-end: attach -> provision succeeded -> pending_restart -> drain ->
# restart succeeded -> routable
# ---------------------------------------------------------------------------


async def test_end_to_end_attach_to_routable(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    worker = _worker(orchestration_session_factory, command_service, settings)

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    await _create_deployment(orchestration_session_factory, gpu_session=gpu_session)

    # Not yet routable — no sibling deployment exists.
    async with orchestration_session_factory() as session:
        assert (
            await GpuSessionDeploymentRepository(session).get_routable(
                user.id, "vex", ModelType.AISHA_VIDEO.value
            )
            is None
        )

    # P1: attach — always additive, always a batch of one (D8, invariant #1/#2).
    deployment, operation_id = await deployment_service.attach(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
    )
    assert deployment.status == DeploymentStatus.deploying
    assert deployment.pending_restart is False
    assert deployment.is_primary is False
    assert deployment.provision_operation_id == operation_id
    assert deployment.batch_id is not None

    provision_command = await _get_command_by_operation(orchestration_session_factory, operation_id)
    assert provision_command.kind == OperationKind.bundle_provision
    assert provision_command.deployment_id == deployment.id
    assert provision_command.payload["mode"] == "additive"  # invariant #2
    assert provision_command.batch_id == deployment.batch_id
    assert provision_command.batch_total == 1  # invariant #1: batch of one

    # Simulate the node completing the provision command.
    await _mark_terminal(
        orchestration_session_factory, provision_command.id, status=CommandStatus.succeeded
    )

    # Not routable while merely 'deploying' with no restart yet (D33 / invariant #5).
    await worker.run_once()
    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.status == DeploymentStatus.deploying
    assert deployment.pending_restart is True
    assert deployment.restart_operation_id is not None  # step 2 fired in the same tick

    async with orchestration_session_factory() as session:
        assert (
            await GpuSessionDeploymentRepository(session).get_routable(
                user.id, "vex", ModelType.AISHA_VIDEO.value
            )
            is None
        )

    restart_command = await _get_command_by_operation(
        orchestration_session_factory, deployment.restart_operation_id
    )
    assert restart_command.kind == OperationKind.comfyui_restart

    await _mark_terminal(
        orchestration_session_factory, restart_command.id, status=CommandStatus.succeeded
    )
    await worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.status == DeploymentStatus.active
    assert deployment.pending_restart is False
    assert deployment.activated_at is not None

    async with orchestration_session_factory() as session:
        routable = await GpuSessionDeploymentRepository(session).get_routable(
            user.id, "vex", ModelType.AISHA_VIDEO.value
        )
    assert routable is not None
    routed_session, routed_deployment = routable
    assert routed_session.id == gpu_session.id
    assert routed_deployment.id == deployment.id


async def test_provision_failure_frees_the_slot(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    worker = _worker(orchestration_session_factory, command_service, settings)

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)

    deployment, operation_id = await deployment_service.attach(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
    )
    command = await _get_command_by_operation(orchestration_session_factory, operation_id)
    await _mark_terminal(
        orchestration_session_factory, command.id, status=CommandStatus.failed, error="disk full"
    )

    await worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.status == DeploymentStatus.failed
    assert deployment.removed_at is not None

    # The slot is free — attaching the same model again now succeeds.
    second, _ = await deployment_service.attach(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
    )
    assert second.id != deployment.id


# ---------------------------------------------------------------------------
# O1: attach compensation and no-command orphan reaper
# ---------------------------------------------------------------------------


async def test_attach_compensates_for_any_enqueue_failure(
    orchestration_session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-session enqueue failure must not strand the committed deployment row."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    monkeypatch.setattr(
        command_service,
        "enqueue_batch",
        AsyncMock(side_effect=ConnectionError("Redis temporarily unavailable")),
    )

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)

    with pytest.raises(ConnectionError, match="Redis temporarily unavailable"):
        await deployment_service.attach(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_VIDEO,
        )

    async with orchestration_session_factory() as session:
        deployment = await GpuSessionDeploymentRepository(session).get_live_for_model(
            user.id, "vex", ModelType.AISHA_VIDEO.value
        )
    assert deployment is None


async def test_orphan_reaper_fails_old_deployment_without_provision_command(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The second line of defense frees a row if attach's compensation could not run."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    deployment = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
    )

    async with orchestration_session_factory() as session:
        await session.execute(
            update(GpuSessionDeployment)
            .where(GpuSessionDeployment.id == deployment.id)
            .values(created_at=datetime.now(UTC) - timedelta(minutes=2))
        )
        await session.commit()

    await worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.status == DeploymentStatus.failed
    assert deployment.removed_at is not None
    async with orchestration_session_factory() as session:
        assert (
            await GpuSessionDeploymentRepository(session).get_live_for_model(
                user.id, "vex", ModelType.AISHA_VIDEO.value
            )
            is None
        )


async def test_orphan_reaper_recovers_pointer_instead_of_failing_when_command_exists(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """N3: attach's create() and enqueue_batch() commit in separate transactions —
    a process death between them can leave a real, queued provision command with
    the deployment's provision_operation_id still null. That must be repaired, not
    reaped: the node may already be downloading weights for it."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    deployment = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
    )
    batch_id = f"batch-{uuid4().hex}"
    command = await _create_raw_batch_command(
        orchestration_session_factory,
        gpu_session=gpu_session,
        deployment=deployment,
        batch_id=batch_id,
        batch_index=0,
        batch_total=1,
    )

    # Age the row past the reaper's grace window — recovery must still win even
    # though the reaper's own predicate would otherwise be ready to fire.
    async with orchestration_session_factory() as session:
        await session.execute(
            update(GpuSessionDeployment)
            .where(GpuSessionDeployment.id == deployment.id)
            .values(created_at=datetime.now(UTC) - timedelta(minutes=2))
        )
        await session.commit()

    await worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.status == DeploymentStatus.deploying
    assert deployment.removed_at is None
    assert deployment.provision_operation_id == command.operation_id
    assert deployment.batch_id == batch_id


# ---------------------------------------------------------------------------
# D36: duplicate attach
# ---------------------------------------------------------------------------


async def test_attach_rejects_duplicate_live_model(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    await deployment_service.attach(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
    )

    with pytest.raises(DeploymentAlreadyLiveError):
        await deployment_service.attach(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_VIDEO,
        )

    # Invariant #3: no extra deployment row, no extra command, from the rejected call.
    async with orchestration_session_factory() as session:
        deployments = await GpuSessionDeploymentRepository(session).list_for_session(gpu_session.id)
    assert len(deployments) == 1


async def test_attach_rejects_non_active_session(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(
        orchestration_session_factory, user=user, status=GpuSessionStatus.paused
    )

    with pytest.raises(InvalidSessionStateError):
        await deployment_service.attach(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_VIDEO,
        )

    async with orchestration_session_factory() as session:
        deployments = await GpuSessionDeploymentRepository(session).list_for_session(gpu_session.id)
    assert deployments == []


async def test_remove_rejects_paused_session_without_enqueuing_a_command(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A stopped node cannot consume a removal command; resume must come first."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(
        orchestration_session_factory, user=user, status=GpuSessionStatus.paused
    )
    target = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
    )

    with pytest.raises(InvalidSessionStateError, match="resume the session first"):
        await deployment_service.remove(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_VIDEO,
            force=True,
        )

    target = await _get_deployment(orchestration_session_factory, target.id)
    assert target.status == DeploymentStatus.active
    async with orchestration_session_factory() as session:
        commands = (
            (
                await session.execute(
                    select(GpuSessionCommand).where(
                        GpuSessionCommand.deployment_id == target.id,
                        GpuSessionCommand.kind == OperationKind.bundle_removal,
                    )
                )
            )
            .scalars()
            .all()
        )
    assert commands == []


# ---------------------------------------------------------------------------
# R1 / D37 / last-deployment guard / D11 retain_bundles
# ---------------------------------------------------------------------------


async def test_remove_marks_not_routable_before_the_in_flight_count(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count runs after active -> removing, never before that transition."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    await _create_deployment(orchestration_session_factory, gpu_session=gpu_session)
    target = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
    )

    observed_statuses: list[str] = []
    original_count = JobRepository.count_in_flight_for_session_and_model

    async def count_after_transition(
        self: JobRepository, gpu_session_id: Any, model_type: str
    ) -> int:
        deployment = await self._session.get(GpuSessionDeployment, target.id)
        assert deployment is not None
        observed_statuses.append(deployment.status)
        return await original_count(self, gpu_session_id, model_type)

    monkeypatch.setattr(
        JobRepository, "count_in_flight_for_session_and_model", count_after_transition
    )

    await deployment_service.remove(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
        force=False,
    )

    assert observed_statuses == [DeploymentStatus.removing]
    async with orchestration_session_factory() as session:
        assert (
            await GpuSessionDeploymentRepository(session).get_routable(
                user.id, "vex", ModelType.AISHA_VIDEO.value
            )
            is None
        )


async def test_remove_enqueue_failure_rolls_the_deployment_back_to_active(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The state transition and removal command are one transaction."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    await _create_deployment(orchestration_session_factory, gpu_session=gpu_session)
    target = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
    )
    monkeypatch.setattr(
        command_service,
        "enqueue",
        AsyncMock(side_effect=ConnectionError("queue unavailable")),
    )

    with pytest.raises(ConnectionError, match="queue unavailable"):
        await deployment_service.remove(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_VIDEO,
            force=False,
        )

    target = await _get_deployment(orchestration_session_factory, target.id)
    assert target.status == DeploymentStatus.active
    async with orchestration_session_factory() as session:
        command = await GpuSessionCommandRepository(session).get_latest_by_deployment_and_kind(
            target.id, OperationKind.bundle_removal
        )
    assert command is None


async def test_remove_converts_terminal_session_enqueue_race_to_invalid_state(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T3: a terminal-session race is a normal 409, not a raw command-service error."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    await _create_deployment(orchestration_session_factory, gpu_session=gpu_session)
    target = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
    )
    monkeypatch.setattr(
        command_service,
        "enqueue",
        AsyncMock(side_effect=CommandEnqueueSessionError("GPU session became terminal")),
    )

    with pytest.raises(InvalidSessionStateError, match="became unavailable") as exc_info:
        await deployment_service.remove(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_VIDEO,
            force=False,
        )

    assert exc_info.value.operation == "remove"
    target = await _get_deployment(orchestration_session_factory, target.id)
    assert target.status == DeploymentStatus.active


async def test_two_concurrent_non_forced_removals_leave_one_active(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The removal advisory lock and guarded UPDATE serialize the last-slot rule."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    await _create_deployment(orchestration_session_factory, gpu_session=gpu_session)
    await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
    )

    results = await asyncio.gather(
        deployment_service.remove(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_IMAGE,
            force=False,
        ),
        deployment_service.remove(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_VIDEO,
            force=False,
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(result, GpuSessionDeployment) for result in results) == 1
    assert sum(isinstance(result, LastDeploymentRequiresForceError) for result in results) == 1
    async with orchestration_session_factory() as session:
        live = await GpuSessionDeploymentRepository(session).list_live_for_session(gpu_session.id)
    assert sum(deployment.status == DeploymentStatus.active for deployment in live) == 1


async def test_remove_blocked_by_in_flight_job_of_same_model_only(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    await _create_deployment(orchestration_session_factory, gpu_session=gpu_session)
    await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
    )
    await _create_job(
        orchestration_session_factory,
        user=user,
        gpu_session=gpu_session,
        model=ModelType.AISHA_VIDEO.value,
        status="running",
    )

    with pytest.raises(DeploymentHasInFlightJobsError):
        await deployment_service.remove(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_VIDEO,
            force=False,
        )

    # A job on the OTHER (primary) model does not block removing this one.
    removed = await deployment_service.remove(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_IMAGE,
        force=True,  # last-deployment guard would otherwise trip once the sibling exists too
    )
    assert removed.status == DeploymentStatus.removing


async def test_removing_no_command_reaper_returns_old_row_to_active(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Backstop a legacy/partial removal that has no durable removal command."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    deployment = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
        status=DeploymentStatus.removing,
    )
    async with orchestration_session_factory() as session:
        await session.execute(
            update(GpuSessionDeployment)
            .where(GpuSessionDeployment.id == deployment.id)
            .values(created_at=datetime.now(UTC) - timedelta(minutes=2))
        )
        await session.commit()

    await worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.status == DeploymentStatus.active


async def test_remove_last_deployment_requires_force(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    await _create_deployment(orchestration_session_factory, gpu_session=gpu_session)

    with pytest.raises(LastDeploymentRequiresForceError):
        await deployment_service.remove(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_IMAGE,
            force=False,
        )

    removed = await deployment_service.remove(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_IMAGE,
        force=True,
    )
    assert removed.status == DeploymentStatus.removing

    # The session itself is untouched — it keeps running and billing (D37 pitfall).
    async with orchestration_session_factory() as session:
        refreshed_session = await session.get(GpuSession, gpu_session.id)
    assert refreshed_session is not None
    assert refreshed_session.status == GpuSessionStatus.active


async def test_remove_not_live_returns_409_equivalent(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)

    with pytest.raises(DeploymentNotLiveError):
        await deployment_service.remove(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_VIDEO,
            force=True,
        )


async def test_remove_derives_retain_bundles_from_other_live_deployments(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """D11: retain_bundles lists every OTHER live deployment's bare bundle name —
    never the one being removed, never a name:version spec."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(
        orchestration_session_factory, user=user, bundle_name="primary-bundle"
    )
    await _create_deployment(
        orchestration_session_factory, gpu_session=gpu_session, bundle_name="primary-bundle"
    )
    sibling = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
        bundle_name="video-bundle",
    )

    await deployment_service.remove(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
        force=False,
    )

    async with orchestration_session_factory() as session:
        removal_command = await GpuSessionCommandRepository(
            session
        ).get_latest_by_deployment_and_kind(sibling.id, OperationKind.bundle_removal)
    assert removal_command is not None
    assert removal_command.payload["bundle"] == "video-bundle"
    assert removal_command.payload["retain_bundles"] == ["primary-bundle"]


async def test_remove_refuses_when_a_sibling_bundle_name_is_unresolvable(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """D11: fail loud rather than ship a short retain list — an empty bundle_name
    on another live deployment must block the removal entirely, not silently omit
    it from retain_bundles (which would let aisha delete weights still in use)."""
    from src.api.services.gpu_session.exceptions import RetainBundlesUnresolvableError

    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(
        orchestration_session_factory, user=user, bundle_name="primary-bundle"
    )
    await _create_deployment(
        orchestration_session_factory, gpu_session=gpu_session, bundle_name="primary-bundle"
    )
    await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
        bundle_name="",  # data invariant violated — must be refused, not silently dropped
    )

    with pytest.raises(RetainBundlesUnresolvableError):
        await deployment_service.remove(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_IMAGE,
            force=True,
        )

    # No removal command was ever enqueued for the primary — refused before enqueue.
    async with orchestration_session_factory() as session:
        primary = await GpuSessionDeploymentRepository(session).get_live_for_session_and_model(
            gpu_session.id, ModelType.AISHA_IMAGE.value
        )
    assert primary is not None
    assert primary.status == DeploymentStatus.active


# ---------------------------------------------------------------------------
# S3: an invariant-violating no-batch_id row is failed, not skipped forever
# ---------------------------------------------------------------------------


async def test_pending_restart_missing_batch_id_is_failed_not_skipped(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """S3: a pending_restart row with no batch_id can never match the
    batch-readiness correlation (SQL NULL = NULL is unknown), so it would
    otherwise sit at the head of the bounded restart-enqueue page forever —
    logged and skipped every tick. Should be unreachable in practice (every
    writer of pending_restart also stamps batch_id); the fix makes the
    invariant enforceable: fail the row once, loudly, instead."""
    settings = _StubSettings()
    worker = _worker(
        orchestration_session_factory,
        _command_service(orchestration_session_factory, settings),
        settings,
    )
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    deployment = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
        pending_restart=True,
    )
    assert deployment.batch_id is None

    await worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.status == DeploymentStatus.failed
    assert deployment.pending_restart is False
    assert deployment.removed_at is not None

    async with orchestration_session_factory() as session:
        candidates = await GpuSessionDeploymentRepository(
            session
        ).list_pending_restart_awaiting_operation()
    assert deployment.id not in {candidate.id for candidate in candidates}


async def test_orphaned_routing_suspension_is_released_after_missing_batch_reaper(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """T2: a terminal outcome outside step 4 cannot strand session-wide routing."""
    settings = _StubSettings()
    worker = _worker(
        orchestration_session_factory,
        _command_service(orchestration_session_factory, settings),
        settings,
    )
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    primary = await _create_deployment(orchestration_session_factory, gpu_session=gpu_session)
    pending = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
        pending_restart=True,
    )
    async with orchestration_session_factory() as session:
        await session.execute(
            update(GpuSessionDeployment)
            .where(GpuSessionDeployment.id == primary.id)
            .values(routing_suspended=True)
        )
        await session.commit()

    await worker.run_once()

    pending = await _get_deployment(orchestration_session_factory, pending.id)
    primary = await _get_deployment(orchestration_session_factory, primary.id)
    assert pending.status == DeploymentStatus.failed
    assert primary.routing_suspended is False


async def test_implausible_restart_cohort_is_skipped_without_enqueuing(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T5: never silently truncate a restart cohort to make the worker progress."""
    from src.api.services.gpu_session import deployment_orchestration_worker

    settings = _StubSettings()
    command_service = AsyncMock()
    worker = _worker(orchestration_session_factory, command_service, settings)  # type: ignore[arg-type]
    session_id = uuid4()
    log = MagicMock()

    async def list_candidates(
        self: GpuSessionDeploymentRepository,
    ) -> list[MagicMock]:
        del self
        candidate = MagicMock()
        candidate.session_id = session_id
        return [candidate]

    async def list_members(
        self: GpuSessionDeploymentRepository, candidate_session_id: Any
    ) -> list[MagicMock]:
        del self
        assert candidate_session_id == session_id
        return [MagicMock()] * (RESTART_COHORT_SANITY_LIMIT + 1)

    monkeypatch.setattr(
        GpuSessionDeploymentRepository,
        "list_pending_restart_awaiting_operation",
        list_candidates,
    )
    monkeypatch.setattr(
        GpuSessionDeploymentRepository,
        "list_pending_restart_awaiting_operation_for_session",
        list_members,
    )
    monkeypatch.setattr(worker, "_reap_pending_restart_missing_batch_id", AsyncMock())
    monkeypatch.setattr(deployment_orchestration_worker, "logger", log)

    await worker._process_restart_enqueue()

    command_service.enqueue.assert_not_awaited()
    log.error.assert_called_once_with(
        "gpu_session.deployment.restart_cohort_implausible",
        session_id=str(session_id),
        count=RESTART_COHORT_SANITY_LIMIT + 1,
    )


# ---------------------------------------------------------------------------
# D34: one restart per batch, fires even when one member failed
# ---------------------------------------------------------------------------


async def test_restart_fires_once_per_batch_even_with_a_failed_member(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    worker = _worker(orchestration_session_factory, command_service, settings)

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    batch_id = f"batch-{uuid4().hex}"
    succeeding = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
        bundle_name="video-bundle",
        batch_id=batch_id,
    )
    failing = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type=ModelType.AISHA_IMAGE_LITE.value,
        bundle_name="lite-bundle",
        batch_id=batch_id,
    )
    succeeding_command = await _create_raw_batch_command(
        orchestration_session_factory,
        gpu_session=gpu_session,
        deployment=succeeding,
        batch_id=batch_id,
        batch_index=0,
        batch_total=2,
    )
    failing_command = await _create_raw_batch_command(
        orchestration_session_factory,
        gpu_session=gpu_session,
        deployment=failing,
        batch_id=batch_id,
        batch_index=1,
        batch_total=2,
    )
    await _mark_terminal(
        orchestration_session_factory, succeeding_command.id, status=CommandStatus.succeeded
    )
    await _mark_terminal(
        orchestration_session_factory,
        failing_command.id,
        status=CommandStatus.failed,
        error="out of disk",
    )

    await worker.run_once()

    succeeding = await _get_deployment(orchestration_session_factory, succeeding.id)
    failing = await _get_deployment(orchestration_session_factory, failing.id)
    assert succeeding.pending_restart is True
    assert succeeding.restart_operation_id is not None  # the restart still fired
    assert failing.status == DeploymentStatus.failed
    assert failing.restart_operation_id is None  # reported separately, no restart for it

    restart_command = await _get_command_by_operation(
        orchestration_session_factory, succeeding.restart_operation_id
    )
    assert restart_command.kind == OperationKind.comfyui_restart


# ---------------------------------------------------------------------------
# D35: restart waits for drain, then a bounded timeout forces it anyway
# ---------------------------------------------------------------------------


async def test_restart_waits_for_in_flight_jobs_to_drain(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    settings = _StubSettings(restart_drain_timeout_seconds=900)
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    worker = _worker(orchestration_session_factory, command_service, settings)

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    await _create_job(
        orchestration_session_factory,
        user=user,
        gpu_session=gpu_session,
        model=ModelType.AISHA_IMAGE.value,
        status="running",
    )

    deployment, operation_id = await deployment_service.attach(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
    )
    command = await _get_command_by_operation(orchestration_session_factory, operation_id)
    await _mark_terminal(orchestration_session_factory, command.id, status=CommandStatus.succeeded)

    await worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.pending_restart is True
    assert deployment.restart_operation_id is None  # blocked: a job is still in flight


async def test_restart_suspends_routing_before_the_drain_count_and_it_persists(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S5: routing must close before the in-flight count is read — the count is
    only a real bound on newly-admitted work if nothing new can be admitted
    while it's read — and the suspension must not be rolled back just because
    the session hasn't drained yet; it survives across ticks until the whole
    restart cycle resolves."""
    settings = _StubSettings(restart_drain_timeout_seconds=900)
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    worker = _worker(orchestration_session_factory, command_service, settings)

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    primary = await _create_deployment(orchestration_session_factory, gpu_session=gpu_session)
    await _create_job(
        orchestration_session_factory,
        user=user,
        gpu_session=gpu_session,
        model=ModelType.AISHA_IMAGE.value,
        status="running",
    )

    observed_suspended: list[bool] = []
    original_count = JobRepository.count_in_flight_for_session

    async def count_after_suspend(self: JobRepository, gpu_session_id: Any) -> int:
        current = await self._session.get(GpuSessionDeployment, primary.id)
        assert current is not None
        observed_suspended.append(current.routing_suspended)
        return await original_count(self, gpu_session_id)

    monkeypatch.setattr(JobRepository, "count_in_flight_for_session", count_after_suspend)

    deployment, operation_id = await deployment_service.attach(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
    )
    command = await _get_command_by_operation(orchestration_session_factory, operation_id)
    await _mark_terminal(orchestration_session_factory, command.id, status=CommandStatus.succeeded)

    await worker.run_once()

    assert observed_suspended == [True]  # suspended before the count was read
    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.restart_operation_id is None  # still draining
    async with orchestration_session_factory() as session:
        assert (
            await GpuSessionDeploymentRepository(session).get_routable(
                user.id, "vex", ModelType.AISHA_IMAGE.value
            )
            is None
        )

    # A second tick with the job still in flight must not lift the suspension.
    await worker.run_once()

    assert observed_suspended == [True, True]
    primary = await _get_deployment(orchestration_session_factory, primary.id)
    assert primary.routing_suspended is True


async def test_routing_suspension_is_projected_and_published(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """U1: API and SSE expose the temporary whole-session routing closure."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    event_bus = AsyncMock()
    deployment_service = GpuSessionDeploymentService(
        session_factory=orchestration_session_factory,
        bundle_index=_FakeBundleIndex(),  # type: ignore[arg-type]
        command_service=command_service,
        event_bus=event_bus,
    )
    worker = _worker(orchestration_session_factory, command_service, settings, event_bus=event_bus)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    primary = await _create_deployment(orchestration_session_factory, gpu_session=gpu_session)
    deployment, provision_operation_id = await deployment_service.attach(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
    )
    provision_command = await _get_command_by_operation(
        orchestration_session_factory, provision_operation_id
    )
    await _mark_terminal(
        orchestration_session_factory, provision_command.id, status=CommandStatus.succeeded
    )

    event_bus.reset_mock()
    await worker.run_once()

    primary = await _get_deployment(orchestration_session_factory, primary.id)
    assert DeploymentResponse.from_model(primary).routing_suspended is True
    suspended_events = [
        call.kwargs["payload"]
        for call in event_bus.publish.call_args_list
        if call.kwargs["event_type"] == EventType.GPU_DEPLOYMENT_STATUS_CHANGED
        and call.kwargs["payload"].deployment_id == primary.id
        and call.kwargs["payload"].routing_suspended is True
    ]
    assert suspended_events

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.restart_operation_id is not None
    restart_command = await _get_command_by_operation(
        orchestration_session_factory, deployment.restart_operation_id
    )
    await _mark_terminal(
        orchestration_session_factory, restart_command.id, status=CommandStatus.succeeded
    )
    event_bus.reset_mock()
    await worker.run_once()

    primary = await _get_deployment(orchestration_session_factory, primary.id)
    assert DeploymentResponse.from_model(primary).routing_suspended is False
    released_events = [
        call.kwargs["payload"]
        for call in event_bus.publish.call_args_list
        if call.kwargs["event_type"] == EventType.GPU_DEPLOYMENT_STATUS_CHANGED
        and call.kwargs["payload"].deployment_id == primary.id
        and call.kwargs["payload"].routing_suspended is False
    ]
    assert released_events


async def test_restart_enqueue_and_pointer_stamp_roll_back_together(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V2: a pointer failure cannot commit an orphaned restart command."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    deployment = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
        pending_restart=True,
        batch_id=f"batch-{uuid4().hex}",
        readiness_marker_node_class="VideoMarker",
    )
    original_set_restart_pointer = GpuSessionDeploymentRepository.set_restart_pointer

    async def fail_pointer_stamp(
        self: GpuSessionDeploymentRepository, deployment_ids: Any, *, operation_id: Any
    ) -> int:
        _ = self, deployment_ids, operation_id
        raise RuntimeError("pointer write failed")

    monkeypatch.setattr(GpuSessionDeploymentRepository, "set_restart_pointer", fail_pointer_stamp)
    with pytest.raises(RuntimeError, match="pointer write failed"):
        await worker._enqueue_restart_cohort(
            session_id=gpu_session.id,
            members=[deployment],
            in_flight=0,
            readiness_marker_node_class="VideoMarker",
        )

    async with orchestration_session_factory() as session:
        commands = (
            (
                await session.execute(
                    select(GpuSessionCommand).where(
                        GpuSessionCommand.session_id == gpu_session.id,
                        GpuSessionCommand.kind == OperationKind.comfyui_restart,
                    )
                )
            )
            .scalars()
            .all()
        )
    assert commands == []

    monkeypatch.setattr(
        GpuSessionDeploymentRepository, "set_restart_pointer", original_set_restart_pointer
    )
    await worker._enqueue_restart_cohort(
        session_id=gpu_session.id,
        members=[deployment],
        in_flight=0,
        readiness_marker_node_class="VideoMarker",
    )

    async with orchestration_session_factory() as session:
        commands = (
            (
                await session.execute(
                    select(GpuSessionCommand).where(
                        GpuSessionCommand.session_id == gpu_session.id,
                        GpuSessionCommand.kind == OperationKind.comfyui_restart,
                        GpuSessionCommand.status.notin_(tuple(TERMINAL_COMMAND_STATUSES)),
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(commands) == 1


async def test_restart_fires_anyway_after_drain_timeout(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """D35: gpu_deployment_restart_drain_timeout_seconds=0 means "timed out" the
    instant the batch completes, regardless of in-flight jobs."""
    provisioning_settings = _StubSettings(restart_drain_timeout_seconds=900)
    command_service = _command_service(orchestration_session_factory, provisioning_settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    await _create_job(
        orchestration_session_factory,
        user=user,
        gpu_session=gpu_session,
        model=ModelType.AISHA_IMAGE.value,
        status="queued",
    )

    deployment, operation_id = await deployment_service.attach(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
    )
    command = await _get_command_by_operation(orchestration_session_factory, operation_id)
    await _mark_terminal(orchestration_session_factory, command.id, status=CommandStatus.succeeded)

    timed_out_settings = _StubSettings(restart_drain_timeout_seconds=0)
    timed_out_worker = _worker(orchestration_session_factory, command_service, timed_out_settings)
    await timed_out_worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.pending_restart is True
    assert deployment.restart_operation_id is not None  # forced through despite the in-flight job


async def test_node_clock_behind_does_not_shorten_restart_drain_window(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A node terminal timestamp older than the timeout is not a server timeout."""
    settings = _StubSettings(restart_drain_timeout_seconds=900)
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    await _create_job(
        orchestration_session_factory,
        user=user,
        gpu_session=gpu_session,
        model=ModelType.AISHA_IMAGE.value,
    )

    deployment, operation_id = await deployment_service.attach(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
    )
    command = await _get_command_by_operation(orchestration_session_factory, operation_id)
    await _mark_terminal(
        orchestration_session_factory,
        command.id,
        status=CommandStatus.succeeded,
        at=datetime.now(UTC) - timedelta(seconds=901),
    )

    await worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.pending_restart is True
    assert deployment.pending_restart_since is not None
    assert deployment.restart_operation_id is None


async def test_node_clock_ahead_does_not_extend_restart_drain_window(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A future node terminal timestamp cannot prevent an already-due restart."""
    settings = _StubSettings(restart_drain_timeout_seconds=0)
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    await _create_job(
        orchestration_session_factory,
        user=user,
        gpu_session=gpu_session,
        model=ModelType.AISHA_IMAGE.value,
    )

    deployment, operation_id = await deployment_service.attach(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
    )
    command = await _get_command_by_operation(orchestration_session_factory, operation_id)
    await _mark_terminal(
        orchestration_session_factory,
        command.id,
        status=CommandStatus.succeeded,
        at=datetime.now(UTC) + timedelta(seconds=901),
    )

    await worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.pending_restart is True
    assert deployment.restart_operation_id is not None


# ---------------------------------------------------------------------------
# R3: one restart per readiness marker, not merely per session
# ---------------------------------------------------------------------------


async def test_different_readiness_markers_get_independent_restart_commands(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Each marker cohort gets its own restart command with the correct
    node_class payload — but per S6, the OUTCOME is shared across the whole
    session's restart cycle, not resolved independently per marker; see
    test_restart_cycle_fails_every_cohort_when_one_restart_fails for that."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)

    video, lite = await asyncio.gather(
        deployment_service.attach(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_VIDEO,
        ),
        deployment_service.attach(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_IMAGE_LITE,
        ),
    )
    video_deployment, video_operation_id = video
    lite_deployment, lite_operation_id = lite
    video_command = await _get_command_by_operation(
        orchestration_session_factory, video_operation_id
    )
    lite_command = await _get_command_by_operation(orchestration_session_factory, lite_operation_id)
    await _mark_terminal(
        orchestration_session_factory, video_command.id, status=CommandStatus.succeeded
    )
    await _mark_terminal(
        orchestration_session_factory, lite_command.id, status=CommandStatus.succeeded
    )

    await worker.run_once()

    video_deployment = await _get_deployment(orchestration_session_factory, video_deployment.id)
    lite_deployment = await _get_deployment(orchestration_session_factory, lite_deployment.id)
    assert video_deployment.restart_operation_id is not None
    assert lite_deployment.restart_operation_id is not None
    assert video_deployment.restart_operation_id != lite_deployment.restart_operation_id

    video_restart_command = await _get_command_by_operation(
        orchestration_session_factory, video_deployment.restart_operation_id
    )
    lite_restart_command = await _get_command_by_operation(
        orchestration_session_factory, lite_deployment.restart_operation_id
    )
    assert video_restart_command.payload["node_class"] == "NodeClassForaisha-video"
    assert lite_restart_command.payload["node_class"] == "NodeClassForaisha-image-lite"
    await _mark_terminal(
        orchestration_session_factory, video_restart_command.id, status=CommandStatus.succeeded
    )
    await _mark_terminal(
        orchestration_session_factory, lite_restart_command.id, status=CommandStatus.succeeded
    )
    await worker.run_once()

    video_deployment = await _get_deployment(orchestration_session_factory, video_deployment.id)
    lite_deployment = await _get_deployment(orchestration_session_factory, lite_deployment.id)
    assert video_deployment.status == DeploymentStatus.active
    assert lite_deployment.status == DeploymentStatus.active


# ---------------------------------------------------------------------------
# N1: a completed cohort must not activate while a sibling restart is pending
# ---------------------------------------------------------------------------


async def _attach_two_marker_cohorts_and_enqueue_restarts(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
    worker: DeploymentOrchestrationWorker,
    deployment_service: GpuSessionDeploymentService,
    gpu_session: GpuSession,
    user: User,
) -> tuple[GpuSessionDeployment, GpuSessionDeployment]:
    """Shared N1 setup: two independent readiness-marker cohorts on one session,
    both provisioned and both with a restart already enqueued (but not yet
    terminal) — the point at which the two cohorts' restart outcomes can race."""
    video, lite = await asyncio.gather(
        deployment_service.attach(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_VIDEO,
        ),
        deployment_service.attach(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type=ModelType.AISHA_IMAGE_LITE,
        ),
    )
    video_deployment, video_operation_id = video
    lite_deployment, lite_operation_id = lite
    video_command = await _get_command_by_operation(
        orchestration_session_factory, video_operation_id
    )
    lite_command = await _get_command_by_operation(orchestration_session_factory, lite_operation_id)
    await _mark_terminal(
        orchestration_session_factory, video_command.id, status=CommandStatus.succeeded
    )
    await _mark_terminal(
        orchestration_session_factory, lite_command.id, status=CommandStatus.succeeded
    )

    await worker.run_once()  # enqueues one restart command per marker cohort

    video_deployment = await _get_deployment(orchestration_session_factory, video_deployment.id)
    lite_deployment = await _get_deployment(orchestration_session_factory, lite_deployment.id)
    assert video_deployment.restart_operation_id is not None
    assert lite_deployment.restart_operation_id is not None
    return video_deployment, lite_deployment


async def test_restart_success_defers_activation_while_a_sibling_restart_is_pending(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """N1: R3's per-marker cohorts made this possible — one cohort's restart can
    finish while a sibling cohort's restart for the same session is still queued.
    Activating the finished cohort immediately would make it briefly routable
    right before the sibling restart takes ComfyUI down again."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)

    video_deployment, lite_deployment = await _attach_two_marker_cohorts_and_enqueue_restarts(
        orchestration_session_factory, worker, deployment_service, gpu_session, user
    )
    async with orchestration_session_factory() as session:
        assert (
            await GpuSessionDeploymentRepository(session).get_routable(
                user.id, "vex", ModelType.AISHA_IMAGE.value
            )
            is None
        )
    video_restart_command = await _get_command_by_operation(
        orchestration_session_factory, video_deployment.restart_operation_id
    )
    # Only the video cohort's restart finishes this tick — lite's stays queued.
    await _mark_terminal(
        orchestration_session_factory, video_restart_command.id, status=CommandStatus.succeeded
    )

    await worker.run_once()

    video_deployment = await _get_deployment(orchestration_session_factory, video_deployment.id)
    assert video_deployment.status == DeploymentStatus.deploying  # deferred, not activated
    assert video_deployment.pending_restart is True
    async with orchestration_session_factory() as session:
        routable = await GpuSessionDeploymentRepository(session).get_routable(
            user.id, "vex", ModelType.AISHA_VIDEO.value
        )
    assert routable is None  # and therefore not routable either

    # Once the sibling restart also goes terminal, the deferred cohort clears on
    # the very next tick.
    lite_restart_command = await _get_command_by_operation(
        orchestration_session_factory, lite_deployment.restart_operation_id
    )
    await _mark_terminal(
        orchestration_session_factory, lite_restart_command.id, status=CommandStatus.succeeded
    )

    await worker.run_once()

    video_deployment = await _get_deployment(orchestration_session_factory, video_deployment.id)
    lite_deployment = await _get_deployment(orchestration_session_factory, lite_deployment.id)
    assert video_deployment.status == DeploymentStatus.active
    assert lite_deployment.status == DeploymentStatus.active


async def test_restart_failure_fails_its_cohort_immediately_despite_pending_sibling(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """S6: a failed restart is never going to become routable, and D34 rules out
    a retry, so there is nothing to gain by waiting on a still-pending sibling's
    own outcome before failing — the failure cascades to the whole cycle
    immediately, including a sibling whose own restart command hasn't even
    resolved yet."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    primary = await _create_deployment(orchestration_session_factory, gpu_session=gpu_session)

    video_deployment, lite_deployment = await _attach_two_marker_cohorts_and_enqueue_restarts(
        orchestration_session_factory, worker, deployment_service, gpu_session, user
    )
    video_restart_command = await _get_command_by_operation(
        orchestration_session_factory, video_deployment.restart_operation_id
    )
    lite_restart_command = await _get_command_by_operation(
        orchestration_session_factory, lite_deployment.restart_operation_id
    )
    # Only the video cohort's restart finishes this tick — with a failure — while
    # lite's restart command stays queued.
    await _mark_terminal(
        orchestration_session_factory,
        video_restart_command.id,
        status=CommandStatus.failed,
        error="node agent crashed",
    )

    await worker.run_once()

    video_deployment = await _get_deployment(orchestration_session_factory, video_deployment.id)
    lite_deployment = await _get_deployment(orchestration_session_factory, lite_deployment.id)
    assert video_deployment.status == DeploymentStatus.failed  # not deferred
    assert video_deployment.pending_restart is False
    # S6: the still-queued sibling is failed too — Apex no longer knows which
    # node classes ComfyUI has loaded, so nothing from this cycle is routable.
    assert lite_deployment.status == DeploymentStatus.failed
    assert lite_deployment.pending_restart is False
    lite_restart_command = await _get_command_by_operation(
        orchestration_session_factory, lite_deployment.restart_operation_id
    )
    assert lite_restart_command.status == CommandStatus.cancelled

    # T1: cancelling the queued sibling means no restart remains that could
    # take ComfyUI down, so the session-wide primary suspension clears in the
    # same failed-cycle transaction.
    primary = await _get_deployment(orchestration_session_factory, primary.id)
    assert primary.routing_suspended is False
    async with orchestration_session_factory() as session:
        routable = await GpuSessionDeploymentRepository(session).get_routable(
            user.id, "vex", ModelType.AISHA_IMAGE.value
        )
    assert routable is not None


async def test_failed_restart_cycle_keeps_routing_suspended_for_claimed_sibling(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """T1/T2: a claimed sibling can still restart the node, so it owns the suspension.

    Once it becomes terminal, the reconciler releases the orphaned suspension;
    it must not do so while the command is non-terminal.
    """
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    primary = await _create_deployment(orchestration_session_factory, gpu_session=gpu_session)

    video_deployment, lite_deployment = await _attach_two_marker_cohorts_and_enqueue_restarts(
        orchestration_session_factory, worker, deployment_service, gpu_session, user
    )
    async with orchestration_session_factory() as session:
        assert (
            await GpuSessionDeploymentRepository(session).get_routable(
                user.id, "vex", ModelType.AISHA_IMAGE.value
            )
            is None
        )
    video_restart_command = await _get_command_by_operation(
        orchestration_session_factory, video_deployment.restart_operation_id
    )
    lite_restart_command = await _get_command_by_operation(
        orchestration_session_factory, lite_deployment.restart_operation_id
    )
    async with orchestration_session_factory() as session:
        await session.execute(
            update(GpuSessionCommand)
            .where(GpuSessionCommand.id == lite_restart_command.id)
            .values(status=CommandStatus.claimed)
        )
        await session.commit()
    await _mark_terminal(
        orchestration_session_factory,
        video_restart_command.id,
        status=CommandStatus.failed,
        error="node agent crashed",
    )

    await worker.run_once()

    primary = await _get_deployment(orchestration_session_factory, primary.id)
    lite_restart_command = await _get_command_by_operation(
        orchestration_session_factory, lite_deployment.restart_operation_id
    )
    assert lite_restart_command.status == CommandStatus.claimed
    assert primary.routing_suspended is True
    async with orchestration_session_factory() as session:
        assert (
            await GpuSessionDeploymentRepository(session).get_routable(
                user.id, "vex", ModelType.AISHA_IMAGE.value
            )
            is None
        )

    await _mark_terminal(
        orchestration_session_factory, lite_restart_command.id, status=CommandStatus.succeeded
    )
    await worker.run_once()

    primary = await _get_deployment(orchestration_session_factory, primary.id)
    assert primary.routing_suspended is False
    async with orchestration_session_factory() as session:
        assert (
            await GpuSessionDeploymentRepository(session).get_routable(
                user.id, "vex", ModelType.AISHA_IMAGE.value
            )
            is not None
        )


async def test_restart_cycle_fails_every_cohort_when_a_sibling_restart_fails(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """S6: the earlier cohort's own restart succeeding does not license routing
    when a later restart of the same node did not — mere terminality of a
    sibling command is the wrong bar (that's what let this cohort activate
    incorrectly before S6); the sibling must also have succeeded."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)

    video_deployment, lite_deployment = await _attach_two_marker_cohorts_and_enqueue_restarts(
        orchestration_session_factory, worker, deployment_service, gpu_session, user
    )
    video_restart_command = await _get_command_by_operation(
        orchestration_session_factory, video_deployment.restart_operation_id
    )
    # The first cohort's restart succeeds and defers (lite's is still pending).
    await _mark_terminal(
        orchestration_session_factory, video_restart_command.id, status=CommandStatus.succeeded
    )
    await worker.run_once()
    video_deployment = await _get_deployment(orchestration_session_factory, video_deployment.id)
    assert video_deployment.status == DeploymentStatus.deploying  # deferred

    # The second cohort's restart later fails.
    lite_restart_command = await _get_command_by_operation(
        orchestration_session_factory, lite_deployment.restart_operation_id
    )
    await _mark_terminal(
        orchestration_session_factory,
        lite_restart_command.id,
        status=CommandStatus.failed,
        error="marker did not register",
    )
    await worker.run_once()

    video_deployment = await _get_deployment(orchestration_session_factory, video_deployment.id)
    lite_deployment = await _get_deployment(orchestration_session_factory, lite_deployment.id)
    assert video_deployment.status == DeploymentStatus.failed  # never activated
    assert lite_deployment.status == DeploymentStatus.failed
    async with orchestration_session_factory() as session:
        assert (
            await GpuSessionDeploymentRepository(session).get_routable(
                user.id, "vex", ModelType.AISHA_VIDEO.value
            )
            is None
        )


async def test_restart_cycle_fails_every_cohort_when_a_sibling_restart_expires(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Companion to the failed-sibling case: an operation the P3 sweep expires
    is just as terminal-but-not-succeeded as an explicit failure, and the
    terminality gate this round fixes got that case wrong too."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)

    video_deployment, lite_deployment = await _attach_two_marker_cohorts_and_enqueue_restarts(
        orchestration_session_factory, worker, deployment_service, gpu_session, user
    )
    video_restart_command = await _get_command_by_operation(
        orchestration_session_factory, video_deployment.restart_operation_id
    )
    await _mark_terminal(
        orchestration_session_factory, video_restart_command.id, status=CommandStatus.succeeded
    )
    await worker.run_once()
    video_deployment = await _get_deployment(orchestration_session_factory, video_deployment.id)
    assert video_deployment.status == DeploymentStatus.deploying  # deferred

    lite_restart_command = await _get_command_by_operation(
        orchestration_session_factory, lite_deployment.restart_operation_id
    )
    await _mark_terminal(
        orchestration_session_factory,
        lite_restart_command.id,
        status=CommandStatus.expired,
        error="deadline exceeded",
    )
    await worker.run_once()

    video_deployment = await _get_deployment(orchestration_session_factory, video_deployment.id)
    lite_deployment = await _get_deployment(orchestration_session_factory, lite_deployment.id)
    assert video_deployment.status == DeploymentStatus.failed
    assert lite_deployment.status == DeploymentStatus.failed


# ---------------------------------------------------------------------------
# N4: deployment SSE events carry an operation id
# ---------------------------------------------------------------------------


async def test_restart_enqueued_event_carries_operation_id(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """N4: the client needs operation_id on the restart-enqueued event to poll the
    operation it's waiting on, instead of sitting on 'pending restart' until it
    independently refetches the session."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    deployment_service = _deployment_service(orchestration_session_factory, command_service)
    event_bus = AsyncMock()
    worker = _worker(orchestration_session_factory, command_service, settings, event_bus=event_bus)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)

    deployment, operation_id = await deployment_service.attach(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
    )
    command = await _get_command_by_operation(orchestration_session_factory, operation_id)
    await _mark_terminal(orchestration_session_factory, command.id, status=CommandStatus.succeeded)

    event_bus.reset_mock()
    await worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.restart_operation_id is not None

    published_operation_ids = {
        call.kwargs["payload"].operation_id for call in event_bus.publish.call_args_list
    }
    assert deployment.restart_operation_id in published_operation_ids


# ---------------------------------------------------------------------------
# S1: a removal event must not report a stale restart_operation_id
# ---------------------------------------------------------------------------


async def test_remove_event_reports_the_removal_operation_not_a_stale_restart(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """S1: resolve_restart_outcome never clears restart_operation_id on
    activation (only the D15 terminal cascade does), so a deployment that was
    ever restarted carries a stale restart_operation_id for the rest of its
    active life. remove()'s event must name its own removal operation
    explicitly instead of falling back to that stale id — otherwise a client
    that follows it sees an operation which never changes and concludes the
    removal isn't progressing, the exact "stuck UI" failure N4 was introduced
    to eliminate, relocated to the removal path. Fails against the pre-S1
    code, which publishes remove()'s event with no operation/operation_id at
    all and so falls back to the stale restart id."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    event_bus = AsyncMock()
    deployment_service = GpuSessionDeploymentService(
        session_factory=orchestration_session_factory,
        bundle_index=_FakeBundleIndex(),  # type: ignore[arg-type]
        command_service=command_service,
        event_bus=event_bus,
    )
    worker = _worker(orchestration_session_factory, command_service, settings, event_bus=event_bus)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    await _create_deployment(orchestration_session_factory, gpu_session=gpu_session)  # primary

    deployment, operation_id = await deployment_service.attach(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
    )
    command = await _get_command_by_operation(orchestration_session_factory, operation_id)
    await _mark_terminal(orchestration_session_factory, command.id, status=CommandStatus.succeeded)
    await worker.run_once()  # pending_restart -> restart enqueued

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    restart_command = await _get_command_by_operation(
        orchestration_session_factory, deployment.restart_operation_id
    )
    await _mark_terminal(
        orchestration_session_factory, restart_command.id, status=CommandStatus.succeeded
    )
    await worker.run_once()  # restart succeeded -> active

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.status == DeploymentStatus.active
    stale_restart_operation_id = deployment.restart_operation_id
    assert stale_restart_operation_id is not None  # never cleared on activation — the bug's cause

    event_bus.reset_mock()
    removed = await deployment_service.remove(
        session_id=gpu_session.id,
        user_id=user.id,
        product_id="vex",
        model_type=ModelType.AISHA_VIDEO,
        force=False,
    )

    async with orchestration_session_factory() as session:
        removal_command = await GpuSessionCommandRepository(
            session
        ).get_latest_by_deployment_and_kind(removed.id, OperationKind.bundle_removal)
    assert removal_command is not None

    published = event_bus.publish.call_args_list[-1]
    assert published.kwargs["payload"].operation_id == removal_command.operation_id
    assert published.kwargs["payload"].operation_id != stale_restart_operation_id


async def test_oldest_ready_deployment_sets_the_restart_drain_deadline(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A newly ready attachment must not extend an older cohort member's wait."""
    settings = _StubSettings(restart_drain_timeout_seconds=10)
    command_service = _command_service(orchestration_session_factory, settings)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    first_batch_id = f"batch-{uuid4().hex}"
    second_batch_id = f"batch-{uuid4().hex}"
    first = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
        pending_restart=True,
        batch_id=first_batch_id,
    )
    second = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type=ModelType.AISHA_IMAGE_LITE.value,
        pending_restart=True,
        batch_id=second_batch_id,
    )
    first_command = await _create_raw_batch_command(
        orchestration_session_factory,
        gpu_session=gpu_session,
        deployment=first,
        batch_id=first_batch_id,
        batch_index=0,
        batch_total=1,
    )
    second_command = await _create_raw_batch_command(
        orchestration_session_factory,
        gpu_session=gpu_session,
        deployment=second,
        batch_id=second_batch_id,
        batch_index=0,
        batch_total=1,
    )
    await _mark_terminal(
        orchestration_session_factory, first_command.id, status=CommandStatus.succeeded
    )
    await _mark_terminal(
        orchestration_session_factory, second_command.id, status=CommandStatus.succeeded
    )
    async with orchestration_session_factory() as session:
        await session.execute(
            update(GpuSessionDeployment)
            .where(GpuSessionDeployment.id == first.id)
            .values(pending_restart_since=datetime.now(UTC) - timedelta(seconds=11))
        )
        await session.execute(
            update(GpuSessionDeployment)
            .where(GpuSessionDeployment.id == second.id)
            .values(pending_restart_since=datetime.now(UTC))
        )
        await session.commit()
    await _create_job(
        orchestration_session_factory,
        user=user,
        gpu_session=gpu_session,
        model=ModelType.AISHA_IMAGE.value,
        status="running",
    )

    await worker.run_once()

    first = await _get_deployment(orchestration_session_factory, first.id)
    second = await _get_deployment(orchestration_session_factory, second.id)
    assert first.restart_operation_id is not None
    assert first.restart_operation_id == second.restart_operation_id


async def test_terminal_cascade_cannot_be_resurrected_by_restart_success(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A completed restart must not reactivate a deployment the session stopped."""
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)
    worker = _worker(orchestration_session_factory, command_service, settings)
    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    deployment = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
        pending_restart=True,
    )
    restart_command = await _create_raw_batch_command(
        orchestration_session_factory,
        gpu_session=gpu_session,
        deployment=deployment,
        batch_id=f"batch-{uuid4().hex}",
        batch_index=0,
        batch_total=1,
        kind=OperationKind.comfyui_restart,
    )
    async with orchestration_session_factory() as session, session.begin():
        repo = GpuSessionDeploymentRepository(session)
        assert (
            await repo.set_restart_pointer(
                [deployment.id], operation_id=restart_command.operation_id
            )
            == 1
        )
        assert (
            await repo.mark_status(
                gpu_session.id,
                from_statuses=(DeploymentStatus.deploying,),
                to_status=DeploymentStatus.removed,
                at=datetime.now(UTC),
            )
            == 1
        )

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.status == DeploymentStatus.removed
    assert deployment.pending_restart is False
    assert deployment.restart_operation_id is None
    await _mark_terminal(
        orchestration_session_factory, restart_command.id, status=CommandStatus.succeeded
    )

    await worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.status == DeploymentStatus.removed
    async with orchestration_session_factory() as session:
        assert (
            await GpuSessionDeploymentRepository(session).get_live_for_model(
                user.id, "vex", ModelType.AISHA_VIDEO.value
            )
            is None
        )


# ---------------------------------------------------------------------------
# CO5 / invariant #13: removal outcomes
# ---------------------------------------------------------------------------


def test_already_removed_marker_matches_pinned_aisha_message_format() -> None:
    """Keep the intentional coupling to aisha's pinned remover.py prose visible."""
    aisha_error = (
        "bundle 'video-bundle' is not recorded in residency; "
        "resident bundles: primary-bundle, video-bundle"
    )
    assert _ALREADY_REMOVED_ERROR_MARKER in aisha_error
    assert _is_already_removed_error(aisha_error)


async def test_removal_tolerates_already_removed_bundle(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """CO5: a D23 re-claim can run the removal command twice; aisha's second run
    fails with 'not recorded in residency' — the worker must treat that as success."""
    settings = _StubSettings()
    worker = _worker(
        orchestration_session_factory,
        _command_service(orchestration_session_factory, settings),
        settings,
    )

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    deployment = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
        status=DeploymentStatus.removing,
    )
    command = await _create_raw_batch_command(
        orchestration_session_factory,
        gpu_session=gpu_session,
        deployment=deployment,
        batch_id=f"batch-{uuid4().hex}",
        batch_index=0,
        batch_total=1,
        kind=OperationKind.bundle_removal,
    )
    await _mark_terminal(
        orchestration_session_factory,
        command.id,
        status=CommandStatus.failed,
        error="bundle 'video-bundle' is not recorded in residency; resident bundles: primary-bundle",
    )

    await worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.status == DeploymentStatus.removed
    assert deployment.removed_at is not None


async def test_removal_failure_returns_deployment_to_active(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Invariant #13: a genuine removal failure (not the CO5 tolerance case)
    returns the deployment to 'active' — the bundle is still resident."""
    settings = _StubSettings()
    worker = _worker(
        orchestration_session_factory,
        _command_service(orchestration_session_factory, settings),
        settings,
    )

    user = await _create_user(orchestration_session_factory)
    gpu_session = await _create_gpu_session(orchestration_session_factory, user=user)
    deployment = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
        status=DeploymentStatus.removing,
    )
    command = await _create_raw_batch_command(
        orchestration_session_factory,
        gpu_session=gpu_session,
        deployment=deployment,
        batch_id=f"batch-{uuid4().hex}",
        batch_index=0,
        batch_total=1,
        kind=OperationKind.bundle_removal,
    )
    await _mark_terminal(
        orchestration_session_factory,
        command.id,
        status=CommandStatus.failed,
        error="disk I/O error",
    )

    await worker.run_once()

    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.status == DeploymentStatus.active


# ---------------------------------------------------------------------------
# Invariant #14: OperationEventService stays a pure writer
# ---------------------------------------------------------------------------


async def test_operation_event_service_never_transitions_a_deployment(
    orchestration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    settings = _StubSettings()
    command_service = _command_service(orchestration_session_factory, settings)

    user = await _create_user(orchestration_session_factory)
    token = "test-callback-token"
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    gpu_session = await _create_gpu_session(
        orchestration_session_factory, user=user, callback_token_hash=token_hash
    )
    deployment = await _create_deployment(
        orchestration_session_factory,
        gpu_session=gpu_session,
        is_primary=False,
        model_type=ModelType.AISHA_VIDEO.value,
        status=DeploymentStatus.deploying,
    )
    commands = await command_service.enqueue_batch(
        session_id=gpu_session.id,
        product_id="vex",
        commands=[ProvisionCommand(bundle="video-bundle", mode="additive")],
        deployment_id=deployment.id,
    )
    operation_id = commands[0].operation_id

    receiver = OperationEventService(session_factory=orchestration_session_factory)
    now = datetime.now(UTC)
    authorized, status_code = await receiver.handle_event(
        session_id=gpu_session.id,
        bearer_token=token,
        event=OperationEventBody(
            schema_version=2,
            event_id="evt-1",
            session_id=gpu_session.id,
            operation_id=operation_id,
            operation_kind=OperationKind.bundle_provision,
            batch=None,
            sequence=1,
            target=None,
            status=OperationStatus.succeeded,
            phase=None,
            started_at=now,
            ts=now,
            elapsed_seconds=1.0,
            phase_elapsed_seconds=None,
            progress=None,
            plan=None,
            summary=None,
            message="done",
            error=None,
        ),
    )

    assert authorized is True
    assert status_code == 200

    # The command DID close (P3/D27's cross-write) — proving the event was really applied.
    async with orchestration_session_factory() as session:
        command = await GpuSessionCommandRepository(session).get_by_operation(operation_id)
    assert command is not None
    assert command.status == CommandStatus.succeeded

    # But the deployment itself is untouched — that's DeploymentOrchestrationWorker's job.
    deployment = await _get_deployment(orchestration_session_factory, deployment.id)
    assert deployment.status == DeploymentStatus.deploying
    assert deployment.pending_restart is False
