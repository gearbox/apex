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

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.schemas.gpu_session import OperationEventBody
from src.api.services.gpu_session.command_payload import ProvisionCommand
from src.api.services.gpu_session.command_service import GpuSessionCommandService
from src.api.services.gpu_session.deployment_orchestration_worker import (
    DeploymentOrchestrationWorker,
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
    CommandStatus,
    DeploymentStatus,
    GpuSessionStatus,
    ModelType,
    OperationKind,
    OperationStatus,
)
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.gpu_session_deployment import GpuSessionDeployment
from src.db.models.storage import GenerationJob
from src.db.models.user import User
from src.db.repositories.gpu_session_command import GpuSessionCommandRepository
from src.db.repositories.gpu_session_deployment import GpuSessionDeploymentRepository
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.db.models.gpu_session_command import GpuSessionCommand


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
) -> DeploymentOrchestrationWorker:
    return DeploymentOrchestrationWorker(
        session_factory=session_factory,
        command_service=command_service,
        settings=settings,  # type: ignore[arg-type]
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
) -> None:
    async with session_factory() as session, session.begin():
        applied = await GpuSessionCommandRepository(session).mark_terminal(
            command_id, status=status, at=datetime.now(UTC), error=error
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


# ---------------------------------------------------------------------------
# D37 / last-deployment guard / D11 retain_bundles
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CO5 / invariant #13: removal outcomes
# ---------------------------------------------------------------------------


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
