"""Integration tests for GpuSessionDeploymentRepository against a real PostgreSQL database.

Covers the P2 contracts and invariants listed in the deployments-routing design:
the slot moved off gpu_sessions (1), it frees on stop/failure (2, 3) but not
pause (4), routing requires both session and deployment active (5), the orphan
guard self-heals (9), creation is transactionally atomic with the session and
operation inserts (10), and the sessions-list endpoint's batch fetch avoids N+1
(13). Migration backfill/round-trip (11, 12) live in
test_gpu_session_deployments_migration.py.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.core.enums import DeploymentStatus, GpuSessionStatus, OperationKind
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.gpu_session_deployment import GpuSessionDeployment
from src.db.models.user import User
from src.db.repositories.gpu_session import GpuSessionRepository
from src.db.repositories.gpu_session_deployment import GpuSessionDeploymentRepository
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

UserFactory = Callable[..., Coroutine[Any, Any, User]]
GpuSessionFactory = Callable[..., Coroutine[Any, Any, GpuSession]]
DeploymentFactory = Callable[..., Coroutine[Any, Any, GpuSessionDeployment]]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def deployment_repo(db_session: AsyncSession) -> GpuSessionDeploymentRepository:
    return GpuSessionDeploymentRepository(db_session)


@pytest_asyncio.fixture
async def make_gpu_session(
    db_session: AsyncSession,
    make_user: UserFactory,
) -> GpuSessionFactory:
    async def _factory(
        *,
        user: User | None = None,
        product_id: str = "vex",
        status: str = GpuSessionStatus.pending,
        bundle_name: str = "wan_2.2_i2v",
        model_type: str = "aisha-image",
        bundle_version: str | None = None,
        session_id: UUID | None = None,
    ) -> GpuSession:
        if user is None:
            user = await make_user(email=f"gpu-dep-{uuid4().hex[:8]}@example.com")
        repo = GpuSessionRepository(db_session)
        return await repo.create(
            id=session_id or new_id(),
            user_id=user.id,
            product_id=product_id,
            status=status,
            bundle_name=bundle_name,
            bundle_version=bundle_version,
            model_type=model_type,
        )

    return _factory


@pytest_asyncio.fixture
async def make_deployment(
    db_session: AsyncSession,
) -> DeploymentFactory:
    async def _factory(
        *,
        session: GpuSession,
        status: str = DeploymentStatus.active,
        is_primary: bool = True,
        model_type: str | None = None,
        bundle_name: str | None = None,
        bundle_version: str | None = None,
        readiness_marker_node_class: str | None = None,
        provision_operation_id: UUID | None = None,
    ) -> GpuSessionDeployment:
        repo = GpuSessionDeploymentRepository(db_session)
        return await repo.create(
            id=new_id(),
            session_id=session.id,
            user_id=session.user_id,
            product_id=session.product_id,
            model_type=model_type or session.model_type,
            bundle_name=bundle_name or session.bundle_name,
            bundle_version=bundle_version if bundle_version is not None else session.bundle_version,
            readiness_marker_node_class=readiness_marker_node_class,
            status=status,
            is_primary=is_primary,
            provision_operation_id=provision_operation_id,
        )

    return _factory


# ---------------------------------------------------------------------------
# Tests: create / get_live_for_model
# ---------------------------------------------------------------------------


async def test_create_persists_all_fields(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
) -> None:
    session = await make_gpu_session()
    operation_id = new_id()
    deployment = await deployment_repo.create(
        id=new_id(),
        session_id=session.id,
        user_id=session.user_id,
        product_id="vex",
        model_type="aisha-image",
        bundle_name="wan_2.2_i2v",
        bundle_version="260105-01",
        readiness_marker_node_class="WanVideoSampler",
        status=DeploymentStatus.deploying,
        provision_operation_id=operation_id,
        is_primary=True,
    )

    assert deployment.session_id == session.id
    assert deployment.user_id == session.user_id
    assert deployment.status == DeploymentStatus.deploying
    assert deployment.readiness_marker_node_class == "WanVideoSampler"
    assert deployment.provision_operation_id == operation_id
    assert deployment.is_primary is True
    assert deployment.pending_restart is False
    assert deployment.activated_at is None
    assert deployment.removed_at is None


async def test_get_live_for_model_returns_live_statuses(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    for status in (
        DeploymentStatus.deploying,
        DeploymentStatus.active,
        DeploymentStatus.removing,
    ):
        session = await make_gpu_session()
        deployment = await make_deployment(session=session, status=status)
        found = await deployment_repo.get_live_for_model(
            session.user_id, session.product_id, session.model_type
        )
        assert found is not None, f"Expected a row for status={status}"
        assert found.id == deployment.id


async def test_get_live_for_model_excludes_terminal(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    for status in (DeploymentStatus.removed, DeploymentStatus.failed):
        session = await make_gpu_session()
        await make_deployment(session=session, status=status)
        found = await deployment_repo.get_live_for_model(
            session.user_id, session.product_id, session.model_type
        )
        assert found is None, f"Expected None for terminal status={status}"


# ---------------------------------------------------------------------------
# Tests: partial unique index — invariant 1 (the slot moved)
# ---------------------------------------------------------------------------


async def test_unique_index_blocks_two_live_deployments_same_model(
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    session1 = await make_gpu_session(status=GpuSessionStatus.active)
    await make_deployment(session=session1, status=DeploymentStatus.active)

    session2 = await make_gpu_session(
        user=None,
        status=GpuSessionStatus.pending,
    )
    # Force the same (user, product, model_type) as session1's deployment —
    # simulate a second session for the same user/model racing in.
    repo2 = GpuSessionDeploymentRepository(db_session)
    with pytest.raises(IntegrityError):
        await repo2.create(
            id=new_id(),
            session_id=session2.id,
            user_id=session1.user_id,
            product_id=session1.product_id,
            model_type=session1.model_type,
            bundle_name="wan_2.2_i2v",
            status=DeploymentStatus.deploying,
        )
        await db_session.flush()


async def test_unique_index_allows_different_model_types(
    make_gpu_session: GpuSessionFactory,
    make_user: UserFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    """One user can hold concurrent live deployments for two different model_types."""
    user = await make_user(email=f"multi-model-{uuid4().hex[:8]}@example.com")
    session1 = await make_gpu_session(user=user, model_type="aisha-image")
    await make_deployment(session=session1, status=DeploymentStatus.active)

    session2 = await make_gpu_session(user=user, model_type="aisha-image-lite")
    repo2 = GpuSessionDeploymentRepository(db_session)
    second = await repo2.create(
        id=new_id(),
        session_id=session2.id,
        user_id=user.id,
        product_id="vex",
        model_type="aisha-image-lite",
        bundle_name="zit_cyberrealistic",
        status=DeploymentStatus.active,
    )
    await db_session.flush()

    assert second.model_type == "aisha-image-lite"


async def test_unique_index_allows_terminal_then_new(
    make_gpu_session: GpuSessionFactory,
    make_user: UserFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    """Invariant 2/3: the slot frees once the prior deployment is removed/failed."""
    user = await make_user(email=f"term-dep-{uuid4().hex[:8]}@example.com")
    stopped_session = await make_gpu_session(user=user, status=GpuSessionStatus.stopped)
    await make_deployment(session=stopped_session, status=DeploymentStatus.removed)

    new_session = await make_gpu_session(user=user, status=GpuSessionStatus.pending)
    repo2 = GpuSessionDeploymentRepository(db_session)
    new_deployment = await repo2.create(
        id=new_id(),
        session_id=new_session.id,
        user_id=user.id,
        product_id="vex",
        model_type="aisha-image",
        bundle_name="wan_2.2_i2v",
        status=DeploymentStatus.deploying,
    )
    assert new_deployment.session_id == new_session.id


# ---------------------------------------------------------------------------
# Tests: mark_status — the D15 cascade primitive (invariants 2, 3, 4, 7)
# ---------------------------------------------------------------------------


async def test_mark_status_active_stamps_activated_at(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    session = await make_gpu_session()
    deployment = await make_deployment(session=session, status=DeploymentStatus.deploying)
    at = datetime.now(UTC)

    count = await deployment_repo.mark_status(
        session.id,
        from_statuses=(DeploymentStatus.deploying,),
        to_status=DeploymentStatus.active,
        at=at,
    )

    assert count == 1
    await db_session.refresh(deployment)
    assert deployment.status == DeploymentStatus.active
    assert deployment.activated_at is not None


async def test_mark_status_stop_frees_the_slot(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    """Invariant 2: after mark_status(removed), the same model is startable again."""
    session = await make_gpu_session(status=GpuSessionStatus.active)
    await make_deployment(session=session, status=DeploymentStatus.active)

    from src.core.enums import LIVE_DEPLOYMENT_STATUSES

    await deployment_repo.mark_status(
        session.id,
        from_statuses=LIVE_DEPLOYMENT_STATUSES,
        to_status=DeploymentStatus.removed,
        at=datetime.now(UTC),
    )

    found = await deployment_repo.get_live_for_model(
        session.user_id, session.product_id, session.model_type
    )
    assert found is None, "slot must be free immediately after stop"


async def test_mark_status_failed_frees_the_slot(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    """Invariant 3: same as stop, via _mark_failed's cascade."""
    session = await make_gpu_session(status=GpuSessionStatus.provisioning)
    await make_deployment(session=session, status=DeploymentStatus.deploying)

    from src.core.enums import LIVE_DEPLOYMENT_STATUSES

    await deployment_repo.mark_status(
        session.id,
        from_statuses=LIVE_DEPLOYMENT_STATUSES,
        to_status=DeploymentStatus.failed,
        at=datetime.now(UTC),
    )

    found = await deployment_repo.get_live_for_model(
        session.user_id, session.product_id, session.model_type
    )
    assert found is None


async def test_mark_status_is_guarded_by_from_statuses(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    """A deployment not in from_statuses is left untouched (guarded UPDATE)."""
    session = await make_gpu_session()
    deployment = await make_deployment(session=session, status=DeploymentStatus.active)

    count = await deployment_repo.mark_status(
        session.id,
        from_statuses=(DeploymentStatus.deploying,),  # deployment is 'active', not 'deploying'
        to_status=DeploymentStatus.active,
        at=datetime.now(UTC),
    )

    assert count == 0
    await db_session.refresh(deployment)
    assert deployment.activated_at is None


async def test_pause_does_not_free_the_slot(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    """Invariant 4: D17 — a paused session's deployment stays 'active'.

    Simulated here by simply never invoking mark_status for a pause/resume
    transition (pause/resume never call it — see GpuSessionService._set_status
    and D17), then confirming the slot is still occupied.
    """
    session = await make_gpu_session(status=GpuSessionStatus.paused)
    await make_deployment(session=session, status=DeploymentStatus.active)

    found = await deployment_repo.get_live_for_model(
        session.user_id, session.product_id, session.model_type
    )
    assert found is not None, "paused session's deployment must still occupy the slot"


# ---------------------------------------------------------------------------
# Tests: get_routable — invariant 5 (routing requires both active)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "session_status",
    [
        GpuSessionStatus.stopping,
        GpuSessionStatus.paused,
        GpuSessionStatus.stale,
        GpuSessionStatus.provisioning,
    ],
)
async def test_get_routable_none_when_session_not_active(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    session_status: str,
) -> None:
    session = await make_gpu_session(status=session_status)
    await make_deployment(session=session, status=DeploymentStatus.active)

    result = await deployment_repo.get_routable(
        session.user_id, session.product_id, session.model_type
    )
    assert result is None


@pytest.mark.parametrize(
    "deployment_status",
    [DeploymentStatus.deploying, DeploymentStatus.removing, DeploymentStatus.failed],
)
async def test_get_routable_none_when_deployment_not_active(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    deployment_status: str,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.active)
    await make_deployment(session=session, status=deployment_status)

    result = await deployment_repo.get_routable(
        session.user_id, session.product_id, session.model_type
    )
    assert result is None


async def test_get_routable_returns_pair_when_both_active(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.active)
    deployment = await make_deployment(session=session, status=DeploymentStatus.active)

    result = await deployment_repo.get_routable(
        session.user_id, session.product_id, session.model_type
    )

    assert result is not None
    found_session, found_deployment = result
    assert found_session.id == session.id
    assert found_deployment.id == deployment.id


async def test_get_routable_uses_deployment_bundle_identity(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    """Invariant 6: the session and deployment may disagree on bundle identity;
    callers (handlers.py) must read the deployment's, not the session's."""
    session = await make_gpu_session(status=GpuSessionStatus.active, bundle_name="session_bundle")
    deployment = await make_deployment(
        session=session, status=DeploymentStatus.active, bundle_name="deployment_bundle"
    )

    result = await deployment_repo.get_routable(
        session.user_id, session.product_id, session.model_type
    )

    assert result is not None
    found_session, found_deployment = result
    assert found_session.bundle_name == "session_bundle"
    assert found_deployment.bundle_name == "deployment_bundle"
    assert deployment.bundle_name == "deployment_bundle"


# ---------------------------------------------------------------------------
# Tests: list_for_session / list_for_sessions — invariant 13 (no N+1)
# ---------------------------------------------------------------------------


async def test_list_for_session_returns_only_that_sessions_deployments(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    session1 = await make_gpu_session()
    session2 = await make_gpu_session()
    d1 = await make_deployment(session=session1, status=DeploymentStatus.active)
    await make_deployment(session=session2, status=DeploymentStatus.active)

    result = await deployment_repo.list_for_session(session1.id)

    assert [d.id for d in result] == [d1.id]


async def test_list_for_sessions_batches_across_many_sessions(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    """One call covers N sessions — the mechanism the list endpoint uses to
    avoid an N+1 (a single ``session_id IN (...)`` query, not one per session)."""
    sessions = [await make_gpu_session() for _ in range(5)]
    expected: dict[UUID, UUID] = {}
    for session in sessions:
        d = await make_deployment(session=session, status=DeploymentStatus.active)
        expected[session.id] = d.id

    by_session = await deployment_repo.list_for_sessions([s.id for s in sessions])

    assert set(by_session.keys()) == set(expected.keys())
    for session_id, deployment_id in expected.items():
        assert [d.id for d in by_session[session_id]] == [deployment_id]


async def test_list_for_sessions_empty_input_returns_empty_dict(
    deployment_repo: GpuSessionDeploymentRepository,
) -> None:
    assert await deployment_repo.list_for_sessions([]) == {}


# ---------------------------------------------------------------------------
# Tests: update_provision_operation_id — invariant 8 (retry does not fork)
# ---------------------------------------------------------------------------


async def test_update_provision_operation_id_repoints_without_forking(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.provisioning)
    original_operation_id = new_id()
    deployment = await make_deployment(
        session=session,
        status=DeploymentStatus.deploying,
        provision_operation_id=original_operation_id,
    )
    new_operation_id = new_id()

    await deployment_repo.update_provision_operation_id(session.id, new_operation_id)

    await db_session.refresh(deployment)
    assert deployment.provision_operation_id == new_operation_id
    # Still exactly one deployment for the session — retry did not fork.
    all_deployments = await deployment_repo.list_for_session(session.id)
    assert len(all_deployments) == 1
    assert all_deployments[0].status == DeploymentStatus.deploying


# ---------------------------------------------------------------------------
# Tests: mark_orphaned — invariant 9 (the orphan guard self-heals)
# ---------------------------------------------------------------------------


async def test_mark_orphaned_removes_live_deployment_of_terminal_session(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.stopped)
    deployment = await make_deployment(session=session, status=DeploymentStatus.active)

    orphaned = await deployment_repo.mark_orphaned(at=datetime.now(UTC))

    assert (deployment.id, session.id) in orphaned
    found = await deployment_repo.get_live_for_model(
        session.user_id, session.product_id, session.model_type
    )
    assert found is None


async def test_mark_orphaned_ignores_live_session(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.active)
    await make_deployment(session=session, status=DeploymentStatus.active)

    orphaned = await deployment_repo.mark_orphaned(at=datetime.now(UTC))

    assert orphaned == []


async def test_mark_orphaned_ignores_already_removed_deployment(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.stopped)
    await make_deployment(session=session, status=DeploymentStatus.removed)

    orphaned = await deployment_repo.mark_orphaned(at=datetime.now(UTC))

    assert orphaned == []


# ---------------------------------------------------------------------------
# Tests: creation atomicity — invariant 10
# ---------------------------------------------------------------------------


async def test_session_operation_deployment_insert_is_atomic(
    db_session: AsyncSession,
    make_user: UserFactory,
) -> None:
    """If something fails after all three inserts but before commit (e.g. the
    billing reservation in GpuSessionService.start_session step 7.5), none of
    the session/operation/deployment rows survive — they share one transaction.
    """
    user = await make_user(email=f"atomic-{uuid4().hex[:8]}@example.com")
    session_id = new_id()
    operation_id = new_id()
    deployment_id = new_id()

    class _SimulatedBillingFailure(Exception):
        pass

    with pytest.raises(_SimulatedBillingFailure):
        async with db_session.begin_nested():
            session_repo = GpuSessionRepository(db_session)
            operation_repo = GpuSessionOperationRepository(db_session)
            deployment_repo = GpuSessionDeploymentRepository(db_session)

            await session_repo.create(
                id=session_id,
                user_id=user.id,
                product_id="vex",
                status=GpuSessionStatus.pending,
                bundle_name="wan_2.2_i2v",
                model_type="aisha-image",
                bootstrap_operation_id=operation_id,
            )
            await operation_repo.create(
                id=operation_id,
                session_id=session_id,
                product_id="vex",
                kind=OperationKind.session_bootstrap,
                target_bundle="wan_2.2_i2v",
                target_mode="full",
            )
            await deployment_repo.create(
                id=deployment_id,
                session_id=session_id,
                user_id=user.id,
                product_id="vex",
                model_type="aisha-image",
                bundle_name="wan_2.2_i2v",
                status=DeploymentStatus.deploying,
                provision_operation_id=operation_id,
                is_primary=True,
            )
            # Simulates check_and_reserve raising inside the same db.begin()
            # block as the three inserts above (see GpuSessionService.start_session).
            raise _SimulatedBillingFailure("balance changed between assert and reserve")

    session_row = await db_session.execute(select(GpuSession).where(GpuSession.id == session_id))
    assert session_row.scalar_one_or_none() is None

    deployment_row = await db_session.execute(
        select(GpuSessionDeployment).where(GpuSessionDeployment.id == deployment_id)
    )
    assert deployment_row.scalar_one_or_none() is None
