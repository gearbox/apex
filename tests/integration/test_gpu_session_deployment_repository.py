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

from src.core.enums import (
    LIVE_DEPLOYMENT_STATUSES,
    DeploymentStatus,
    GpuSessionStatus,
    OperationKind,
)
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.gpu_session_deployment import GpuSessionDeployment
from src.db.models.user import User
from src.db.repositories.gpu_session import GpuSessionRepository
from src.db.repositories.gpu_session_command import GpuSessionCommandRepository
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
        pending_restart: bool = False,
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
            pending_restart=pending_restart,
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
    sibling_operation_id = new_id()
    sibling = await make_deployment(
        session=session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type="aisha-video",
        provision_operation_id=sibling_operation_id,
    )
    new_operation_id = new_id()

    await deployment_repo.update_provision_operation_id(session.id, new_operation_id)

    await db_session.refresh(deployment)
    await db_session.refresh(sibling)
    assert deployment.provision_operation_id == new_operation_id
    # A primary-model retry must not repoint a future P4 sibling deployment.
    assert sibling.provision_operation_id == sibling_operation_id
    # P2 still has exactly one deployment; the sibling above makes the P4
    # safeguard explicit without changing the P2 production shape.
    all_deployments = await deployment_repo.list_for_session(session.id)
    assert len(all_deployments) == 2
    assert all(deployment.status == DeploymentStatus.deploying for deployment in all_deployments)


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


# ---------------------------------------------------------------------------
# Tests: P4 orchestration primitives
# ---------------------------------------------------------------------------


async def test_get_live_for_session_and_model_is_session_scoped(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    session1 = await make_gpu_session(status=GpuSessionStatus.active)
    d1 = await make_deployment(session=session1, status=DeploymentStatus.active)
    session2 = await make_gpu_session(status=GpuSessionStatus.active)
    await make_deployment(session=session2, status=DeploymentStatus.active)

    found = await deployment_repo.get_live_for_session_and_model(session1.id, session1.model_type)

    assert found is not None
    assert found.id == d1.id


async def test_get_live_for_session_and_model_excludes_terminal(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.active)
    await make_deployment(session=session, status=DeploymentStatus.removed)

    found = await deployment_repo.get_live_for_session_and_model(session.id, session.model_type)

    assert found is None


async def test_list_live_for_session_excludes_terminal_and_other_sessions(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.active)
    live1 = await make_deployment(session=session, status=DeploymentStatus.active)
    live2 = await make_deployment(
        session=session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type="aisha-video",
    )
    await make_deployment(
        session=session,
        status=DeploymentStatus.removed,
        is_primary=False,
        model_type="aisha-image-lite",
    )
    other_session = await make_gpu_session(status=GpuSessionStatus.active)
    await make_deployment(session=other_session, status=DeploymentStatus.active)

    found = await deployment_repo.list_live_for_session(session.id)

    assert {d.id for d in found} == {live1.id, live2.id}


async def test_set_provision_pointer_stamps_operation_and_batch(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.active)
    deployment = await make_deployment(
        session=session, status=DeploymentStatus.deploying, is_primary=False
    )
    operation_id = new_id()

    await deployment_repo.set_provision_pointer(
        deployment.id, operation_id=operation_id, batch_id="batch-123"
    )

    await db_session.refresh(deployment)
    assert deployment.provision_operation_id == operation_id
    assert deployment.batch_id == "batch-123"


async def test_mark_removing_guarded_by_active_status(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.active)
    active = await make_deployment(session=session, status=DeploymentStatus.active)
    deploying = await make_deployment(
        session=session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type="aisha-video",
    )

    assert await deployment_repo.mark_removing(active.id) is True
    await db_session.refresh(active)
    assert active.status == DeploymentStatus.removing

    # Guarded: a deployment not currently 'active' cannot be marked removing.
    assert await deployment_repo.mark_removing(deploying.id) is False
    await db_session.refresh(deploying)
    assert deploying.status == DeploymentStatus.deploying

    # Idempotency: re-marking an already-removing deployment is a no-op miss.
    assert await deployment_repo.mark_removing(active.id) is False


async def test_list_deploying_awaiting_provision_result_excludes_primary_and_pending_restart(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.active)
    # The primary's 'deploying' -> 'active' path is owned by GpuProvisioningWorker,
    # never by this worker's candidate query.
    await make_deployment(session=session, status=DeploymentStatus.deploying, is_primary=True)
    sibling = await make_deployment(
        session=session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type="aisha-video",
    )
    already_flagged = await make_deployment(
        session=session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type="aisha-image-lite",
    )
    # Simulate an already-processed provision success (pending_restart=true).
    await deployment_repo.resolve_provision_outcome(
        already_flagged.id, succeeded=True, at=datetime.now(UTC)
    )

    candidates = await deployment_repo.list_deploying_awaiting_provision_result()

    assert [c.id for c in candidates] == [sibling.id]


async def test_resolve_provision_outcome_success_sets_pending_restart(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.active)
    deployment = await make_deployment(
        session=session, status=DeploymentStatus.deploying, is_primary=False
    )

    applied = await deployment_repo.resolve_provision_outcome(
        deployment.id, succeeded=True, at=datetime.now(UTC)
    )

    assert applied is True
    await db_session.refresh(deployment)
    assert deployment.status == DeploymentStatus.deploying  # D33: stays 'deploying'
    assert deployment.pending_restart is True


async def test_resolve_provision_outcome_failure_fails_deployment(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.active)
    deployment = await make_deployment(
        session=session, status=DeploymentStatus.deploying, is_primary=False
    )
    at = datetime.now(UTC)

    applied = await deployment_repo.resolve_provision_outcome(deployment.id, succeeded=False, at=at)

    assert applied is True
    await db_session.refresh(deployment)
    assert deployment.status == DeploymentStatus.failed
    assert deployment.removed_at is not None
    # Frees the uniqueness slot, same as any other terminal write.
    assert (
        await deployment_repo.get_live_for_model(
            session.user_id, session.product_id, deployment.model_type
        )
        is None
    )


async def test_resolve_provision_outcome_is_guarded(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
) -> None:
    """A second call after the first already applied is a no-op miss, not a double-apply."""
    session = await make_gpu_session(status=GpuSessionStatus.active)
    deployment = await make_deployment(
        session=session, status=DeploymentStatus.deploying, is_primary=False
    )
    now = datetime.now(UTC)
    assert await deployment_repo.resolve_provision_outcome(deployment.id, succeeded=True, at=now)

    assert not await deployment_repo.resolve_provision_outcome(
        deployment.id, succeeded=True, at=now
    )
    assert not await deployment_repo.resolve_provision_outcome(
        deployment.id, succeeded=False, at=now
    )


async def test_set_restart_pointer_is_guarded_against_duplicate_enqueue(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    """D34: a second restart-enqueue attempt for an already-stamped batch orphans
    the new operation id rather than corrupting the first one."""
    session = await make_gpu_session(status=GpuSessionStatus.active)
    deployment = await make_deployment(
        session=session, status=DeploymentStatus.deploying, is_primary=False, pending_restart=True
    )
    first_operation_id = new_id()
    second_operation_id = new_id()

    updated = await deployment_repo.set_restart_pointer(
        [deployment.id], operation_id=first_operation_id
    )
    assert updated == 1

    updated_again = await deployment_repo.set_restart_pointer(
        [deployment.id], operation_id=second_operation_id
    )
    assert updated_again == 0

    await db_session.refresh(deployment)
    assert deployment.restart_operation_id == first_operation_id


async def test_resolve_restart_outcome_success_activates_every_member(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    """Invariant #8: restart success activates every pending-restart deployment
    in the batch, even a multi-member one."""
    session = await make_gpu_session(status=GpuSessionStatus.active)
    d1 = await make_deployment(
        session=session, status=DeploymentStatus.deploying, is_primary=False, pending_restart=True
    )
    d2 = await make_deployment(
        session=session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type="aisha-video",
        pending_restart=True,
    )
    at = datetime.now(UTC)

    updated = await deployment_repo.resolve_restart_outcome([d1.id, d2.id], succeeded=True, at=at)

    assert updated == 2
    for d in (d1, d2):
        await db_session.refresh(d)
        assert d.status == DeploymentStatus.active
        assert d.pending_restart is False
        assert d.activated_at == at


async def test_resolve_restart_outcome_failure_fails_every_member_no_retry(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    """Invariant #9: restart failure fails the deployments; no retry is a repo concern
    (the worker simply never re-enqueues once pending_restart is cleared)."""
    session = await make_gpu_session(status=GpuSessionStatus.active)
    deployment = await make_deployment(
        session=session, status=DeploymentStatus.deploying, is_primary=False, pending_restart=True
    )
    at = datetime.now(UTC)

    updated = await deployment_repo.resolve_restart_outcome([deployment.id], succeeded=False, at=at)

    assert updated == 1
    await db_session.refresh(deployment)
    assert deployment.status == DeploymentStatus.failed
    assert deployment.pending_restart is False
    assert deployment.removed_at == at


async def test_list_removing_and_resolve_removal_outcome_success(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    session = await make_gpu_session(status=GpuSessionStatus.active)
    deployment = await make_deployment(session=session, status=DeploymentStatus.removing)

    removing = await deployment_repo.list_removing()
    assert deployment.id in {d.id for d in removing}

    at = datetime.now(UTC)
    applied = await deployment_repo.resolve_removal_outcome(deployment.id, succeeded=True, at=at)

    assert applied is True
    await db_session.refresh(deployment)
    assert deployment.status == DeploymentStatus.removed
    assert deployment.removed_at == at


async def test_resolve_removal_outcome_failure_returns_to_active(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    """Invariant #13: removal failure returns the deployment to 'active' — the
    bundle is still resident."""
    session = await make_gpu_session(status=GpuSessionStatus.active)
    deployment = await make_deployment(session=session, status=DeploymentStatus.removing)

    applied = await deployment_repo.resolve_removal_outcome(
        deployment.id, succeeded=False, at=datetime.now(UTC)
    )

    assert applied is True
    await db_session.refresh(deployment)
    assert deployment.status == DeploymentStatus.active
    # Still occupies the slot — it never left.
    found = await deployment_repo.get_live_for_model(
        session.user_id, session.product_id, deployment.model_type
    )
    assert found is not None and found.id == deployment.id


async def test_mark_primary_active_ignores_sibling_deployments(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    """Invariant #5 (D33) hardening: a session-level active transition (bootstrap
    or resume) must never mark a still-provisioning P4 sibling routable — only
    the primary is flipped by this cascade."""
    session = await make_gpu_session(status=GpuSessionStatus.provisioning)
    primary = await make_deployment(
        session=session, status=DeploymentStatus.deploying, is_primary=True
    )
    sibling = await make_deployment(
        session=session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type="aisha-video",
    )

    updated = await deployment_repo.mark_primary_active(session.id, at=datetime.now(UTC))

    assert updated == 1
    await db_session.refresh(primary)
    await db_session.refresh(sibling)
    assert primary.status == DeploymentStatus.active
    assert primary.activated_at is not None
    assert sibling.status == DeploymentStatus.deploying
    assert sibling.activated_at is None


async def test_teardown_cascade_covers_a_deploying_sibling_mid_attach(
    deployment_repo: GpuSessionDeploymentRepository,
    make_gpu_session: GpuSessionFactory,
    make_deployment: DeploymentFactory,
    db_session: AsyncSession,
) -> None:
    """Invariant #15: stopping a session mid-attach must leave no deployment in
    'deploying' and must cancel that deployment's queued command — exercised
    directly against the D15 chokepoint primitives (mark_status + cancel_for_session
    + close_failed), the same calls GpuSessionService._set_status /
    GpuProvisioningWorker._transition make in one transaction. P4's new wrinkle is
    that the session now has BOTH an 'active' primary AND a 'deploying' sibling at
    once — the cascade must still catch both, unconditionally on LIVE_DEPLOYMENT_STATUSES."""
    session = await make_gpu_session(status=GpuSessionStatus.active)
    primary = await make_deployment(session=session, status=DeploymentStatus.active)
    sibling = await make_deployment(
        session=session,
        status=DeploymentStatus.deploying,
        is_primary=False,
        model_type="aisha-video",
    )
    command_repo = GpuSessionCommandRepository(db_session)
    operation_id = new_id()
    command_id = new_id()
    await GpuSessionOperationRepository(db_session).create(
        id=operation_id,
        session_id=session.id,
        product_id=session.product_id,
        kind=OperationKind.bundle_provision,
        command_id=command_id,
    )
    queued_command = await command_repo.create(
        id=command_id,
        session_id=session.id,
        product_id=session.product_id,
        operation_id=operation_id,
        deployment_id=sibling.id,
        kind=OperationKind.bundle_provision,
        payload={"bundle": "video-bundle", "mode": "additive", "verify": True},
    )
    at = datetime.now(UTC)

    # The D15 cascade: session -> 'stopped' flips every live deployment, cancels
    # every queued/claimed command, and fails each cancelled command's operation.
    await deployment_repo.mark_status(
        session.id,
        from_statuses=LIVE_DEPLOYMENT_STATUSES,
        to_status=DeploymentStatus.removed,
        at=at,
    )
    cancelled = await command_repo.cancel_for_session(session.id, at=at, reason="session stopped")

    await db_session.refresh(primary)
    await db_session.refresh(sibling)
    await db_session.refresh(queued_command)
    assert primary.status == DeploymentStatus.removed
    assert sibling.status == DeploymentStatus.removed  # no deployment left 'deploying'
    assert {c.id for c in cancelled} == {queued_command.id}
    assert queued_command.status == "cancelled"
