"""Integration tests for GpuSessionRepository against a real PostgreSQL database."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest_asyncio

from src.core.enums import GpuSessionStatus
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User
from src.db.repositories.gpu_session import GpuSessionRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

UserFactory = Callable[..., Coroutine[Any, Any, User]]
GpuSessionFactory = Callable[..., Coroutine[Any, Any, GpuSession]]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def gpu_repo(db_session: AsyncSession) -> GpuSessionRepository:
    return GpuSessionRepository(db_session)


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
        vastai_instance_id: int | None = None,
        session_id: UUID | None = None,
    ) -> GpuSession:
        if user is None:
            user = await make_user(email=f"gpu-{uuid4().hex[:8]}@example.com")
        repo = GpuSessionRepository(db_session)
        return await repo.create(
            id=session_id or uuid4(),
            user_id=user.id,
            product_id=product_id,
            status=status,
            bundle_name=bundle_name,
            bundle_version=bundle_version,
            model_type=model_type,
            vastai_instance_id=vastai_instance_id,
        )

    return _factory


# ---------------------------------------------------------------------------
# Tests: create
# ---------------------------------------------------------------------------


async def test_create_persists_all_fields(
    gpu_repo: GpuSessionRepository,
    make_user: UserFactory,
) -> None:
    user = await make_user(email=f"create-{uuid4().hex[:8]}@example.com")
    session_id = uuid4()
    row = await gpu_repo.create(
        id=session_id,
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.pending,
        bundle_name="wan_2.2_i2v",
        bundle_version="260105-01",
        model_type="aisha-video",
        cf_tunnel_id="tun-abc",
        cf_dns_record_id="rec-xyz",
        tunnel_hostname="gpu-abc.gpu-domain.com",
        vastai_instance_id=12345,
        vastai_offer_id=99,
        vastai_cost_per_hour_micros=500_000,
        vastai_gpu_name="RTX_4090",
        callback_token_hash="a" * 64,
    )

    assert row.id == session_id
    assert row.user_id == user.id
    assert row.product_id == "vex"
    assert row.status == GpuSessionStatus.pending
    assert row.bundle_name == "wan_2.2_i2v"
    assert row.bundle_version == "260105-01"
    assert row.model_type == "aisha-video"
    assert row.cf_tunnel_id == "tun-abc"
    assert row.cf_dns_record_id == "rec-xyz"
    assert row.tunnel_hostname == "gpu-abc.gpu-domain.com"
    assert row.vastai_instance_id == 12345
    assert row.vastai_offer_id == 99
    assert row.vastai_cost_per_hour_micros == 500_000
    assert row.vastai_gpu_name == "RTX_4090"
    assert row.callback_token_hash == "a" * 64
    assert row.provision_attempt == 1


# ---------------------------------------------------------------------------
# get_active_for_model and get_non_terminal_for_model were removed in P2 —
# model identity and the uniqueness slot moved to GpuSessionDeploymentRepository
# (get_routable / get_live_for_model). See test_gpu_session_deployment_repository.py.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tests: list_by_status
# ---------------------------------------------------------------------------


async def test_list_by_status_multiple_statuses(
    gpu_repo: GpuSessionRepository,
    make_gpu_session: GpuSessionFactory,
) -> None:
    pending_row = await make_gpu_session(status=GpuSessionStatus.pending)
    provisioning_row = await make_gpu_session(status=GpuSessionStatus.provisioning)
    await make_gpu_session(status=GpuSessionStatus.active)

    rows = await gpu_repo.list_by_status(GpuSessionStatus.pending, GpuSessionStatus.provisioning)
    ids = {r.id for r in rows}
    assert pending_row.id in ids
    assert provisioning_row.id in ids


# ---------------------------------------------------------------------------
# Tests: update_status
# ---------------------------------------------------------------------------


async def test_update_status_transition(
    gpu_repo: GpuSessionRepository,
    make_gpu_session: GpuSessionFactory,
    db_session: AsyncSession,
) -> None:
    from datetime import UTC, datetime

    row = await make_gpu_session(status=GpuSessionStatus.pending)
    started = datetime.now(UTC)
    await gpu_repo.update_status(
        row.id,
        GpuSessionStatus.active,
        started_at=started,
    )

    await db_session.refresh(row)
    assert row.status == GpuSessionStatus.active
    assert row.started_at is not None


# ---------------------------------------------------------------------------
# Partial unique index enforcement moved off gpu_sessions in P2 — the slot
# now lives on gpu_session_deployments; see
# test_gpu_session_deployment_repository.py::TestLiveUserModelIndex.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tests: stopping is non-terminal at the repo layer (regression)
# ---------------------------------------------------------------------------


async def test_stopping_is_non_terminal_list_by_user(
    gpu_repo: GpuSessionRepository,
    make_gpu_session: GpuSessionFactory,
) -> None:
    """A session in `stopping` must appear in list_by_user(include_terminal=False).

    Guards the catalog path: the frontend must see `session_state="stopping"` so
    it can show "Stopping…" and disable the Start CTA.
    """
    row = await make_gpu_session(status=GpuSessionStatus.stopping)
    rows = await gpu_repo.list_by_user(row.user_id, row.product_id, include_terminal=False)
    ids = {r.id for r in rows}
    assert row.id in ids, "stopping session must appear in the non-terminal catalog"
