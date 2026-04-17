"""Integration tests for GpuSessionRepository against a real PostgreSQL database."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import GpuSessionStatus
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User
from src.db.repositories.gpu_session import GpuSessionRepository

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
        tunnel_hostname="gpu-abc.gpu.cloudin.space",
        vastai_instance_id=12345,
        vastai_offer_id=99,
        vastai_cost_per_hour_micros=500_000,
        vastai_gpu_name="RTX_4090",
        callback_token="tok-secret",
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
    assert row.tunnel_hostname == "gpu-abc.gpu.cloudin.space"
    assert row.vastai_instance_id == 12345
    assert row.vastai_offer_id == 99
    assert row.vastai_cost_per_hour_micros == 500_000
    assert row.vastai_gpu_name == "RTX_4090"
    assert row.callback_token == "tok-secret"
    assert row.provision_attempt == 1


# ---------------------------------------------------------------------------
# Tests: get_active_for_model
# ---------------------------------------------------------------------------


async def test_get_active_for_model_returns_active(
    gpu_repo: GpuSessionRepository,
    make_gpu_session: GpuSessionFactory,
) -> None:
    row = await make_gpu_session(status=GpuSessionStatus.active)
    found = await gpu_repo.get_active_for_model(row.user_id, row.product_id, row.model_type)
    assert found is not None
    assert found.id == row.id


async def test_get_active_for_model_ignores_non_active(
    gpu_repo: GpuSessionRepository,
    make_gpu_session: GpuSessionFactory,
) -> None:
    for status in (
        GpuSessionStatus.pending,
        GpuSessionStatus.provisioning,
        GpuSessionStatus.stale,
        GpuSessionStatus.paused,
    ):
        row = await make_gpu_session(status=status)
        found = await gpu_repo.get_active_for_model(row.user_id, row.product_id, row.model_type)
        assert found is None, f"Expected None for status={status}"


# ---------------------------------------------------------------------------
# Tests: get_non_terminal_for_model
# ---------------------------------------------------------------------------


async def test_get_non_terminal_for_model_returns_non_terminal(
    gpu_repo: GpuSessionRepository,
    make_gpu_session: GpuSessionFactory,
) -> None:
    for status in (
        GpuSessionStatus.pending,
        GpuSessionStatus.provisioning,
        GpuSessionStatus.active,
        GpuSessionStatus.stale,
        GpuSessionStatus.paused,
        GpuSessionStatus.resuming,
    ):
        row = await make_gpu_session(status=status)
        found = await gpu_repo.get_non_terminal_for_model(
            row.user_id, row.product_id, row.model_type
        )
        assert found is not None, f"Expected a row for status={status}"
        assert found.id == row.id


async def test_get_non_terminal_for_model_excludes_terminal(
    gpu_repo: GpuSessionRepository,
    make_gpu_session: GpuSessionFactory,
) -> None:
    for status in (GpuSessionStatus.stopped, GpuSessionStatus.failed):
        row = await make_gpu_session(status=status)
        found = await gpu_repo.get_non_terminal_for_model(
            row.user_id, row.product_id, row.model_type
        )
        assert found is None, f"Expected None for terminal status={status}"


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
# Tests: partial unique index enforcement
# ---------------------------------------------------------------------------


async def test_unique_index_blocks_two_non_terminal_same_model(
    make_gpu_session: GpuSessionFactory,
    make_user: UserFactory,
    db_session: AsyncSession,
) -> None:
    user = await make_user(email=f"uniq-{uuid4().hex[:8]}@example.com")
    await make_gpu_session(
        user=user,
        product_id="vex",
        model_type="aisha-image",
        status=GpuSessionStatus.active,
    )

    repo2 = GpuSessionRepository(db_session)
    with pytest.raises(IntegrityError):
        await repo2.create(
            id=uuid4(),
            user_id=user.id,
            product_id="vex",
            status=GpuSessionStatus.pending,
            bundle_name="wan_2.2_i2v",
            model_type="aisha-image",
        )
        await db_session.flush()


async def test_unique_index_allows_terminal_then_new(
    make_gpu_session: GpuSessionFactory,
    make_user: UserFactory,
    db_session: AsyncSession,
) -> None:
    user = await make_user(email=f"term-{uuid4().hex[:8]}@example.com")
    # Create and immediately stop a session
    stopped = await make_gpu_session(
        user=user,
        product_id="vex",
        model_type="aisha-image",
        status=GpuSessionStatus.stopped,
    )
    assert stopped is not None

    # Should be allowed: previous session is terminal
    repo2 = GpuSessionRepository(db_session)
    new_session = await repo2.create(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.pending,
        bundle_name="wan_2.2_i2v",
        model_type="aisha-image",
    )
    assert new_session.id != stopped.id
