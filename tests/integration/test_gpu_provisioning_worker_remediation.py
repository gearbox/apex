"""Integration coverage for provisioning-worker remediation paths."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.services.gpu_session.node_cooldown import NullNodeCooldownStore
from src.api.services.gpu_session.provisioning_worker import (
    _REASON_PENDING_TIMEOUT,
    GpuProvisioningWorker,
)
from src.api.services.vastai.schemas import VastAIOffer
from src.core.bundle_config import BundleMapping, HardwareRequirements
from src.core.enums import DeploymentStatus, GpuSessionStatus
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User
from src.db.repositories.gpu_session_deployment import GpuSessionDeploymentRepository

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


class _RetrySettings:
    """The narrow Settings surface used by the retry path."""

    gpu_provision_poll_interval_seconds = 15
    gpu_provision_worker_concurrency = 1
    provisioning_recreation_attempts = 3
    vastai_offer_search_limit = 20
    provisioning_offer_walk_depth = 1
    ai_bundles_github_token = "test-github-token"
    ai_bundles_repo_url = "https://example.test/ai-bundles.git"
    ai_bundles_branch = "main"
    aisha_repo_url = "https://example.test/aisha.git"
    aisha_branch = "main"
    apex_callback_url = "https://apex.example.test/callback"
    hf_token = "test-hf-token"
    civitai_api_token = "test-civitai-token"
    aisha_comfyui_host = "0.0.0.0"  # noqa: S104
    aisha_comfyui_extra_args = ""


@pytest.fixture
def provisioning_session_factory(
    db_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_provisioning_remediation_rows(
    provisioning_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[None]:
    yield
    async with provisioning_session_factory() as session:
        await session.execute(delete(User).where(User.email.like("retry-remediation-%")))
        await session.commit()


async def test_retry_missing_primary_destroys_new_instance_fails_and_stops_future_retries(
    provisioning_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A retry-created node is cleaned up before the D15 failure is made terminal."""
    user = User(
        id=new_id(),
        email=f"retry-remediation-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.pending,
        bundle_name="retry-bundle",
        bundle_version="20260908-01",
        model_type="aisha-image",
        vastai_instance_id=111,
        cf_tunnel_id="tunnel-id",
        cf_dns_record_id="dns-id",
    )
    async with provisioning_session_factory() as db, db.begin():
        db.add(user)
        await db.flush()
        db.add(gpu_session)
        await db.flush()
        await GpuSessionDeploymentRepository(db).create(
            id=new_id(),
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type="aisha-image",
            bundle_name="retry-bundle",
            status=DeploymentStatus.deploying,
            is_primary=True,
        )
    # Simulate the D15 violation after the session and its primary were already created.
    async with provisioning_session_factory() as db, db.begin():
        deleted = await GpuSessionDeploymentRepository(db).get_primary_for_session(gpu_session.id)
        assert deleted is not None
        await db.delete(deleted)

    hardware = HardwareRequirements(
        gpu_whitelist=("RTX_4090",),
        min_disk_gb=100,
        min_network_upload_mbps=100,
        min_network_download_mbps=500,
        cuda_min_version="12.1",
        num_gpus=1,
    )
    bundle_index = MagicMock()
    bundle_index.resolve_bundle_override.return_value = BundleMapping(
        bundle_name="retry-bundle",
        bundle_version="20260908-01",
        hardware=hardware,
    )
    vastai = AsyncMock()
    vastai.search_offers.return_value = [
        VastAIOffer(id=42, gpu_name="RTX_4090", dph_total=0.5, machine_id=77)
    ]
    vastai.create_instance.return_value = 222
    cloudflare = AsyncMock()
    cloudflare.get_tunnel_token.return_value = "fresh-tunnel-token"

    worker = GpuProvisioningWorker(
        session_factory=provisioning_session_factory,
        vastai_client=vastai,
        cf_client=cloudflare,
        bundle_index=bundle_index,
        http_client=AsyncMock(),
        settings=_RetrySettings(),  # type: ignore[arg-type]
        cooldown_store=NullNodeCooldownStore(),
        redis_enabled=False,
        redis_client_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )

    await worker._retry_or_fail(gpu_session, reason=_REASON_PENDING_TIMEOUT)

    vastai.destroy_instance.assert_any_await(222)
    async with provisioning_session_factory() as db:
        failed = await db.get(GpuSession, gpu_session.id)
    assert failed is not None
    assert failed.status == GpuSessionStatus.failed
    assert failed.error_message == "retry_missing_primary: session has no primary deployment"

    await worker.run_once()
    assert vastai.create_instance.await_count == 1
