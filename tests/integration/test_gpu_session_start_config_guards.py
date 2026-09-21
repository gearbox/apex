"""Integration coverage for start_session's D6 config guards (S9, round-2 remediation).

The round-1 DoD asked for an integration test proving each config guard returns
503 with no tunnel and no instance created; only unit-level coverage (mocked
session_factory, mocked GpuSessionRepository) landed. This proves the same
invariant against a real Postgres schema: after each guard fires, zero
gpu_sessions rows exist for the attempting user, and neither vastai_client nor
cf_client was ever called — the guards run at step 3.6, strictly before step 4
(tunnel creation) and step 6 (instance creation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.services.gpu_session import GpuSessionService, NullNodeCooldownStore
from src.api.services.gpu_session.exceptions import ProvisioningUnavailableError
from src.core.bundle_config import BundleMapping, HardwareRequirements
from src.core.enums import ModelType
from src.db.models.gpu_session import GpuSession

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

pytestmark = pytest.mark.asyncio


class _GuardSettings:
    """The narrow Settings surface start_session's config guards touch."""

    github_content_token = "test-github-token"
    provisioning_script_ref = "v1.0.0"
    apex_callback_url = "https://apex.example.test"
    gpu_session_tokens_per_minute = 100


def _bundle_mapping() -> BundleMapping:
    return BundleMapping(
        bundle_name="wan_2.2_i2v",
        bundle_version="260105-01",
        hardware=HardwareRequirements(
            gpu_whitelist=("RTX_4090",),
            min_disk_gb=100,
            min_network_upload_mbps=100,
            min_network_download_mbps=500,
            cuda_min_version="12.1",
            num_gpus=1,
            comfyui_port=18188,
        ),
        readiness_marker=None,
    )


@pytest.fixture
def session_factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


def _make_service(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    github_content_token: str = "test-github-token",
    provisioning_script_ref: str = "v1.0.0",
    apex_callback_url: str = "https://apex.example.test",
) -> tuple[GpuSessionService, MagicMock, MagicMock]:
    settings = _GuardSettings()
    settings.github_content_token = github_content_token  # type: ignore[misc]
    settings.provisioning_script_ref = provisioning_script_ref  # type: ignore[misc]
    settings.apex_callback_url = apex_callback_url  # type: ignore[misc]

    vastai_client = AsyncMock()
    cf_client = AsyncMock()
    bundle_index = MagicMock()
    bundle_index.resolve_bundle.return_value = _bundle_mapping()
    billing_service = AsyncMock()
    billing_service.assert_sufficient_balance = AsyncMock(return_value=None)

    service = GpuSessionService(
        vastai_client=vastai_client,
        cf_client=cf_client,
        bundle_index=bundle_index,
        session_factory=session_factory,
        settings=settings,  # type: ignore[arg-type]
        billing_service=billing_service,
        cooldown_store=NullNodeCooldownStore(),
        provisioning_script_service=AsyncMock(),
    )
    return service, vastai_client, cf_client


async def _assert_no_session_row_for_user(
    session_factory: async_sessionmaker[AsyncSession], user_id: object
) -> None:
    async with session_factory() as db:
        result = await db.execute(select(GpuSession).where(GpuSession.user_id == user_id))
        assert result.scalars().first() is None


class TestStartSessionConfigGuardsAgainstRealPostgres:
    async def test_empty_github_content_token_returns_503_with_no_side_effects(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        service, vastai_client, cf_client = _make_service(session_factory, github_content_token="")
        user_id = uuid4()

        with pytest.raises(ProvisioningUnavailableError, match="github_content_token"):
            await service.start_session(
                user_id=user_id,
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=uuid4(),
            )

        cf_client.create_session_tunnel.assert_not_awaited()
        vastai_client.search_offers.assert_not_awaited()
        vastai_client.create_instance.assert_not_awaited()
        await _assert_no_session_row_for_user(session_factory, user_id)

    async def test_empty_provisioning_script_ref_returns_503_with_no_side_effects(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        service, vastai_client, cf_client = _make_service(
            session_factory, provisioning_script_ref=""
        )
        user_id = uuid4()

        with pytest.raises(ProvisioningUnavailableError, match="provisioning_script_ref"):
            await service.start_session(
                user_id=user_id,
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=uuid4(),
            )

        cf_client.create_session_tunnel.assert_not_awaited()
        vastai_client.search_offers.assert_not_awaited()
        vastai_client.create_instance.assert_not_awaited()
        await _assert_no_session_row_for_user(session_factory, user_id)

    async def test_invalid_apex_callback_url_returns_503_with_no_side_effects(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        service, vastai_client, cf_client = _make_service(
            session_factory, apex_callback_url="https://apex.example.test/callback"
        )
        user_id = uuid4()

        with pytest.raises(ProvisioningUnavailableError, match="apex_callback_url"):
            await service.start_session(
                user_id=user_id,
                product_id="vex",
                model_type=ModelType.AISHA_IMAGE,
                account_id=uuid4(),
            )

        cf_client.create_session_tunnel.assert_not_awaited()
        vastai_client.search_offers.assert_not_awaited()
        vastai_client.create_instance.assert_not_awaited()
        await _assert_no_session_row_for_user(session_factory, user_id)
