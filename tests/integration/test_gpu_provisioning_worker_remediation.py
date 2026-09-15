"""Integration coverage for provisioning-worker remediation paths."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.schemas.events import EventType
from src.api.schemas.provisioning import ProvisionerFailureWebhookBody
from src.api.services.gpu_session.node_cooldown import NullNodeCooldownStore
from src.api.services.gpu_session.provisioning_worker import (
    _REASON_PENDING_TIMEOUT,
    GpuProvisioningWorker,
)
from src.api.services.gpu_session.service import GpuSessionService
from src.api.services.provisioning_script import ProvisioningScriptService, ResolvedScript
from src.api.services.provisioning_webhook import ProvisioningWebhookService
from src.api.services.vastai.schemas import VastAIOffer
from src.core.bundle_config import BundleMapping, HardwareRequirements
from src.core.enums import (
    DeploymentStatus,
    GpuSessionStatus,
    OperationKind,
    OperationStatus,
    ScriptServeOutcome,
)
from src.core.uid import new_id
from src.db.models.billing import TokenAccount
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User
from src.db.repositories.gpu_session import GpuSessionRepository
from src.db.repositories.gpu_session_command import GpuSessionCommandRepository
from src.db.repositories.gpu_session_deployment import GpuSessionDeploymentRepository
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository

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
    github_content_token = "test-github-token"
    ai_bundles_repo_url = "https://example.test/ai-bundles.git"
    ai_bundles_branch = "main"
    aisha_repo_url = "https://example.test/aisha.git"
    aisha_branch = "main"
    apex_callback_url = "https://apex.example.test"
    provisioning_script_ref = "v1.0.0"
    hf_token = "test-hf-token"
    civitai_api_token = "test-civitai-token"
    aisha_comfyui_host = "0.0.0.0"  # noqa: S104
    aisha_comfyui_extra_args = ""
    vastai_destroy_retry_attempts = 3


def _make_provisioning_script_service() -> AsyncMock:
    service = AsyncMock()
    service.resolve.return_value = ResolvedScript(
        content="#!/bin/sh\necho hi\n", sha256="a" * 64, cache_hit=True
    )
    return service


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
        provisioning_script_service=_make_provisioning_script_service(),
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


async def test_provisioning_failure_cascade_bumps_revision_and_emits_operation_update(
    provisioning_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A provisioning failure publishes every command operation it cascade-closes.

    ``publish_operation_event`` runs after the provisioning-failure transaction commits.
    """
    user = User(
        id=new_id(),
        email=f"retry-remediation-operation-event-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="retry-bundle",
        model_type="aisha-image",
    )
    operation_id = new_id()
    command_id = new_id()
    async with provisioning_session_factory() as db, db.begin():
        db.add(user)
        await db.flush()
        db.add(gpu_session)
        await db.flush()
        operation = await GpuSessionOperationRepository(db).create(
            id=operation_id,
            session_id=gpu_session.id,
            user_id=user.id,
            product_id=gpu_session.product_id,
            kind=OperationKind.bundle_provision,
            command_id=command_id,
        )
        await GpuSessionCommandRepository(db).create(
            id=command_id,
            session_id=gpu_session.id,
            product_id=gpu_session.product_id,
            operation_id=operation.id,
            kind=OperationKind.bundle_provision,
            payload={},
        )

    event_bus = AsyncMock()
    worker = GpuProvisioningWorker(
        session_factory=provisioning_session_factory,
        vastai_client=AsyncMock(),
        cf_client=AsyncMock(),
        bundle_index=MagicMock(),
        http_client=AsyncMock(),
        settings=_RetrySettings(),  # type: ignore[arg-type]
        cooldown_store=NullNodeCooldownStore(),
        provisioning_script_service=_make_provisioning_script_service(),
        event_bus=event_bus,
        redis_enabled=False,
        redis_client_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )

    await worker._transition(
        gpu_session,
        new_status=GpuSessionStatus.failed,
        log_event="gpu_session.provision.test_failed",
        error_message="provisioning failed",
    )

    async with provisioning_session_factory() as db:
        operation = await GpuSessionOperationRepository(db).get(operation_id)
    assert operation is not None
    assert operation.status == OperationStatus.failed
    assert operation.revision == 1
    operation_frames = [
        call.kwargs["payload"]
        for call in event_bus.publish.call_args_list
        if call.kwargs["event_type"] == EventType.GPU_SESSION_OPERATION_UPDATED
    ]
    assert len(operation_frames) == 1
    assert operation_frames[0].id == operation_id
    assert operation_frames[0].revision == 1


class _WorkerFailureAdapter:
    """Exercise the webhook's handoff against the worker's real locked transition."""

    def __init__(self, worker: GpuProvisioningWorker, session: GpuSession) -> None:
        self._worker = worker
        self._session = session

    async def fail_pre_active_session(
        self,
        session_id: object,
        *,
        reason: str,
        expected_callback_token: str | None = None,  # noqa: ARG002 — matches real signature
    ) -> None:
        assert session_id == self._session.id
        await self._worker._mark_failed(self._session, reason)


def _webhook_payload() -> ProvisionerFailureWebhookBody:
    return ProvisionerFailureWebhookBody(
        action="continue",
        manifest="/node/manifest.yaml",
        error="failed to fetch bootstrap script",
        container_id="98765",
        timestamp="2026-09-14T12:00:00",
    )


async def test_webhook_and_probe_failure_race_tears_down_and_refunds_once(
    provisioning_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Postgres row locking makes external teardown terminal-once across both paths."""
    token = "callback-token"
    user = User(
        id=new_id(),
        email=f"retry-remediation-race-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="retry-bundle",
        model_type="aisha-image",
        vastai_instance_id=98765,
        cf_tunnel_id="tunnel-id",
        cf_dns_record_id="dns-id",
        callback_token_hash=hashlib.sha256(token.encode()).hexdigest(),
    )
    async with provisioning_session_factory() as db, db.begin():
        db.add(user)
        await db.flush()
        db.add(gpu_session)

    vastai = AsyncMock()
    cloudflare = AsyncMock()
    billing = AsyncMock()
    worker = GpuProvisioningWorker(
        session_factory=provisioning_session_factory,
        vastai_client=vastai,
        cf_client=cloudflare,
        bundle_index=MagicMock(),
        http_client=AsyncMock(),
        settings=_RetrySettings(),  # type: ignore[arg-type]
        cooldown_store=NullNodeCooldownStore(),
        provisioning_script_service=_make_provisioning_script_service(),
        billing_service=billing,
        redis_enabled=False,
        redis_client_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )
    webhook = ProvisioningWebhookService(
        gpu_session_service=_WorkerFailureAdapter(worker, gpu_session),  # type: ignore[arg-type]
        session_factory=provisioning_session_factory,
    )

    statuses = await asyncio.gather(
        webhook.handle_failure(session_id=gpu_session.id, token=token, payload=_webhook_payload()),
        worker._mark_failed(gpu_session, reason="probe_fail_fast"),
    )

    assert statuses[0] == 200
    vastai.destroy_instance.assert_awaited_once_with(98765)
    cloudflare.delete_session_tunnel.assert_awaited_once_with("tunnel-id", "dns-id")
    billing.refund.assert_awaited_once()
    async with provisioning_session_factory() as db:
        failed = await db.get(GpuSession, gpu_session.id)
    assert failed is not None
    assert failed.status == GpuSessionStatus.failed


async def test_webhook_persists_only_the_fixed_reason(
    provisioning_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    token = "callback-token"
    leaked_token = "must-not-persist"
    user = User(
        id=new_id(),
        email=f"retry-remediation-webhook-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="retry-bundle",
        model_type="aisha-image",
        callback_token_hash=hashlib.sha256(token.encode()).hexdigest(),
    )
    async with provisioning_session_factory() as db, db.begin():
        db.add(user)
        await db.flush()
        db.add(gpu_session)

    worker = GpuProvisioningWorker(
        session_factory=provisioning_session_factory,
        vastai_client=AsyncMock(),
        cf_client=AsyncMock(),
        bundle_index=MagicMock(),
        http_client=AsyncMock(),
        settings=_RetrySettings(),  # type: ignore[arg-type]
        cooldown_store=NullNodeCooldownStore(),
        provisioning_script_service=_make_provisioning_script_service(),
        redis_enabled=False,
        redis_client_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )
    webhook = ProvisioningWebhookService(
        gpu_session_service=_WorkerFailureAdapter(worker, gpu_session),  # type: ignore[arg-type]
        session_factory=provisioning_session_factory,
    )
    payload = _webhook_payload()
    payload = ProvisionerFailureWebhookBody(
        action=payload.action,
        manifest=f"/node/{leaked_token}/manifest.yaml",
        error=f"fetch https://apex.test/script?token={leaked_token}",
        container_id=payload.container_id,
        timestamp=payload.timestamp,
    )

    assert (
        await webhook.handle_failure(session_id=gpu_session.id, token=token, payload=payload) == 200
    )
    async with provisioning_session_factory() as db:
        failed = await db.get(GpuSession, gpu_session.id)
    assert failed is not None
    assert failed.error_message == "node_provision_script_failed"
    assert leaked_token not in failed.error_message


async def test_script_service_authorizes_real_session_rows(
    provisioning_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The script endpoint's service layer authenticates against committed Postgres rows."""
    token = "callback-token"
    user = User(
        id=new_id(),
        email=f"retry-remediation-script-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="retry-bundle",
        model_type="aisha-image",
        callback_token_hash=hashlib.sha256(token.encode()).hexdigest(),
    )
    async with provisioning_session_factory() as db, db.begin():
        db.add(user)
        await db.flush()
        db.add(gpu_session)

    http = AsyncMock()
    response = MagicMock()
    response.status_code = 200
    response.content = b"#!/bin/sh\necho bootstrap\n"
    response.text = response.content.decode()
    response.headers = {}
    http.get.return_value = response
    script_service = ProvisioningScriptService(
        http=http,
        redis=None,
        settings=_RetrySettings(),  # type: ignore[arg-type]
    )

    async with provisioning_session_factory() as db:
        valid = await script_service.serve_for_session(
            db=db,
            session_id=gpu_session.id,
            token=token,
            variant="comfyui",
            ref="v1.0.0",
        )
        wrong = await script_service.serve_for_session(
            db=db,
            session_id=gpu_session.id,
            token="wrong-token",
            variant="comfyui",
            ref="v1.0.0",
        )
        unknown = await script_service.serve_for_session(
            db=db,
            session_id=new_id(),
            token=token,
            variant="comfyui",
            ref="v1.0.0",
        )

    assert valid.outcome == ScriptServeOutcome.ok
    assert wrong.outcome == ScriptServeOutcome.unauthorized
    assert unknown.outcome == ScriptServeOutcome.unauthorized


async def test_replacement_node_atomically_resets_contract_failure_grace(
    provisioning_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    user = User(
        id=new_id(),
        email=f"retry-remediation-grace-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.pending,
        bundle_name="retry-bundle",
        model_type="aisha-image",
        consecutive_contract_failures=2,
    )
    async with provisioning_session_factory() as db, db.begin():
        db.add(user)
        await db.flush()
        db.add(gpu_session)

    fresh_hash = hashlib.sha256(b"replacement-token").hexdigest()
    async with provisioning_session_factory() as db, db.begin():
        repo = GpuSessionRepository(db)
        await repo.update_instance(
            gpu_session.id,
            vastai_instance_id=777,
            vastai_offer_id=42,
            vastai_cost_per_hour_micros=500_000,
            vastai_gpu_name="RTX_4090",
            vastai_machine_id=99,
            provisioning_started_at=datetime.now(UTC),
            callback_token_hash=fresh_hash,
        )

    async with provisioning_session_factory() as db, db.begin():
        repo = GpuSessionRepository(db)
        replacement = await repo.get_by_id(gpu_session.id)
        assert replacement is not None
        assert replacement.vastai_instance_id == 777
        assert replacement.callback_token_hash == fresh_hash
        assert replacement.consecutive_contract_failures == 0
        assert await repo.increment_consecutive_contract_failures(gpu_session.id) == 1


# ---------------------------------------------------------------------------
# S2: fail_pre_active_session's two-stage token check
# ---------------------------------------------------------------------------


async def _seed_personal_account(
    provisioning_session_factory: async_sessionmaker[AsyncSession], user: User
) -> TokenAccount:
    """gpu_sessions.account_id carries a real FK to token_accounts — insert one."""
    account = TokenAccount(id=new_id(), account_type="personal", user_id=user.id, product_id="vex")
    async with provisioning_session_factory() as db, db.begin():
        db.add(account)
    return account


def _make_gpu_session_service(
    provisioning_session_factory: async_sessionmaker[AsyncSession],
    *,
    vastai: AsyncMock,
    cloudflare: AsyncMock,
    billing: AsyncMock,
) -> GpuSessionService:
    return GpuSessionService(
        vastai_client=vastai,
        cf_client=cloudflare,
        bundle_index=MagicMock(),
        session_factory=provisioning_session_factory,
        settings=_RetrySettings(),  # type: ignore[arg-type]
        billing_service=billing,
        cooldown_store=NullNodeCooldownStore(),
        provisioning_script_service=_make_provisioning_script_service(),
    )


async def test_fail_pre_active_session_rejects_a_rotated_token_under_the_lock(
    provisioning_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """S2: a delayed webhook call validated against a token the row no longer
    carries must not tear down or refund the session — the row may since be a
    concurrent retry's *replacement* node, not the one that actually failed."""
    old_token = "old-callback-token"
    new_token = "new-callback-token-after-retry"
    user = User(
        id=new_id(),
        email=f"retry-remediation-s2-rotated-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    async with provisioning_session_factory() as db, db.begin():
        db.add(user)
    account = await _seed_personal_account(provisioning_session_factory, user)
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="retry-bundle",
        model_type="aisha-image",
        vastai_instance_id=11111,
        cf_tunnel_id="tunnel-id-rotated",
        cf_dns_record_id="dns-id-rotated",
        callback_token_hash=hashlib.sha256(old_token.encode()).hexdigest(),
        account_id=account.id,
    )
    async with provisioning_session_factory() as db, db.begin():
        db.add(gpu_session)

    # Simulate a concurrent _retry_with_new_node committing a replacement
    # instance + fresh callback token hash between the webhook's pre-lock
    # (detached) check and this locked call.
    async with provisioning_session_factory() as db, db.begin():
        await GpuSessionRepository(db).update_callback_token_hash(
            gpu_session.id, hashlib.sha256(new_token.encode()).hexdigest()
        )

    vastai, cloudflare, billing = AsyncMock(), AsyncMock(), AsyncMock()
    service = _make_gpu_session_service(
        provisioning_session_factory, vastai=vastai, cloudflare=cloudflare, billing=billing
    )

    result = await service.fail_pre_active_session(
        gpu_session.id,
        reason="node_provision_script_failed",
        expected_callback_token=old_token,
    )

    assert result is None
    vastai.destroy_instance.assert_not_awaited()
    cloudflare.delete_session_tunnel.assert_not_awaited()
    billing.refund.assert_not_awaited()
    async with provisioning_session_factory() as db:
        unchanged = await db.get(GpuSession, gpu_session.id)
    assert unchanged is not None
    assert unchanged.status == GpuSessionStatus.provisioning


class _RotateThenDelegate:
    """Simulates a concurrent retry rotating the hash between the webhook's own
    pre-lock (detached) check and the locked check inside fail_pre_active_session —
    the exact race window S2 closes. Wraps a real GpuSessionService."""

    def __init__(
        self,
        service: GpuSessionService,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        session_id: UUID,
        new_token_hash: str,
    ) -> None:
        self._service = service
        self._session_factory = session_factory
        self._session_id = session_id
        self._new_token_hash = new_token_hash

    async def fail_pre_active_session(
        self, session_id: UUID, *, reason: str, expected_callback_token: str | None = None
    ) -> GpuSession | None:
        assert session_id == self._session_id
        async with self._session_factory() as db, db.begin():
            await GpuSessionRepository(db).update_callback_token_hash(
                self._session_id, self._new_token_hash
            )
        return await self._service.fail_pre_active_session(
            session_id, reason=reason, expected_callback_token=expected_callback_token
        )


async def test_webhook_returns_200_not_401_when_a_rotation_races_its_locked_check(
    provisioning_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """End-to-end through the webhook: its own pre-lock check passes against the
    still-current token, a concurrent retry rotates the hash before the locked
    check runs, and the webhook must still answer 200 — the token was valid when
    presented; the provisioner must not retry."""
    old_token = "old-callback-token-e2e"
    new_token_hash = hashlib.sha256(b"new-callback-token-e2e").hexdigest()
    user = User(
        id=new_id(),
        email=f"retry-remediation-s2-e2e-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    async with provisioning_session_factory() as db, db.begin():
        db.add(user)
    account = await _seed_personal_account(provisioning_session_factory, user)
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="retry-bundle",
        model_type="aisha-image",
        vastai_instance_id=44444,
        cf_tunnel_id="tunnel-id-e2e",
        cf_dns_record_id="dns-id-e2e",
        callback_token_hash=hashlib.sha256(old_token.encode()).hexdigest(),
        account_id=account.id,
    )
    async with provisioning_session_factory() as db, db.begin():
        db.add(gpu_session)

    vastai, cloudflare, billing = AsyncMock(), AsyncMock(), AsyncMock()
    real_service = _make_gpu_session_service(
        provisioning_session_factory, vastai=vastai, cloudflare=cloudflare, billing=billing
    )
    rotating_service = _RotateThenDelegate(
        real_service,
        session_factory=provisioning_session_factory,
        session_id=gpu_session.id,
        new_token_hash=new_token_hash,
    )
    webhook = ProvisioningWebhookService(
        gpu_session_service=rotating_service,  # type: ignore[arg-type]
        session_factory=provisioning_session_factory,
    )

    status = await webhook.handle_failure(
        session_id=gpu_session.id, token=old_token, payload=_webhook_payload()
    )

    assert status == 200
    vastai.destroy_instance.assert_not_awaited()
    cloudflare.delete_session_tunnel.assert_not_awaited()
    billing.refund.assert_not_awaited()
    async with provisioning_session_factory() as db:
        unchanged = await db.get(GpuSession, gpu_session.id)
    assert unchanged is not None
    assert unchanged.status == GpuSessionStatus.provisioning
    assert unchanged.callback_token_hash == new_token_hash


async def test_fail_pre_active_session_with_current_token_still_fails_and_refunds_once(
    provisioning_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The happy path through the new check: a token that still matches the
    locked row authorizes teardown + refund exactly as before."""
    token = "current-callback-token"
    user = User(
        id=new_id(),
        email=f"retry-remediation-s2-current-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    async with provisioning_session_factory() as db, db.begin():
        db.add(user)
    account = await _seed_personal_account(provisioning_session_factory, user)
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="retry-bundle",
        model_type="aisha-image",
        vastai_instance_id=22222,
        cf_tunnel_id="tunnel-id-current",
        cf_dns_record_id="dns-id-current",
        callback_token_hash=hashlib.sha256(token.encode()).hexdigest(),
        account_id=account.id,
    )
    async with provisioning_session_factory() as db, db.begin():
        db.add(gpu_session)

    vastai, cloudflare, billing = AsyncMock(), AsyncMock(), AsyncMock()
    service = _make_gpu_session_service(
        provisioning_session_factory, vastai=vastai, cloudflare=cloudflare, billing=billing
    )

    result = await service.fail_pre_active_session(
        gpu_session.id,
        reason="node_provision_script_failed",
        expected_callback_token=token,
    )

    assert result is not None
    assert result.status == GpuSessionStatus.failed
    vastai.destroy_instance.assert_awaited_once_with(22222)
    cloudflare.delete_session_tunnel.assert_awaited_once_with("tunnel-id-current", "dns-id-current")
    billing.refund.assert_awaited_once()


async def test_fail_pre_active_session_with_no_expected_token_is_unaffected(
    provisioning_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """expected_callback_token=None (the worker's own future callers, which hold
    no token) must skip the new check entirely — behaviour is unchanged."""
    user = User(
        id=new_id(),
        email=f"retry-remediation-s2-no-token-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    async with provisioning_session_factory() as db, db.begin():
        db.add(user)
    account = await _seed_personal_account(provisioning_session_factory, user)
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="retry-bundle",
        model_type="aisha-image",
        vastai_instance_id=33333,
        cf_tunnel_id="tunnel-id-none",
        cf_dns_record_id="dns-id-none",
        callback_token_hash=hashlib.sha256(b"irrelevant").hexdigest(),
        account_id=account.id,
    )
    async with provisioning_session_factory() as db, db.begin():
        db.add(gpu_session)

    vastai, cloudflare, billing = AsyncMock(), AsyncMock(), AsyncMock()
    service = _make_gpu_session_service(
        provisioning_session_factory, vastai=vastai, cloudflare=cloudflare, billing=billing
    )

    result = await service.fail_pre_active_session(gpu_session.id, reason="probe_fail_fast")

    assert result is not None
    assert result.status == GpuSessionStatus.failed
    vastai.destroy_instance.assert_awaited_once_with(33333)
    billing.refund.assert_awaited_once()
