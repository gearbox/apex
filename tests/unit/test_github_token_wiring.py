"""Composition root: ONE GitHub credential provider per process reaches every consumer.

A per-consumer provider would defeat the App token cache (and the single-flight mint), and a
consumer that is not handed one fails closed or sends no credential — so the wiring itself
is worth pinning.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.api.dependencies import common as common_module
from src.api.services.github.token_provider import (
    GitHubAppTokenProvider,
    StaticGitHubTokenProvider,
)
from src.core.enums import GitHubAuthMode, WorkerMode
from tests.unit.helpers import hermetic_settings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.core.config import Settings

_JWT_SECRET = "test-secret-key-that-is-definitely-long-enough-32bytes"
_WORKER_CLASSES = (
    "TokenCleanupWorker",
    "AishaJobPoller",
    "HealthSnapshotWorker",
    "HealthSnapshotCleanupWorker",
    "GpuProvisioningWorker",
    "OrphanedTunnelCleanupWorker",
    "BillingReconcilerWorker",
    "GpuSessionCommandSweepWorker",
    "DeploymentOrchestrationWorker",
)


@pytest.fixture(autouse=True)
def _clean_github_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GITHUB_APP_ID",
        "GITHUB_APP_INSTALLATION_ID",
        "GITHUB_APP_PRIVATE_KEY_B64",
        "AI_BUNDLES_GITHUB_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


def _settings(worker_mode: WorkerMode) -> Settings:
    return hermetic_settings(
        jwt_secret_key=_JWT_SECRET,
        worker_mode=worker_mode,
        redis_url=None,
        r2_account_id="",
        xai_api_key="",
        aisha_poller_enabled=False,
        # The GPU session stack needs both Vast.ai and Cloudflare configured.
        vastai_api_key="vastai-test-key",
        aisha_cf_api_token="cf-token",
        aisha_cf_account_id="cf-account",
        aisha_cf_zone_id="cf-zone",
    )


@pytest.fixture
def gpu_stack_patches() -> Iterator[dict[str, MagicMock]]:
    """Replace the GPU stack's heavy collaborators with recording fakes."""
    db_manager = MagicMock()
    db_manager.session_factory = MagicMock()
    db_manager.close = AsyncMock()

    def lifecycle_fake() -> MagicMock:
        return MagicMock(return_value=MagicMock(start=AsyncMock(), stop=AsyncMock()))

    names = ("BundleIndexService", "GpuSessionService", "ProvisioningScriptService")
    with (
        patch("src.api.dependencies.common.init_db", return_value=db_manager),
        patch("src.api.dependencies.common.init_rate_limiter"),
        patch("src.api.dependencies.common.HealthSnapshotCleanupWorker"),
        patch("src.api.dependencies.common.WorkflowService"),
    ):
        fakes = {name: lifecycle_fake() for name in (*names, *_WORKER_CLASSES)}
        patchers = [
            patch(f"src.api.dependencies.common.{name}", fake) for name, fake in fakes.items()
        ]
        for patcher in patchers:
            patcher.start()
        try:
            yield fakes
        finally:
            for patcher in reversed(patchers):
                patcher.stop()


async def _init_and_shutdown(settings: Settings) -> object:
    with patch("src.api.dependencies.common.get_settings", return_value=settings):
        await common_module.init_services(settings)
        provider = common_module._services.github_token_provider
        token_service = common_module._services.gpu_session_github_token_service
        http_client = common_module._services.github_http_client
        await common_module.shutdown_services()
    assert token_service is not None
    assert http_client is not None
    assert http_client.is_closed  # shutdown closes the provider's client
    return provider, token_service


class TestOneProviderPerProcess:
    async def test_pat_mode_hands_the_same_provider_to_every_consumer(
        self, monkeypatch: pytest.MonkeyPatch, gpu_stack_patches: dict[str, MagicMock]
    ) -> None:
        monkeypatch.setenv("AI_BUNDLES_GITHUB_TOKEN", "ghp_wiring_test_pat")
        settings = _settings(WorkerMode.all)
        assert settings.github_auth_mode is GitHubAuthMode.pat

        provider, token_service = await _init_and_shutdown(settings)  # type: ignore[misc]

        assert isinstance(provider, StaticGitHubTokenProvider)
        bundle_kwargs = gpu_stack_patches["BundleIndexService"].call_args.kwargs
        script_kwargs = gpu_stack_patches["ProvisioningScriptService"].call_args.kwargs
        session_kwargs = gpu_stack_patches["GpuSessionService"].call_args.kwargs
        worker_kwargs = gpu_stack_patches["GpuProvisioningWorker"].call_args.kwargs
        assert bundle_kwargs["token_provider"] is provider
        assert script_kwargs["token_provider"] is provider
        assert session_kwargs["github_token_provider"] is provider
        assert worker_kwargs["github_token_provider"] is provider
        assert token_service._provider is provider

    async def test_app_mode_builds_the_app_provider(
        self, monkeypatch: pytest.MonkeyPatch, gpu_stack_patches: dict[str, MagicMock]
    ) -> None:
        pem = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        monkeypatch.setenv("GITHUB_APP_ID", "12345")
        monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "678")
        monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_B64", base64.b64encode(pem).decode())
        settings = _settings(WorkerMode.all)

        provider, token_service = await _init_and_shutdown(settings)  # type: ignore[misc]

        assert isinstance(provider, GitHubAppTokenProvider)
        for fake, kwarg in (
            ("BundleIndexService", "token_provider"),
            ("ProvisioningScriptService", "token_provider"),
            ("GpuSessionService", "github_token_provider"),
            ("GpuProvisioningWorker", "github_token_provider"),
        ):
            assert gpu_stack_patches[fake].call_args.kwargs[kwarg] is provider, fake
        assert token_service._provider is provider

    async def test_unconfigured_process_builds_no_provider_but_still_serves_503s(
        self, gpu_stack_patches: dict[str, MagicMock]
    ) -> None:
        """Mode `none`: consumers get None (fail closed) and the endpoint service still exists."""
        settings = _settings(WorkerMode.api_only)
        assert settings.github_auth_mode is GitHubAuthMode.none

        provider, token_service = await _init_and_shutdown(settings)  # type: ignore[misc]

        assert provider is None
        assert token_service._provider is None
        assert gpu_stack_patches["BundleIndexService"].call_args.kwargs["token_provider"] is None
        assert (
            gpu_stack_patches["GpuSessionService"].call_args.kwargs["github_token_provider"] is None
        )

    async def test_api_only_process_still_serves_the_token_endpoint(
        self, monkeypatch: pytest.MonkeyPatch, gpu_stack_patches: dict[str, MagicMock]
    ) -> None:
        """A node's call can land on any replica, including one that runs no workers."""
        monkeypatch.setenv("AI_BUNDLES_GITHUB_TOKEN", "ghp_wiring_test_pat")
        settings = _settings(WorkerMode.api_only)

        provider, token_service = await _init_and_shutdown(settings)  # type: ignore[misc]

        gpu_stack_patches["GpuProvisioningWorker"].assert_not_called()
        assert token_service._provider is provider
        assert (
            gpu_stack_patches["GpuSessionService"].call_args.kwargs["github_token_provider"]
            is provider
        )
