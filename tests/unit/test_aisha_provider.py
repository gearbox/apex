"""Unit tests for AishaGenerationProvider.

Focused on provider-specific behaviours: session routing, per-session client
lifecycle, and the ProviderResponseError path when the backend returns a
malformed response (missing prompt_id).

The orchestrator-level refund flow is covered by ``test_unified_generation.py``
and is not duplicated here — raising any exception triggers the same code path.
"""

from __future__ import annotations

import dataclasses
import io
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import structlog.testing
from PIL import Image

from src.api.schemas.unified_generation import SourceMediaReference, UnifiedGenerationRequest
from src.api.services.bundle_index import BundleNotFoundError
from src.api.services.generation.aisha.handlers import AishaImageGenerationHandler
from src.api.services.generation.aisha_provider import AishaGenerationProvider
from src.api.services.generation.service import FeatureNotSupportedError, ProviderResponseError
from src.api.services.generation.source_media import ResolvedSourceMedia
from src.api.services.gpu_session.exceptions import NoActiveSessionError
from src.api.services.image_normalization import ImageTooLargeError
from src.api.services.workflow.applier import apply as apply_bound_workflow
from src.core.enums import (
    AspectRatio,
    GenerationType,
    JobStatus,
    MediaKind,
    MediaSlot,
    ModelType,
    Resolution,
    Sampler,
    Scheduler,
)
from src.core.generation_config import (
    BundleGenerationConfig,
    GenerationConstraints,
    GenerationDefaults,
)
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref
from src.core.resolution import resolve_dimensions
from tests.unit.helpers import (
    QWEN_ENCODER_NODE,
    QWEN_LOAD_IMAGE_NODES,
    qwen_rapid_aio_bound_workflow,
)


def _make_request() -> UnifiedGenerationRequest:
    return UnifiedGenerationRequest(
        prompt="a cat",
        generation_type=GenerationType.T2I,
        model=ModelType.AISHA_IMAGE,
        aspect_ratio=AspectRatio.RATIO_1_1,
        n=1,
    )


def _make_active_gpu_session() -> MagicMock:
    """Stand-in for a GpuSession row in `active` status with a tunnel hostname."""
    gs = MagicMock()
    gs.id = uuid4()
    gs.tunnel_hostname = "01abcdef.gpu.test"
    gs.bundle_name = "test_bundle"
    gs.bundle_version = None
    return gs


def _routing(session: MagicMock) -> tuple[MagicMock, MagicMock]:
    """Build the (session, deployment) pair get_active_session_for_model returns.

    Since P2, generation reads model identity off the deployment, not the
    session (D18) — see handlers.py. The stand-in deployment mirrors whatever
    bundle_name/bundle_version the caller's session mock carries.
    """
    deployment = MagicMock()
    deployment.bundle_name = session.bundle_name
    deployment.bundle_version = session.bundle_version
    return session, deployment


def _make_default_gen_config() -> BundleGenerationConfig:
    return BundleGenerationConfig(
        defaults=GenerationDefaults(
            resolution=Resolution.STANDARD,
            steps=12,
            cfg=1.1,
            sampler=Sampler.EULER,
            scheduler=Scheduler.BETA,
            denoise=1.0,
        ),
        constraints=GenerationConstraints(
            max_megapixels=1.05,
            latent_multiple=16,
            max_edge=1536,
            min_steps=1,
            max_steps=30,
            min_cfg=0.0,
            max_cfg=15.0,
            allowed_samplers=frozenset(),
            allowed_schedulers=frozenset(),
        ),
    )


class TestAishaProviderValidation:
    def test_aisha_video_is_not_implemented(self) -> None:
        provider = AishaGenerationProvider(
            workflow_service=MagicMock(),
            gpu_session_service=MagicMock(),
            bundle_index=MagicMock(),
        )
        request = UnifiedGenerationRequest(
            prompt="make a video",
            generation_type=GenerationType.T2V,
            model=ModelType.AISHA_VIDEO,
        )

        with pytest.raises(FeatureNotSupportedError, match="Aisha video"):
            provider.validate(request)

    def test_source_media_validation_is_deferred_to_the_service(self) -> None:
        provider = AishaGenerationProvider(
            workflow_service=MagicMock(),
            gpu_session_service=MagicMock(),
            bundle_index=MagicMock(),
        )
        request = UnifiedGenerationRequest(
            prompt="edit multiple references",
            generation_type=GenerationType.I2I,
            model=ModelType.AISHA_IMAGE,
            source_media=[SourceMediaReference(asset_ref=f"upload:{uuid4()}")],
        )

        provider.validate(request)


def _make_bundle_index_mock() -> MagicMock:
    """A minimal bundle-index mock that returns a sentinel Path for any bundle name."""
    bi = MagicMock()
    bi.get_bundle_path = MagicMock(return_value=MagicMock())
    bi.get_generation_config = MagicMock(return_value=_make_default_gen_config())
    return bi


def _make_provider_with_mocks() -> tuple[AishaGenerationProvider, dict]:
    workflow = MagicMock()
    workflow.load.return_value = MagicMock()
    workflow.apply.return_value = {"3": {"inputs": {"text": "a cat"}}}

    gpu_session_service = AsyncMock()
    gpu_session_service.get_active_session_for_model = AsyncMock(
        return_value=_routing(_make_active_gpu_session())
    )
    gpu_session_service.is_generation_routing_suspended = AsyncMock(return_value=False)
    bundle_index = _make_bundle_index_mock()

    # tunnel_domain matches the mocked hostname suffix so the SSRF allowlist
    # check passes by default; SSRF-specific tests construct their own provider
    # with a deliberately-mismatched tunnel_domain.
    provider = AishaGenerationProvider(
        workflow_service=workflow,
        gpu_session_service=gpu_session_service,
        tunnel_domain="gpu.test",
        bundle_index=bundle_index,
    )
    return provider, {
        "workflow": workflow,
        "gpu_session_service": gpu_session_service,
        "bundle_index": bundle_index,
    }


class TestAishaProviderMissingPromptId:
    """Issue 2 regression: backend accepts the request but returns no prompt_id.

    Before the fix, the provider set ``db_job.status = FAILED`` and returned
    normally. ``GenerationService.generate`` only refunds on exception, so the
    DEBIT committed alongside the FAILED job with no refund.

    After the fix, the provider flushes the FAILED status + error_message and
    raises ``ProviderResponseError`` with an infrastructure-agnostic message.
    """

    async def test_raises_provider_response_error_when_queue_prompt_returns_empty(
        self,
    ) -> None:
        provider, _mocks = _make_provider_with_mocks()

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))

        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as MockComfyClient,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            mock_client.queue_prompt = AsyncMock(return_value={})  # empty, no prompt_id
            MockComfyClient.return_value = mock_client

            with pytest.raises(ProviderResponseError) as exc_info:
                await provider.submit(
                    _make_request(),
                    user_id=uuid4(),
                    session=session,
                    billing_service=billing,
                    account_id=uuid4(),
                    token_cost=50,
                    product_id="vex",
                )

        # Job was marked FAILED and flushed so the user sees the failure via
        # GET /v1/jobs/{id} even though the response came back as an error.
        assert db_job.status == JobStatus.FAILED
        assert db_job.error_message == "Generation backend returned an unexpected response."
        # Flush must have been called before raise so the FAILED state persists
        # when the orchestrator commits the refund on the same session.
        session.flush.assert_awaited()

        # User-facing message must NOT disclose internal infrastructure.
        msg = str(exc_info.value)
        assert "ComfyUI" not in msg
        assert "comfyui" not in msg
        assert "tunnel" not in msg.lower()
        # But it should be actionable.
        assert "refunded" in msg.lower()

    async def test_raises_when_response_missing_prompt_id_key(self) -> None:
        """A response dict lacking 'prompt_id' (even if non-empty) must raise."""
        provider, _ = _make_provider_with_mocks()

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))

        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as MockComfyClient,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            # Response is non-empty but missing 'prompt_id' — the walrus falls through
            mock_client.queue_prompt = AsyncMock(return_value={"number": 42, "node_errors": {}})
            MockComfyClient.return_value = mock_client

            with pytest.raises(ProviderResponseError):
                await provider.submit(
                    _make_request(),
                    user_id=uuid4(),
                    session=session,
                    billing_service=billing,
                    account_id=uuid4(),
                    token_cost=50,
                    product_id="vex",
                )

        assert db_job.status == JobStatus.FAILED


class TestAishaProviderRouting:
    """Regression coverage for session routing and per-session client lifecycle."""

    async def test_raises_no_active_session_when_none_found(self) -> None:
        provider, mocks = _make_provider_with_mocks()
        mocks["gpu_session_service"].get_active_session_for_model = AsyncMock(return_value=None)

        with pytest.raises(NoActiveSessionError):
            await provider.submit(
                _make_request(),
                user_id=uuid4(),
                session=AsyncMock(),
                billing_service=AsyncMock(),
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
            )

    async def test_explains_restart_suspension_when_routing_is_temporarily_closed(self) -> None:
        provider, mocks = _make_provider_with_mocks()
        mocks["gpu_session_service"].get_active_session_for_model = AsyncMock(return_value=None)
        mocks["gpu_session_service"].is_generation_routing_suspended = AsyncMock(return_value=True)

        with pytest.raises(NoActiveSessionError, match="temporarily paused while"):
            await provider.submit(
                _make_request(),
                user_id=uuid4(),
                session=AsyncMock(),
                billing_service=AsyncMock(),
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
            )

        mocks["gpu_session_service"].is_generation_routing_suspended.assert_awaited_once()

    async def test_routing_uses_deployment_bundle_identity_not_session(self) -> None:
        """Invariant 6: with the session's bundle_name deliberately set to a
        different value than the deployment's, generation must resolve the
        *deployment's* bundle — see D18."""
        provider, mocks = _make_provider_with_mocks()

        gpu_session = _make_active_gpu_session()
        gpu_session.bundle_name = "session_bundle_should_not_be_used"
        gpu_session.bundle_version = "session-version"
        deployment = MagicMock()
        deployment.bundle_name = "deployment_bundle_wins"
        deployment.bundle_version = "deployment-version"
        mocks["gpu_session_service"].get_active_session_for_model = AsyncMock(
            return_value=(gpu_session, deployment)
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as MockComfyClient,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            mock_client.queue_prompt = AsyncMock(return_value={"prompt_id": "queued-routing"})
            MockComfyClient.return_value = mock_client

            await provider.submit(
                _make_request(),
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
            )

        # gen_cfg lookup used the deployment's identity.
        mocks["bundle_index"].get_generation_config.assert_called_once_with(
            "deployment_bundle_wins", "deployment-version"
        )
        # bundle_path lookup used the deployment's identity.
        mocks["bundle_index"].get_bundle_path.assert_called_once_with("deployment_bundle_wins")
        # Workflow application received the deployment's identity, not the session's.
        apply_kwargs = mocks["workflow"].apply.call_args.kwargs
        assert apply_kwargs["bundle_name"] == "deployment_bundle_wins"
        assert apply_kwargs["bundle_version"] == "deployment-version"
        load_args = mocks["workflow"].load.call_args.args
        assert load_args[1] == "deployment-version"


class TestAishaProviderSSRFGuard:
    """Round 4 / Issue 1: tunnel_hostname must end with the configured tunnel
    domain before we make any outbound HTTP request. Defense in depth — the
    write path always produces hostnames of the form ``{id_short}.{domain}``,
    but a corrupted DB row or future code path that writes the column could
    produce something else, and we don't want to make outbound HTTPS calls
    to attacker-controlled destinations.
    """

    async def test_rejects_off_domain_hostname(self) -> None:
        """Hostname ending in attacker.com (not gpu.test) must raise."""
        workflow = MagicMock()
        workflow.load_workflow.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        # GPU session row whose hostname is OUTSIDE our allowlist. Could happen
        # via a corrupted column, bad migration, or a hypothetical future bug
        # that lets attacker-controlled input flow into the column.
        malicious_session = MagicMock()
        malicious_session.id = uuid4()
        malicious_session.tunnel_hostname = "evil.attacker.com"

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(malicious_session)
        )

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            bundle_index=_make_bundle_index_mock(),
            tunnel_domain="gpu.test",  # legitimate domain — hostname must end with .gpu.test
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))

        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo:
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)

            with pytest.raises(ProviderResponseError) as exc_info:
                await provider.submit(
                    _make_request(),
                    user_id=uuid4(),
                    session=session,
                    billing_service=billing,
                    account_id=uuid4(),
                    token_cost=50,
                    product_id="vex",
                )

        # Job stamped FAILED + flushed so the orchestrator's refund commit
        # persists the user-visible failure state.
        assert db_job.status == JobStatus.FAILED
        assert "misconfigured" in (db_job.error_message or "").lower()
        session.flush.assert_awaited()

        # User-facing message must NOT disclose internal infrastructure.
        msg = str(exc_info.value)
        assert "ComfyUI" not in msg
        assert "tunnel" not in msg.lower()
        assert "attacker.com" not in msg
        assert "gpu.test" not in msg
        # Provider error messages do NOT claim refund — refund semantics
        # are owned by the orchestrator's exception handler. The message
        # describes the failure; the orchestrator handles the money.
        assert "refund" not in msg.lower()

    async def test_rejects_empty_hostname(self) -> None:
        """A NULL/empty tunnel_hostname (race or schema corruption) must raise."""
        workflow = MagicMock()
        workflow.load_workflow.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        broken_session = MagicMock()
        broken_session.id = uuid4()
        broken_session.tunnel_hostname = None  # never populated

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(broken_session)
        )

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            bundle_index=_make_bundle_index_mock(),
            tunnel_domain="gpu.test",
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))

        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo:
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)

            with pytest.raises(ProviderResponseError):
                await provider.submit(
                    _make_request(),
                    user_id=uuid4(),
                    session=session,
                    billing_service=billing,
                    account_id=uuid4(),
                    token_cost=50,
                    product_id="vex",
                )

    async def test_rejects_suffix_match_attack(self) -> None:
        """Hostname like 'evil.com-gpu.test' (suffix attack) must be rejected.

        Regression for the classic suffix-confusion bug — naive
        ``hostname.endswith(domain)`` would pass for ``evil-gpu.test`` if
        domain were ``gpu.test``. The fix uses ``endswith("." + domain)``
        which requires the dot separator.
        """
        workflow = MagicMock()
        workflow.load_workflow.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        sneaky_session = MagicMock()
        sneaky_session.id = uuid4()
        # NOT a subdomain of gpu.test — it's a separate domain that just ends
        # in the literal characters "gpu.test" without the dot separator.
        sneaky_session.tunnel_hostname = "evil-gpu.test"

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(sneaky_session)
        )

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            bundle_index=_make_bundle_index_mock(),
            tunnel_domain="gpu.test",
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))

        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo:
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)

            with pytest.raises(ProviderResponseError):
                await provider.submit(
                    _make_request(),
                    user_id=uuid4(),
                    session=session,
                    billing_service=billing,
                    account_id=uuid4(),
                    token_cost=50,
                    product_id="vex",
                )

    @pytest.mark.parametrize(
        "hostname",
        [
            # URL-meta characters that would let an ``endswith`` check be
            # smuggled past while urllib parses a different host. These
            # all end in ``.gpu.test`` literally but contain forbidden
            # chars that the charset check rejects.
            "evil.com?.gpu.test",
            "evil.com#.gpu.test",
            "evil.com@host.gpu.test",
            "evil.com:443.gpu.test",
            "/etc/passwd.gpu.test",
            "evil\\\\path.gpu.test",
            # Leading dot or hyphen: invalid DNS shape
            ".abc.gpu.test",
            "-abc.gpu.test",
            # Whitespace
            "abc def.gpu.test",
        ],
    )
    async def test_charset_check_rejects_url_meta_and_invalid_dns(self, hostname: str) -> None:
        """Defense in depth: a hostname that ``endswith(".gpu.test")`` but
        contains URL-meta characters (?, #, @, :, /, \\) or invalid DNS
        characters must still be rejected. urllib's host parser would
        otherwise interpret those characters and route the request to a
        different host than the suffix suggests.
        """
        workflow = MagicMock()
        workflow.load_workflow.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        sneaky_session = MagicMock()
        sneaky_session.id = uuid4()
        sneaky_session.tunnel_hostname = hostname

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(sneaky_session)
        )

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            bundle_index=_make_bundle_index_mock(),
            tunnel_domain="gpu.test",
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo:
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)

            with pytest.raises(ProviderResponseError):
                await provider.submit(
                    _make_request(),
                    user_id=uuid4(),
                    session=session,
                    billing_service=billing,
                    account_id=uuid4(),
                    token_cost=50,
                    product_id="vex",
                )


def _make_i2i_request_with_user_image(input_image_id: UUID) -> UnifiedGenerationRequest:
    return UnifiedGenerationRequest(
        prompt="a cat in a hat",
        generation_type=GenerationType.I2I,
        model=ModelType.AISHA_IMAGE,
        aspect_ratio=AspectRatio.RATIO_1_1,
        n=1,
        source_media=[SourceMediaReference(asset_ref=f"upload:{input_image_id}")],
    )


def _make_i2i_request_with_source_output(source_output_id: UUID) -> UnifiedGenerationRequest:
    return UnifiedGenerationRequest(
        prompt="a cat in a hat",
        generation_type=GenerationType.I2I,
        model=ModelType.AISHA_IMAGE,
        aspect_ratio=AspectRatio.RATIO_1_1,
        n=1,
        source_media=[SourceMediaReference(asset_ref=f"output:{source_output_id}")],
    )


def _make_resolved_source_image(
    asset_id: UUID,
    *,
    source: LibraryAssetSource,
    storage_key: str,
) -> ResolvedSourceMedia:
    return ResolvedSourceMedia(
        position=0,
        ref=AssetRef(source=source, asset_id=asset_id),
        asset_ref=format_asset_ref(source, asset_id),
        media_kind=MediaKind.IMAGE,
        content_type="image/png",
        storage_key=storage_key,
        size_bytes=1,
        job_id=uuid4() if source is LibraryAssetSource.OUTPUT else None,
    )


def _webp_bytes(size: tuple[int, int] = (16, 12)) -> bytes:
    im = Image.new("RGB", size, (10, 20, 30))
    buf = io.BytesIO()
    im.save(buf, format="WEBP")
    return buf.getvalue()


def _png_bytes(size: tuple[int, int] = (16, 12)) -> bytes:
    im = Image.new("RGB", size, (255, 0, 0))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size: tuple[int, int] = (16, 12)) -> bytes:
    im = Image.new("RGB", size, (0, 255, 0))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()


def _jpeg_bytes_with_orientation(orientation: int, size: tuple[int, int] = (16, 12)) -> bytes:
    """JPEG whose header ``size`` differs from its EXIF-displayed size."""
    im = Image.new("RGB", size, (0, 255, 0))
    exif = im.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


class TestAishaImageHandlerFailurePaths:
    async def test_oversized_source_image_is_rejected_with_a_clear_public_error(self) -> None:
        r2 = AsyncMock()
        r2.download.return_value = _png_bytes()
        handler = AishaImageGenerationHandler(
            workflow_service=MagicMock(),
            gpu_session_service=None,
            bundle_index=MagicMock(),
            r2_storage=r2,
            tunnel_domain="gpu.test",
            tunnel_hostname_allowed_prefix=None,
            max_input_megapixels=1.0,
        )
        source = _make_resolved_source_image(
            uuid4(), source=LibraryAssetSource.UPLOAD, storage_key="users/abc/uploads/large.png"
        )

        with (
            patch(
                "src.api.services.generation.aisha.handlers.ensure_comfyui_input",
                new=AsyncMock(side_effect=ImageTooLargeError(megapixels=2.0, limit=1.0)),
            ),
            pytest.raises(ValueError, match="exceeds maximum pixel count"),
        ):
            await handler._resolve_input_images([source])

    async def test_submit_raises_when_gpu_sessions_are_not_configured(self) -> None:
        handler = AishaImageGenerationHandler(
            workflow_service=MagicMock(),
            gpu_session_service=None,
            bundle_index=MagicMock(),
            r2_storage=None,
            tunnel_domain="gpu.test",
            tunnel_hostname_allowed_prefix=None,
            max_input_megapixels=100.0,
        )

        with pytest.raises(NoActiveSessionError, match="not configured on this server"):
            await handler.submit(
                _make_request(),
                user_id=uuid4(),
                session=AsyncMock(),
                billing_service=AsyncMock(),
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
            )

    async def test_prepare_inputs_resolves_the_source_itself_when_not_preresolved(self) -> None:
        """submit() always pre-resolves and passes ``resolved_input``; a direct
        caller that omits it still gets a correctly resolved upload."""
        r2 = AsyncMock()
        r2.download = AsyncMock(return_value=_png_bytes())
        handler = AishaImageGenerationHandler(
            workflow_service=MagicMock(),
            gpu_session_service=None,
            bundle_index=MagicMock(),
            r2_storage=r2,
            tunnel_domain="gpu.test",
            tunnel_hostname_allowed_prefix=None,
            max_input_megapixels=100.0,
        )
        source = _make_resolved_source_image(
            uuid4(), source=LibraryAssetSource.UPLOAD, storage_key="users/abc/uploads/cat.png"
        )
        client = AsyncMock()
        client.upload_image = AsyncMock(return_value={"name": "stored.png"})

        result = await handler.prepare_inputs(
            [source],
            client=client,
            job_id=uuid4(),
            gpu_session_id=uuid4(),
        )

        r2.download.assert_awaited_once_with("users/abc/uploads/cat.png")
        assert result == {MediaSlot.REFERENCE: ["stored.png"]}

    async def test_missing_indexed_bundle_becomes_a_session_error_and_closes_client(self) -> None:
        provider, mocks = _make_provider_with_mocks()
        mocks["bundle_index"].get_bundle_path.side_effect = BundleNotFoundError("not indexed")
        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as job_repository,
            patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as comfy_client,
        ):
            job_repository.return_value.create = AsyncMock(return_value=db_job)
            client = AsyncMock()
            comfy_client.return_value = client

            with pytest.raises(NoActiveSessionError, match="not found in index"):
                await provider.submit(
                    _make_request(),
                    user_id=uuid4(),
                    session=session,
                    billing_service=billing,
                    account_id=uuid4(),
                    token_cost=50,
                    product_id="vex",
                )

        client.connect.assert_awaited_once()
        client.close.assert_awaited_once()


class TestAishaProviderI2IBridge:
    """R2 ↔ ComfyUI image bridge for image-to-image generation.

    The bridge resolves the source image (user upload OR prior generation
    output), downloads bytes from R2, uploads them to ComfyUI's input
    folder via the per-session tunnel, and threads the stored filename
    into apply_parameters so the workflow's LoadImage node receives it.
    """

    async def test_i2i_with_user_image_uploads_bytes_to_comfyui(self) -> None:
        """End-to-end: input_image_id → R2 download → ComfyUI upload → workflow wiring."""
        workflow = MagicMock()
        workflow.load.return_value = MagicMock()
        workflow.apply.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(_make_active_gpu_session())
        )

        # R2 returns the source bytes for the configured user image.
        user_image_id = uuid4()
        png_bytes = _png_bytes()
        r2 = AsyncMock()
        r2.download = AsyncMock(return_value=png_bytes)

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            r2_storage=r2,
            tunnel_domain="gpu.test",
            bundle_index=_make_bundle_index_mock(),
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))

        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as MockComfyClient,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            # ComfyUI returns the actual stored filename — may differ from the
            # one we requested (collision-resolution suffix etc).
            mock_client.upload_image = AsyncMock(
                return_value={"name": "input_cat.png_deadbeef", "subfolder": "", "type": "input"}
            )
            mock_client.queue_prompt = AsyncMock(return_value={"prompt_id": "queued-123"})
            MockComfyClient.return_value = mock_client
            await provider.submit(
                _make_i2i_request_with_user_image(user_image_id),
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
                source_media=[
                    _make_resolved_source_image(
                        user_image_id,
                        source=LibraryAssetSource.UPLOAD,
                        storage_key="users/abc/uploads/cat.png",
                    )
                ],
            )

        # R2 receives the storage key resolved by the orchestrator.
        r2.download.assert_awaited_once_with("users/abc/uploads/cat.png")

        # ComfyUI received the bytes with the agreed filename template:
        # f"input_{index}_{stem}_{job_id_short}.{ext}" — extension is terminal.
        mock_client.upload_image.assert_awaited_once()
        upload_kwargs = mock_client.upload_image.await_args.kwargs
        assert upload_kwargs["image_data"] == png_bytes
        # source_filename stem in template; suffix is the 8-char job_id prefix
        # followed by the terminal extension.
        assert upload_kwargs["filename"].startswith("input_0_cat_")
        assert upload_kwargs["filename"].endswith(".png")
        # 8 hex chars between the stem and the extension.
        job_part = upload_kwargs["filename"].rsplit("_", 1)[1].split(".")[0]
        assert len(job_part) == 8

        # Workflow received the ComfyUI-stored name (which may differ from the
        # requested one) keyed to the bundle-declared reference slot.
        workflow.apply.assert_called_once()
        apply_kwargs = workflow.apply.call_args.kwargs
        assert apply_kwargs["media_filenames"] == {MediaSlot.REFERENCE: ["input_cat.png_deadbeef"]}

        # Job persisted as QUEUED with the ComfyUI prompt_id.
        assert db_job.status == JobStatus.QUEUED
        assert db_job.external_request_id == "queued-123"

    async def test_i2i_with_source_output_id_takes_precedence(self) -> None:
        """source_output_id branch: filename derived from storage_key tail."""
        workflow = MagicMock()
        workflow.load.return_value = MagicMock()
        workflow.apply.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(_make_active_gpu_session())
        )

        r2 = AsyncMock()
        r2.download = AsyncMock(return_value=_jpeg_bytes())

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            r2_storage=r2,
            tunnel_domain="gpu.test",
            bundle_index=_make_bundle_index_mock(),
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        output_row = MagicMock()
        output_row.storage_key = "outputs/job_xyz/result_001.jpg"

        # The primary output ID is recorded on the GenerationJob row, while
        # the resolved source supplies the storage key used by the bridge.
        source_id = uuid4()

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as MockComfyClient,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            mock_client.upload_image = AsyncMock(
                return_value={"name": "input_result_001.jpg_aaaaaaaa"}
            )
            mock_client.queue_prompt = AsyncMock(return_value={"prompt_id": "queued-456"})
            MockComfyClient.return_value = mock_client
            await provider.submit(
                _make_i2i_request_with_source_output(source_id),
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
                primary_output_id=source_id,
                source_media=[
                    _make_resolved_source_image(
                        source_id,
                        source=LibraryAssetSource.OUTPUT,
                        storage_key=output_row.storage_key,
                    )
                ],
            )

        # Filename derived from storage_key tail (not original_filename).
        # Extension is normalized to "jpeg" (MediaFormat.JPEG) and kept terminal.
        upload_kwargs = mock_client.upload_image.await_args.kwargs
        assert upload_kwargs["filename"].startswith("input_0_result_001_")
        assert upload_kwargs["filename"].endswith(".jpeg")

        # Workflow wired with the ComfyUI-returned name via the reference slot.
        apply_kwargs = workflow.apply.call_args.kwargs
        assert apply_kwargs["media_filenames"] == {
            MediaSlot.REFERENCE: ["input_result_001.jpg_aaaaaaaa"]
        }

    async def test_i2i_without_r2_dependency_raises_provider_response_error(self) -> None:
        """Defensive: if a deployment forgets to wire R2, I2I requests get a
        clean infrastructure-agnostic error, not a NoneType AttributeError."""
        workflow = MagicMock()
        workflow.load_workflow.return_value = {"3": {"inputs": {}}}
        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(_make_active_gpu_session())
        )

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            bundle_index=MagicMock(),
            r2_storage=None,
            tunnel_domain="gpu.test",
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo:
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)

            with pytest.raises(ProviderResponseError) as exc_info:
                await provider.submit(
                    _make_i2i_request_with_user_image(uuid4()),
                    user_id=uuid4(),
                    session=session,
                    billing_service=billing,
                    account_id=uuid4(),
                    token_cost=50,
                    product_id="vex",
                    source_media=[
                        _make_resolved_source_image(
                            uuid4(),
                            source=LibraryAssetSource.UPLOAD,
                            storage_key="uploads/input.png",
                        )
                    ],
                )

        # User-facing message stays infrastructure-agnostic.
        msg = str(exc_info.value)
        assert "ComfyUI" not in msg
        assert "R2" not in msg
        assert "tunnel" not in msg.lower()

    async def test_t2i_skips_i2i_bridge(self) -> None:
        """T2I requests don't trigger any R2 download or ComfyUI upload —
        the bridge cleanly no-ops when neither input ID is set."""
        provider, mocks = _make_provider_with_mocks()
        # Provider has no r2_storage by default in _make_provider_with_mocks;
        # this asserts the T2I path doesn't even check that field.
        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as MockComfyClient,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            mock_client.queue_prompt = AsyncMock(return_value={"prompt_id": "queued-789"})
            MockComfyClient.return_value = mock_client

            await provider.submit(
                _make_request(),  # T2I
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
            )

        # No upload to ComfyUI for T2I.
        mock_client.upload_image.assert_not_called()
        # No media slots are supplied for a text-to-image request.
        apply_kwargs = mocks["workflow"].apply.call_args.kwargs
        assert apply_kwargs["media_filenames"] == {}
        # Bound-workflow application receives the bundle name as a string.
        _gpu_session, deployment = mocks[
            "gpu_session_service"
        ].get_active_session_for_model.return_value
        assert isinstance(apply_kwargs["bundle_name"], str)
        assert apply_kwargs["bundle_name"] == deployment.bundle_name

    async def test_i2i_webp_source_output_converted_to_png_for_comfyui(self) -> None:
        """D2': a Grok-output WebP remixed via source_output_id must reach
        ComfyUI as PNG bytes with a matching terminal filename extension."""
        workflow = MagicMock()
        workflow.load.return_value = MagicMock()
        workflow.apply.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(_make_active_gpu_session())
        )

        webp_bytes = _webp_bytes()
        r2 = AsyncMock()
        r2.download = AsyncMock(return_value=webp_bytes)

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            r2_storage=r2,
            tunnel_domain="gpu.test",
            bundle_index=_make_bundle_index_mock(),
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        output_row = MagicMock()
        output_row.storage_key = "outputs/job_xyz/result_001.webp"
        source_id = uuid4()

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as MockComfyClient,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            mock_client.upload_image = AsyncMock(return_value={"name": "stored.png"})
            mock_client.queue_prompt = AsyncMock(return_value={"prompt_id": "queued-webp"})
            MockComfyClient.return_value = mock_client
            await provider.submit(
                _make_i2i_request_with_source_output(source_id),
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
                primary_output_id=source_id,
                source_media=[
                    _make_resolved_source_image(
                        source_id,
                        source=LibraryAssetSource.OUTPUT,
                        storage_key=output_row.storage_key,
                    )
                ],
            )

        upload_kwargs = mock_client.upload_image.await_args.kwargs
        # ComfyUI received PNG-sniffing bytes, not the original WebP.
        assert upload_kwargs["image_data"][:8] == b"\x89PNG\r\n\x1a\n"
        assert webp_bytes[:4] == b"RIFF"  # sanity: source really was WebP
        # Filename carries a terminal .png extension.
        assert re.match(r"^input_0_.+_[0-9a-f]{8}\.png$", upload_kwargs["filename"])

    async def test_i2i_comfyui_filename_has_terminal_extension(self) -> None:
        """Regression for the ``input_photo.png_1a2b3c4d`` bug: the job-id
        suffix must not swallow the extension — it must stay terminal."""
        workflow = MagicMock()
        workflow.load_workflow_from_bundle.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.inject_checkpoint = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(_make_active_gpu_session())
        )

        r2 = AsyncMock()
        r2.download = AsyncMock(return_value=_png_bytes())

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            r2_storage=r2,
            tunnel_domain="gpu.test",
            bundle_index=_make_bundle_index_mock(),
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as MockComfyClient,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            mock_client.upload_image = AsyncMock(return_value={"name": "stored.png"})
            mock_client.queue_prompt = AsyncMock(return_value={"prompt_id": "queued-ext"})
            MockComfyClient.return_value = mock_client
            await provider.submit(
                _make_i2i_request_with_user_image(uuid4()),
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
                source_media=[
                    _make_resolved_source_image(
                        uuid4(),
                        source=LibraryAssetSource.UPLOAD,
                        storage_key="users/abc/uploads/photo.png",
                    )
                ],
            )

        upload_kwargs = mock_client.upload_image.await_args.kwargs
        # Old bug: "input_photo.png_1a2b3c4d" — extension swallowed mid-name,
        # no terminal dot. The fix keeps the extension terminal.
        assert re.search(r"\.(png|jpeg)$", upload_kwargs["filename"])

    async def test_i2i_undecodable_source_raises_before_billing(self) -> None:
        """Garbage source bytes fail the ComfyUI bridge decode and raise
        ValueError before any billing reservation is made."""
        workflow = MagicMock()
        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(_make_active_gpu_session())
        )

        r2 = AsyncMock()
        r2.download = AsyncMock(return_value=b"this is not an image, just text")

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            r2_storage=r2,
            tunnel_domain="gpu.test",
            bundle_index=_make_bundle_index_mock(),
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
            pytest.raises(ValueError, match="not decodable"),
        ):
            await provider.submit(
                _make_i2i_request_with_user_image(uuid4()),
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
                source_media=[
                    _make_resolved_source_image(
                        uuid4(),
                        source=LibraryAssetSource.UPLOAD,
                        storage_key="users/abc/uploads/broken.png",
                    )
                ],
            )

        billing.check_and_reserve.assert_not_called()
        MockJobRepo.return_value.create.assert_not_called()


class TestAishaProviderI2IAspectDerivation:
    """C5: when aspect_ratio is omitted on i2i, the output canvas follows the
    source image's exact aspect instead of always defaulting to 1:1."""

    async def _submit_with_source_image(
        self,
        *,
        source_bytes: bytes,
        aspect_ratio: AspectRatio | None,
    ) -> MagicMock:
        """Runs provider.submit() with a real source image and returns the
        MagicMock workflow so callers can inspect apply_parameters' request."""
        workflow = MagicMock()
        workflow.load_workflow_from_bundle.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.inject_checkpoint = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(_make_active_gpu_session())
        )

        r2 = AsyncMock()
        r2.download = AsyncMock(return_value=source_bytes)

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            r2_storage=r2,
            tunnel_domain="gpu.test",
            bundle_index=_make_bundle_index_mock(),
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        request = UnifiedGenerationRequest(
            prompt="a cat in a hat",
            generation_type=GenerationType.I2I,
            model=ModelType.AISHA_IMAGE,
            aspect_ratio=aspect_ratio,
            n=1,
        )

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as MockComfyClient,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            mock_client.upload_image = AsyncMock(return_value={"name": "stored.png"})
            mock_client.queue_prompt = AsyncMock(return_value={"prompt_id": "queued-1"})
            MockComfyClient.return_value = mock_client
            await provider.submit(
                request,
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
                source_media=[
                    _make_resolved_source_image(
                        uuid4(),
                        source=LibraryAssetSource.UPLOAD,
                        storage_key="users/abc/uploads/source.png",
                    )
                ],
            )

        return workflow

    async def test_aisha_i2i_derives_canvas_from_source_dimensions_when_aspect_unset(
        self,
    ) -> None:
        """A 16:9 source with no requested aspect_ratio must resolve to a
        16:9-ish canvas, not the 1:1 t2i default."""
        workflow = await self._submit_with_source_image(
            source_bytes=_png_bytes(size=(1600, 900)),
            aspect_ratio=None,
        )

        expected = resolve_dimensions(
            aspect_ratio=(16, 9),
            max_megapixels=1.05,
            latent_multiple=16,
            max_edge=1536,
            tier=Resolution.STANDARD,
        )
        legacy_request = workflow.apply.call_args.kwargs["request"]
        assert legacy_request.width == expected.width
        assert legacy_request.height == expected.height
        # Sanity: landscape in, landscape out — not the 1:1 t2i fallback.
        assert legacy_request.width > legacy_request.height

    async def test_aisha_i2i_uses_requested_aspect_when_set(self) -> None:
        """An explicit aspect_ratio wins over the source image's own aspect."""
        workflow = await self._submit_with_source_image(
            source_bytes=_png_bytes(size=(1600, 900)),  # 16:9 source
            aspect_ratio=AspectRatio.RATIO_9_16,  # explicit portrait request
        )

        expected = resolve_dimensions(
            aspect_ratio=AspectRatio.RATIO_9_16,
            max_megapixels=1.05,
            latent_multiple=16,
            max_edge=1536,
            tier=Resolution.STANDARD,
        )
        legacy_request = workflow.apply.call_args.kwargs["request"]
        assert legacy_request.width == expected.width
        assert legacy_request.height == expected.height
        assert legacy_request.height > legacy_request.width

    async def test_aisha_i2i_derives_portrait_canvas_from_exif_rotated_source(self) -> None:
        """C-R1: a landscape-stored, EXIF-rotated-to-portrait source must derive
        a portrait canvas — the reduced fraction is (3, 4), not the raw
        header's (4, 3)."""
        workflow = await self._submit_with_source_image(
            source_bytes=_jpeg_bytes_with_orientation(6, size=(400, 300)),
            aspect_ratio=None,
        )

        expected = resolve_dimensions(
            aspect_ratio=(3, 4),
            max_megapixels=1.05,
            latent_multiple=16,
            max_edge=1536,
            tier=Resolution.STANDARD,
        )
        legacy_request = workflow.apply.call_args.kwargs["request"]
        assert legacy_request.width == expected.width
        assert legacy_request.height == expected.height
        assert legacy_request.height > legacy_request.width


class TestResolveEffectiveAspectRatio:
    """R2: _resolve_effective_aspect_ratio is unit-testable in isolation."""

    async def test_resolve_effective_aspect_ratio_explicit_wins(self) -> None:
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.I2I,
            model=ModelType.AISHA_IMAGE,
            aspect_ratio=AspectRatio.RATIO_9_16,
            n=1,
        )
        result = await AishaImageGenerationHandler._resolve_effective_aspect_ratio(
            request, [(_png_bytes(size=(1600, 900)), "source.png")]
        )
        assert result == AspectRatio.RATIO_9_16

    async def test_resolve_effective_aspect_ratio_t2i_defaults_1_1(self) -> None:
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            n=1,
        )
        result = await AishaImageGenerationHandler._resolve_effective_aspect_ratio(request, [])
        assert result is AspectRatio.RATIO_1_1

    async def test_resolve_effective_aspect_ratio_derives_source_fraction(self) -> None:
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.I2I,
            model=ModelType.AISHA_IMAGE,
            n=1,
        )
        result = await AishaImageGenerationHandler._resolve_effective_aspect_ratio(
            request, [(_png_bytes(size=(1600, 900)), "source.png")]
        )
        assert result == (16, 9)

    async def test_resolve_effective_aspect_ratio_undecodable_raises_value_error(self) -> None:
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.I2I,
            model=ModelType.AISHA_IMAGE,
            n=1,
        )
        with pytest.raises(ValueError, match="not decodable"):
            await AishaImageGenerationHandler._resolve_effective_aspect_ratio(
                request, [(b"garbage bytes, not an image", "broken.png")]
            )


class TestSanitizeFilename:
    """Defense-in-depth: source filename derives from user-supplied
    UserImage.original_filename or an R2 storage_key tail. Either could
    contain path separators or hostile characters that would land verbatim
    in ComfyUI's input/ folder. The sanitizer strips directory components
    and restricts to a conservative ASCII charset.
    """

    @pytest.mark.parametrize(
        ("raw", "expected_pattern"),
        [
            # Path traversal — basename is taken first, so deep paths reduce
            # to just the leaf.
            ("../../etc/passwd", "passwd"),
            ("../../../foo.png", "foo.png"),
            ("/abs/path/file.jpg", "file.jpg"),
            ("..\\..\\windows\\evil.exe", "evil.exe"),
            # Slashes in the leaf — basename-style strip
            ("user_uploads/cat.png", "cat.png"),
            # Hostile characters get _'d
            ("hello world.png", "hello_world.png"),
            # Basename of "rm -rf /;.png" is ";.png" (after the last /),
            # then ; is sanitized to _.
            ("rm -rf /;.png", "_.png"),
            ("name'with\"quotes.png", "name_with_quotes.png"),
            # Empty / pathological inputs fall back to a stable default
            ("", "image"),
            (None, "image"),
            ("///", "image"),
            # Trailing slashes leave an empty basename → fallback to "image".
            ("/...//", "image"),
            # Long names get capped (cap is 64)
            ("a" * 200 + ".png", "a" * 64),
        ],
    )
    def test_sanitization_results(self, raw: str | None, expected_pattern: str) -> None:
        from src.api.services.generation.aisha.handlers import _sanitize_filename

        result = _sanitize_filename(raw)
        assert result == expected_pattern, (
            f"input={raw!r}: expected {expected_pattern!r}, got {result!r}"
        )

    def test_sanitization_keeps_safe_chars_unchanged(self) -> None:
        """The conservative charset preserves alphanumerics, dot, underscore, hyphen."""
        from src.api.services.generation.aisha.handlers import _sanitize_filename

        safe = "Photo_2024-12-25.final.v3.png"
        assert _sanitize_filename(safe) == safe


class TestAishaProviderI2IDefensive:
    """Provider-level defenses against misuse and hostile input."""

    async def test_filename_with_path_components_is_sanitized(self) -> None:
        """End-to-end: a user image whose original_filename contains path
        traversal yields a sanitized ComfyUI filename. The job_id suffix
        is preserved for uniqueness.
        """
        workflow = MagicMock()
        workflow.load_workflow_from_bundle.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.inject_checkpoint = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(_make_active_gpu_session())
        )

        r2 = AsyncMock()
        # Real PNG bytes so the normalization bridge's format sniffer takes
        # the static passthrough branch (the pixel-cap gate still opens the
        # image header to read .size, so it must be Pillow-decodable).
        r2.download = AsyncMock(return_value=_png_bytes())

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            r2_storage=r2,
            tunnel_domain="gpu.test",
            bundle_index=_make_bundle_index_mock(),
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        # Storage keys are trusted server values, but sanitize defensively.
        source_key = "users/u_1/uploads/passwd; rm -rf .png"

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as MockComfyClient,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            mock_client.upload_image = AsyncMock(return_value={"name": "stored.png"})
            mock_client.queue_prompt = AsyncMock(return_value={"prompt_id": "queued-x"})
            MockComfyClient.return_value = mock_client
            await provider.submit(
                _make_i2i_request_with_user_image(uuid4()),
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
                source_media=[
                    _make_resolved_source_image(
                        uuid4(),
                        source=LibraryAssetSource.UPLOAD,
                        storage_key=source_key,
                    )
                ],
            )

        upload_kwargs = mock_client.upload_image.await_args.kwargs
        # No slashes survived.
        assert "/" not in upload_kwargs["filename"]
        # No shell metacharacters survived (semicolon, space, exclamation etc).
        assert ";" not in upload_kwargs["filename"]
        assert " " not in upload_kwargs["filename"]
        # The job_id suffix is preserved (8 hex chars before the terminal extension).
        job_part = upload_kwargs["filename"].rsplit("_", 1)[1].split(".")[0]
        assert len(job_part) == 8
        # The "input_" prefix is preserved and the extension is terminal.
        assert upload_kwargs["filename"].startswith("input_")
        assert "." in upload_kwargs["filename"].rsplit("_", 1)[-1]


# ---------------------------------------------------------------------------
# Resolution + sampler validation tests
# ---------------------------------------------------------------------------


def _make_constrained_gen_config(
    *,
    max_megapixels: float = 1.05,
    min_steps: int = 1,
    max_steps: int = 30,
    min_cfg: float = 0.0,
    max_cfg: float = 15.0,
    allowed_samplers: frozenset[Sampler] = frozenset(),
    allowed_schedulers: frozenset[Scheduler] = frozenset(),
) -> BundleGenerationConfig:
    return BundleGenerationConfig(
        defaults=GenerationDefaults(
            resolution=Resolution.STANDARD,
            steps=12,
            cfg=1.1,
            sampler=Sampler.EULER,
            scheduler=Scheduler.BETA,
            denoise=1.0,
        ),
        constraints=GenerationConstraints(
            max_megapixels=max_megapixels,
            latent_multiple=16,
            max_edge=1536,
            min_steps=min_steps,
            max_steps=max_steps,
            min_cfg=min_cfg,
            max_cfg=max_cfg,
            allowed_samplers=allowed_samplers,
            allowed_schedulers=allowed_schedulers,
        ),
    )


async def _submit_with_config(
    request: UnifiedGenerationRequest,
    gen_cfg: BundleGenerationConfig,
) -> None:
    """Helper: submit a request through a provider with the given gen config."""
    workflow = MagicMock()
    workflow.load.return_value = MagicMock()
    workflow.apply.return_value = {"3": {"inputs": {"text": "a cat"}}}

    gpu_session_service = AsyncMock()
    gpu_session_service.get_active_session_for_model = AsyncMock(
        return_value=_routing(_make_active_gpu_session())
    )

    bundle_index = MagicMock()
    bundle_index.get_bundle_path = MagicMock(return_value=MagicMock())
    bundle_index.get_generation_config = MagicMock(return_value=gen_cfg)

    provider = AishaGenerationProvider(
        workflow_service=workflow,
        gpu_session_service=gpu_session_service,
        tunnel_domain="gpu.test",
        bundle_index=bundle_index,
    )

    billing = AsyncMock()
    billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
    session = AsyncMock()
    db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

    with (
        patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
        patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as MockComfyClient,
    ):
        MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
        mock_client = AsyncMock()
        mock_client.queue_prompt = AsyncMock(return_value={"prompt_id": "abc123"})
        MockComfyClient.return_value = mock_client

        await provider.submit(
            request,
            user_id=uuid4(),
            session=session,
            billing_service=billing,
            account_id=uuid4(),
            token_cost=50,
            product_id="vex",
        )


class TestAishaProviderResolutionAndSamplerValidation:
    """Resolution clamping + sampler parameter validation in submit()."""

    async def test_defaults_from_bundle_when_request_omits_params(self) -> None:
        """When the request omits all optional params, bundle defaults are used."""
        request = _make_request()
        gen_cfg = _make_constrained_gen_config()

        # Should complete without error
        await _submit_with_config(request, gen_cfg)

    async def test_steps_out_of_range_raises_before_billing(self) -> None:
        """steps override outside [min, max] must raise ValueError before reservation."""
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            aspect_ratio=AspectRatio.RATIO_1_1,
            steps=999,  # way beyond max_steps=30
        )
        gen_cfg = _make_constrained_gen_config(max_steps=30)

        workflow = MagicMock()
        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(_make_active_gpu_session())
        )
        bundle_index = MagicMock()
        bundle_index.get_bundle_path = MagicMock(return_value=MagicMock())
        bundle_index.get_generation_config = MagicMock(return_value=gen_cfg)

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            tunnel_domain="gpu.test",
            bundle_index=bundle_index,
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))

        with pytest.raises(ValueError, match="steps"):
            await provider.submit(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
            )

        # Billing reservation must NOT have been called
        billing.check_and_reserve.assert_not_awaited()

    async def test_cfg_out_of_range_raises_before_billing(self) -> None:
        """cfg override outside model range raises before billing reservation."""
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            aspect_ratio=AspectRatio.RATIO_1_1,
            cfg=20.0,  # max_cfg=15.0
        )
        gen_cfg = _make_constrained_gen_config(max_cfg=15.0)

        bundle_index = MagicMock()
        bundle_index.get_bundle_path = MagicMock(return_value=MagicMock())
        bundle_index.get_generation_config = MagicMock(return_value=gen_cfg)

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(_make_active_gpu_session())
        )

        provider = AishaGenerationProvider(
            workflow_service=MagicMock(),
            gpu_session_service=gpu_session_service,
            tunnel_domain="gpu.test",
            bundle_index=bundle_index,
        )

        billing = AsyncMock()

        with pytest.raises(ValueError, match="cfg"):
            await provider.submit(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
            )
        billing.check_and_reserve.assert_not_awaited()

    async def test_sampler_not_in_allowed_list_raises_before_billing(self) -> None:
        """sampler override not in allowed set raises before billing reservation."""
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            aspect_ratio=AspectRatio.RATIO_1_1,
            sampler=Sampler.HEUN,  # not in allowed set
        )
        gen_cfg = _make_constrained_gen_config(
            allowed_samplers=frozenset({Sampler.EULER, Sampler.DPMPP_2M})
        )

        bundle_index = MagicMock()
        bundle_index.get_bundle_path = MagicMock(return_value=MagicMock())
        bundle_index.get_generation_config = MagicMock(return_value=gen_cfg)

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(_make_active_gpu_session())
        )

        provider = AishaGenerationProvider(
            workflow_service=MagicMock(),
            gpu_session_service=gpu_session_service,
            tunnel_domain="gpu.test",
            bundle_index=bundle_index,
        )

        billing = AsyncMock()

        with pytest.raises(ValueError, match="sampler"):
            await provider.submit(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
            )
        billing.check_and_reserve.assert_not_awaited()

    async def test_explicit_valid_overrides_are_applied_verbatim(self) -> None:
        """steps/cfg/sampler/scheduler explicitly set within range are used as-is,
        not just defaulted or rejected."""
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            aspect_ratio=AspectRatio.RATIO_1_1,
            steps=15,
            cfg=7.5,
            sampler=Sampler.EULER,
            scheduler=Scheduler.BETA,
        )
        gen_cfg = _make_constrained_gen_config(
            allowed_samplers=frozenset({Sampler.EULER}),
            allowed_schedulers=frozenset({Scheduler.BETA}),
        )

        # Should complete without error, using the explicit values verbatim.
        await _submit_with_config(request, gen_cfg)

    async def test_explicit_dims_over_mp_cap_clamped_no_error(self) -> None:
        """Explicit width+height exceeding mp cap is clamped, not rejected."""
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            aspect_ratio=AspectRatio.RATIO_1_1,
            width=2048,
            height=2048,  # 4.19 MP; model cap is 1.05
        )
        gen_cfg = _make_constrained_gen_config(max_megapixels=1.05)

        # Should not raise
        await _submit_with_config(request, gen_cfg)


class TestAishaProviderClampLogging:
    """R5: the resolution-clamped log reports the actual derived fraction used,
    not the opaque "source" placeholder — and reflects EXIF-corrected source
    dimensions (C-R1)."""

    async def test_clamp_log_reports_derived_fraction_for_source_aspect(self) -> None:
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.I2I,
            model=ModelType.AISHA_IMAGE,
            n=1,
        )
        # tier STANDARD targets 1.0 MP; a 0.5 MP model cap forces
        # dims.megapixels well below requested_mp * 0.9, so the clamp log fires.
        gen_cfg = _make_constrained_gen_config(max_megapixels=0.5)

        workflow = MagicMock()
        workflow.load.return_value = MagicMock()
        workflow.apply.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(_make_active_gpu_session())
        )

        # 400x300 JPEG with EXIF orientation 6 -> displayed 300x400 -> "3:4".
        r2 = AsyncMock()
        r2.download = AsyncMock(return_value=_jpeg_bytes_with_orientation(6, size=(400, 300)))

        bundle_index = MagicMock()
        bundle_index.get_bundle_path = MagicMock(return_value=MagicMock())
        bundle_index.get_generation_config = MagicMock(return_value=gen_cfg)

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            r2_storage=r2,
            tunnel_domain="gpu.test",
            bundle_index=bundle_index,
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as MockComfyClient,
            structlog.testing.capture_logs() as cap,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            mock_client.upload_image = AsyncMock(return_value={"name": "stored.png"})
            mock_client.queue_prompt = AsyncMock(return_value={"prompt_id": "queued-1"})
            MockComfyClient.return_value = mock_client
            await provider.submit(
                request,
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
                source_media=[
                    _make_resolved_source_image(
                        uuid4(),
                        source=LibraryAssetSource.UPLOAD,
                        storage_key="users/abc/uploads/source.jpg",
                    )
                ],
            )

        clamp_events = [e for e in cap if e["event"] == "generation.resolution.clamped"]
        assert clamp_events, "expected a generation.resolution.clamped log event"
        assert clamp_events[0]["aspect_ratio"] == "3:4"

    async def test_clamp_log_reports_aspect_ratio_value_for_explicit_aspect(self) -> None:
        """T2I with an explicit AspectRatio (not a derived source tuple) still
        logs its ``.value``, exercising the other branch of _aspect_log_value."""
        request = _make_request()  # aspect_ratio=AspectRatio.RATIO_1_1, t2i
        gen_cfg = _make_constrained_gen_config(max_megapixels=0.1)

        workflow = MagicMock()
        workflow.load.return_value = MagicMock()
        workflow.apply.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_routing(_make_active_gpu_session())
        )

        bundle_index = MagicMock()
        bundle_index.get_bundle_path = MagicMock(return_value=MagicMock())
        bundle_index.get_generation_config = MagicMock(return_value=gen_cfg)

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            tunnel_domain="gpu.test",
            bundle_index=bundle_index,
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with (
            patch("src.api.services.generation.aisha.handlers.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as MockComfyClient,
            structlog.testing.capture_logs() as cap,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            mock_client.queue_prompt = AsyncMock(return_value={"prompt_id": "queued-2"})
            MockComfyClient.return_value = mock_client
            await provider.submit(
                request,
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
            )

        clamp_events = [e for e in cap if e["event"] == "generation.resolution.clamped"]
        assert clamp_events, "expected a generation.resolution.clamped log event"
        assert clamp_events[0]["aspect_ratio"] == AspectRatio.RATIO_1_1.value


# ---------------------------------------------------------------------------
# Multi-image i2i: request order -> image1, image2, ... through the real applier
# ---------------------------------------------------------------------------


def _source_at(
    position: int, storage_key: str, asset_id: UUID | None = None
) -> ResolvedSourceMedia:
    return dataclasses.replace(
        _make_resolved_source_image(
            asset_id or uuid4(), source=LibraryAssetSource.UPLOAD, storage_key=storage_key
        ),
        position=position,
    )


def _i2i_request(count: int, *, aspect: AspectRatio | None = None) -> UnifiedGenerationRequest:
    return UnifiedGenerationRequest(
        prompt="the person in image 2 wearing the jacket from image 1",
        generation_type=GenerationType.I2I,
        model=ModelType.AISHA_IMAGE,
        aspect_ratio=aspect,
        n=1,
        source_media=[SourceMediaReference(asset_ref=f"upload:{uuid4()}") for _ in range(count)],
    )


class _I2IRun:
    """Everything observable after one ``submit``: uploads, R2 reads, the queued graph."""

    def __init__(self) -> None:
        self.workflow = MagicMock()
        self.r2 = AsyncMock()
        self.client = AsyncMock()
        self.billing = AsyncMock()
        self.job_repo_create = AsyncMock()

    @property
    def uploaded_names(self) -> list[str]:
        return [c.kwargs["filename"] for c in self.client.upload_image.await_args_list]

    @property
    def graph(self) -> dict[str, Any]:
        return self.client.queue_prompt.await_args.args[0]


async def _submit_i2i(
    sources: list[ResolvedSourceMedia],
    *,
    images: dict[str, bytes] | None = None,
    reference_slots: int = 2,
    aspect: AspectRatio | None = None,
    expect: type[Exception] | None = None,
) -> _I2IRun:
    """Submit an i2i request through the real applier over a ``qwen.rapid.aio``-shaped graph."""
    run = _I2IRun()
    bound = qwen_rapid_aio_bound_workflow(reference_slots)
    run.workflow.load.return_value = bound
    run.workflow.apply.side_effect = (
        lambda bound_, *, request, media_filenames, filename_prefix, **_: apply_bound_workflow(
            bound_,
            request,
            media_filenames=media_filenames,
            filename_prefix=filename_prefix,
            model_filenames=lambda _type: None,
        )
    )
    default_png = _png_bytes()
    run.r2.download = AsyncMock(side_effect=lambda key: (images or {}).get(key, default_png))
    # ComfyUI echoes back a stored name derived from the requested one.
    run.client.upload_image = AsyncMock(
        side_effect=lambda *, image_data, filename: {"name": f"stored_{filename}"}  # noqa: ARG005
    )
    run.client.queue_prompt = AsyncMock(return_value={"prompt_id": "queued-1"})
    run.billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))

    gpu_session_service = AsyncMock()
    gpu_session_service.get_active_session_for_model = AsyncMock(
        return_value=_routing(_make_active_gpu_session())
    )
    provider = AishaGenerationProvider(
        workflow_service=run.workflow,
        gpu_session_service=gpu_session_service,
        r2_storage=run.r2,
        tunnel_domain="gpu.test",
        bundle_index=_make_bundle_index_mock(),
    )
    db_job = MagicMock(status=JobStatus.PENDING, error_message=None)
    kwargs: dict[str, Any] = {
        "user_id": uuid4(),
        "session": AsyncMock(),
        "billing_service": run.billing,
        "account_id": uuid4(),
        "token_cost": 50,
        "product_id": "vex",
        "source_media": sources,
    }
    with (
        patch("src.api.services.generation.aisha.handlers.JobRepository") as job_repo,
        patch("src.api.services.generation.aisha.handlers.ComfyUIClient") as comfy,
    ):
        job_repo.return_value.create = run.job_repo_create
        run.job_repo_create.return_value = db_job
        comfy.return_value = run.client
        request = _i2i_request(len(sources), aspect=aspect)
        if expect is not None:
            with pytest.raises(expect):
                await provider.submit(request, **kwargs)
        else:
            await provider.submit(request, **kwargs)
    return run


def _referenced_nodes(graph: dict[str, Any]) -> set[str]:
    """Node ids that some other node's input links to."""
    return {
        value[0]
        for node in graph.values()
        for value in node["inputs"].values()
        if isinstance(value, list) and value and isinstance(value[0], str)
    }


class TestAishaMultiImageI2I:
    async def test_two_sources_upload_twice_and_link_image1_and_image2_in_request_order(
        self,
    ) -> None:
        run = await _submit_i2i(
            [
                _source_at(0, "users/u/uploads/alpha.png"),
                _source_at(1, "users/u/uploads/beta.png"),
            ]
        )

        names = run.uploaded_names
        assert len(names) == 2
        assert len(set(names)) == 2, "each reference needs its own uploaded filename"
        assert names[0].startswith("input_0_alpha_")
        assert names[1].startswith("input_1_beta_")

        graph = run.graph
        encoder_inputs = graph[QWEN_ENCODER_NODE]["inputs"]
        first_loader, second_loader = QWEN_LOAD_IMAGE_NODES
        assert encoder_inputs["image1"] == [first_loader, 0]
        assert encoder_inputs["image2"] == [second_loader, 0]
        # The name ComfyUI stored (not the requested one) reaches each LoadImage.
        assert graph[first_loader]["inputs"]["image"] == f"stored_{names[0]}"
        assert graph[second_loader]["inputs"]["image"] == f"stored_{names[1]}"

    async def test_sources_out_of_position_order_are_bound_in_position_order(self) -> None:
        # Supplied second-then-first; ``position`` says which is image1.
        run = await _submit_i2i(
            [
                _source_at(1, "users/u/uploads/second.png"),
                _source_at(0, "users/u/uploads/first.png"),
            ]
        )

        names = run.uploaded_names
        assert names[0].startswith("input_0_first_")
        assert names[1].startswith("input_1_second_")
        graph = run.graph
        first_loader, second_loader = QWEN_LOAD_IMAGE_NODES
        assert graph[first_loader]["inputs"]["image"] == f"stored_{names[0]}"
        assert graph[second_loader]["inputs"]["image"] == f"stored_{names[1]}"

    async def test_sources_with_the_same_basename_get_distinct_filenames(self) -> None:
        run = await _submit_i2i(
            [
                _source_at(0, "users/one/uploads/photo.png"),
                _source_at(1, "users/two/uploads/photo.png"),
            ]
        )

        names = run.uploaded_names
        assert len(set(names)) == 2
        assert names[0].startswith("input_0_photo_")
        assert names[1].startswith("input_1_photo_")

    async def test_the_same_asset_supplied_twice_does_not_collide(self) -> None:
        asset_id = uuid4()
        run = await _submit_i2i(
            [
                _source_at(0, "users/u/uploads/dup.png", asset_id),
                _source_at(1, "users/u/uploads/dup.png", asset_id),
            ]
        )

        assert len(set(run.uploaded_names)) == 2

    async def test_one_source_against_the_two_slot_bundle_links_only_image1(self) -> None:
        run = await _submit_i2i([_source_at(0, "users/u/uploads/solo.png")])

        graph = run.graph
        first_loader, second_loader = QWEN_LOAD_IMAGE_NODES
        encoder_inputs = graph[QWEN_ENCODER_NODE]["inputs"]
        assert encoder_inputs["image1"] == [first_loader, 0]
        assert "image2" not in encoder_inputs
        # Node 8 stays in the graph but nothing consumes it: ComfyUI never runs it.
        assert second_loader in graph
        assert second_loader not in _referenced_nodes(graph)
        assert first_loader in _referenced_nodes(graph)
        assert len(run.uploaded_names) == 1

    async def test_aspect_ratio_follows_the_first_image_when_the_two_differ(self) -> None:
        landscape = _png_bytes(size=(1600, 900))
        portrait = _png_bytes(size=(900, 1600))
        landscape_first = await _submit_i2i(
            [
                _source_at(0, "users/u/uploads/land.png"),
                _source_at(1, "users/u/uploads/port.png"),
            ],
            images={"users/u/uploads/land.png": landscape, "users/u/uploads/port.png": portrait},
        )
        portrait_first = await _submit_i2i(
            [
                _source_at(0, "users/u/uploads/port.png"),
                _source_at(1, "users/u/uploads/land.png"),
            ],
            images={"users/u/uploads/land.png": landscape, "users/u/uploads/port.png": portrait},
        )

        wide = landscape_first.workflow.apply.call_args.kwargs["request"]
        tall = portrait_first.workflow.apply.call_args.kwargs["request"]
        assert wide.width > wide.height
        assert tall.height > tall.width

    async def test_effective_aspect_reads_the_first_image_only(self) -> None:
        request = _i2i_request(2)
        landscape = (_png_bytes(size=(1600, 900)), "a.png")
        portrait = (_png_bytes(size=(900, 1600)), "b.png")

        assert await AishaImageGenerationHandler._resolve_effective_aspect_ratio(
            request, [landscape, portrait]
        ) == (16, 9)
        assert await AishaImageGenerationHandler._resolve_effective_aspect_ratio(
            request, [portrait, landscape]
        ) == (9, 16)

    async def test_normalization_failure_on_the_second_image_queues_nothing(self) -> None:
        run = await _submit_i2i(
            [
                _source_at(0, "users/u/uploads/good.png"),
                _source_at(1, "users/u/uploads/broken.png"),
            ],
            images={"users/u/uploads/broken.png": b"this is not an image, just text"},
            expect=ValueError,
        )

        # Every image is resolved before anything is uploaded, billed or queued.
        run.client.upload_image.assert_not_awaited()
        run.client.queue_prompt.assert_not_awaited()
        run.billing.check_and_reserve.assert_not_awaited()
        run.job_repo_create.assert_not_awaited()

    async def test_normalization_failure_reports_the_existing_error(self) -> None:
        r2 = AsyncMock()
        r2.download = AsyncMock(side_effect=[_png_bytes(), b"not an image"])
        handler = AishaImageGenerationHandler(
            workflow_service=MagicMock(),
            gpu_session_service=None,
            bundle_index=MagicMock(),
            r2_storage=r2,
            tunnel_domain="gpu.test",
            tunnel_hostname_allowed_prefix=None,
            max_input_megapixels=100.0,
        )

        with pytest.raises(ValueError, match="Input image is not decodable"):
            await handler._resolve_input_images(
                [_source_at(0, "users/u/uploads/a.png"), _source_at(1, "users/u/uploads/b.png")]
            )

    async def test_resolution_is_in_position_order_and_stable_for_equal_positions(self) -> None:
        r2 = AsyncMock()
        r2.download = AsyncMock(return_value=_png_bytes())
        handler = AishaImageGenerationHandler(
            workflow_service=MagicMock(),
            gpu_session_service=None,
            bundle_index=MagicMock(),
            r2_storage=r2,
            tunnel_domain="gpu.test",
            tunnel_hostname_allowed_prefix=None,
            max_input_megapixels=100.0,
        )

        resolved = await handler._resolve_input_images(
            [
                _source_at(2, "users/u/uploads/c.png"),
                _source_at(0, "users/u/uploads/a1.png"),
                _source_at(0, "users/u/uploads/a2.png"),
            ]
        )

        assert [name for _, name in resolved] == ["a1.png", "a2.png", "c.png"]

    async def test_no_sources_resolves_to_nothing_and_uploads_nothing(self) -> None:
        handler = AishaImageGenerationHandler(
            workflow_service=MagicMock(),
            gpu_session_service=None,
            bundle_index=MagicMock(),
            r2_storage=AsyncMock(),
            tunnel_domain="gpu.test",
            tunnel_hostname_allowed_prefix=None,
            max_input_megapixels=100.0,
        )
        client = AsyncMock()

        assert await handler._resolve_input_images([]) == []
        assert (
            await handler.prepare_inputs([], client=client, job_id=uuid4(), gpu_session_id=uuid4())
            == {}
        )
        client.upload_image.assert_not_awaited()
