"""Unit tests for AishaGenerationProvider.

Focused on provider-specific behaviours: session routing, per-session client
lifecycle, and the ProviderResponseError path when the backend returns a
malformed response (missing prompt_id).

The orchestrator-level refund flow is covered by ``test_unified_generation.py``
and is not duplicated here — raising any exception triggers the same code path.
"""

from __future__ import annotations

import io
import re
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import structlog.testing
from PIL import Image

from src.api.schemas.unified_generation import SourceImageReference, UnifiedGenerationRequest
from src.api.services.generation.aisha_provider import AishaGenerationProvider
from src.api.services.generation.service import FeatureNotSupportedError, ProviderResponseError
from src.api.services.generation.source_media import ResolvedSourceMedia
from src.api.services.gpu_session.exceptions import NoActiveSessionError
from src.core.enums import (
    AspectRatio,
    GenerationType,
    JobStatus,
    MediaKind,
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

    def test_source_images_validation_is_deferred_to_the_service(self) -> None:
        provider = AishaGenerationProvider(
            workflow_service=MagicMock(),
            gpu_session_service=MagicMock(),
            bundle_index=MagicMock(),
        )
        request = UnifiedGenerationRequest(
            prompt="edit multiple references",
            generation_type=GenerationType.I2I,
            model=ModelType.AISHA_IMAGE,
            source_images=[SourceImageReference(input_image_id=uuid4())],
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
    workflow.load_workflow_from_bundle.return_value = {"3": {"inputs": {}}}
    workflow.validate_workflow = MagicMock()
    workflow.inject_checkpoint = MagicMock()
    workflow.apply_parameters.return_value = {"3": {"inputs": {"text": "a cat"}}}

    gpu_session_service = AsyncMock()
    gpu_session_service.get_active_session_for_model = AsyncMock(
        return_value=_make_active_gpu_session()
    )
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
    return provider, {"workflow": workflow, "gpu_session_service": gpu_session_service}


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
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
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
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
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
        gpu_session_service.get_active_session_for_model = AsyncMock(return_value=malicious_session)

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

        with patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo:
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
        gpu_session_service.get_active_session_for_model = AsyncMock(return_value=broken_session)

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

        with patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo:
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
        gpu_session_service.get_active_session_for_model = AsyncMock(return_value=sneaky_session)

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

        with patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo:
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
        gpu_session_service.get_active_session_for_model = AsyncMock(return_value=sneaky_session)

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

        with patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo:
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
        input_image_id=input_image_id,
    )


def _make_i2i_request_with_source_output(source_output_id: UUID) -> UnifiedGenerationRequest:
    return UnifiedGenerationRequest(
        prompt="a cat in a hat",
        generation_type=GenerationType.I2I,
        model=ModelType.AISHA_IMAGE,
        aspect_ratio=AspectRatio.RATIO_1_1,
        n=1,
        source_output_id=source_output_id,
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
        workflow.load_workflow_from_bundle.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.inject_checkpoint = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_make_active_gpu_session()
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
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
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
        # f"input_{stem}_{job_id_short}.{ext}" — extension is terminal.
        mock_client.upload_image.assert_awaited_once()
        upload_kwargs = mock_client.upload_image.await_args.kwargs
        assert upload_kwargs["image_data"] == png_bytes
        # source_filename stem in template; suffix is the 8-char job_id prefix
        # followed by the terminal extension.
        assert upload_kwargs["filename"].startswith("input_cat_")
        assert upload_kwargs["filename"].endswith(".png")
        # 8 hex chars between the stem and the extension.
        job_part = upload_kwargs["filename"].rsplit("_", 1)[1].split(".")[0]
        assert len(job_part) == 8

        # Workflow received the ComfyUI-stored name (which may differ from the
        # requested one) wired into LoadImage via input_image_1.
        workflow.apply_parameters.assert_called_once()
        ap_kwargs = workflow.apply_parameters.call_args.kwargs
        assert ap_kwargs["input_image_1"] == "input_cat.png_deadbeef"

        # Job persisted as QUEUED with the ComfyUI prompt_id.
        assert db_job.status == JobStatus.QUEUED
        assert db_job.external_request_id == "queued-123"

    async def test_i2i_with_source_output_id_takes_precedence(self) -> None:
        """source_output_id branch: filename derived from storage_key tail."""
        workflow = MagicMock()
        workflow.load_workflow_from_bundle.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.inject_checkpoint = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_make_active_gpu_session()
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
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
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
        assert upload_kwargs["filename"].startswith("input_result_001_")
        assert upload_kwargs["filename"].endswith(".jpeg")

        # Workflow wired with the ComfyUI-returned name.
        ap_kwargs = workflow.apply_parameters.call_args.kwargs
        assert ap_kwargs["input_image_1"] == "input_result_001.jpg_aaaaaaaa"

    async def test_i2i_multiple_sources_raise_before_billing(self) -> None:
        """Provider-level cardinality checks run before billing or job creation."""
        workflow = MagicMock()
        workflow.load_workflow.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_make_active_gpu_session()
        )
        r2 = AsyncMock()
        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            bundle_index=MagicMock(),
            r2_storage=r2,
            tunnel_domain="gpu.test",
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo:
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)

            with pytest.raises(FeatureNotSupportedError, match="exactly one"):
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
                            storage_key="uploads/one.png",
                        ),
                        _make_resolved_source_image(
                            uuid4(),
                            source=LibraryAssetSource.UPLOAD,
                            storage_key="uploads/two.png",
                        ),
                    ],
                )

        # The cardinality check runs before any R2 access.
        r2.download.assert_not_called()
        billing.check_and_reserve.assert_not_called()
        MockJobRepo.return_value.create.assert_not_called()

    async def test_i2i_without_r2_dependency_raises_provider_response_error(self) -> None:
        """Defensive: if a deployment forgets to wire R2, I2I requests get a
        clean infrastructure-agnostic error, not a NoneType AttributeError."""
        workflow = MagicMock()
        workflow.load_workflow.return_value = {"3": {"inputs": {}}}
        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_make_active_gpu_session()
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

        with patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo:
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
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
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
        # Workflow received None for input_image_1.
        ap_kwargs = mocks["workflow"].apply_parameters.call_args.kwargs
        assert ap_kwargs["input_image_1"] is None
        # inject_checkpoint must receive bundle_name as str, not a Path.
        mocks["workflow"].inject_checkpoint.assert_called_once()
        ic_args, _ic_kwargs = mocks["workflow"].inject_checkpoint.call_args
        assert isinstance(ic_args[1], str), (
            "inject_checkpoint second arg must be bundle_name (str), not Path"
        )
        gpu_session = mocks["gpu_session_service"].get_active_session_for_model.return_value
        assert ic_args[1] == gpu_session.bundle_name

    async def test_i2i_webp_source_output_converted_to_png_for_comfyui(self) -> None:
        """D2': a Grok-output WebP remixed via source_output_id must reach
        ComfyUI as PNG bytes with a matching terminal filename extension."""
        workflow = MagicMock()
        workflow.load_workflow_from_bundle.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.inject_checkpoint = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_make_active_gpu_session()
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
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
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
        assert re.match(r"^input_.+_[0-9a-f]{8}\.png$", upload_kwargs["filename"])

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
            return_value=_make_active_gpu_session()
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
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
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
            return_value=_make_active_gpu_session()
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
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
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
            return_value=_make_active_gpu_session()
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
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
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
        legacy_request = workflow.apply_parameters.call_args.kwargs["request"]
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
        legacy_request = workflow.apply_parameters.call_args.kwargs["request"]
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
        legacy_request = workflow.apply_parameters.call_args.kwargs["request"]
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
            input_image_id=uuid4(),
        )
        result = await AishaGenerationProvider._resolve_effective_aspect_ratio(
            request, (_png_bytes(size=(1600, 900)), "source.png")
        )
        assert result == AspectRatio.RATIO_9_16

    async def test_resolve_effective_aspect_ratio_t2i_defaults_1_1(self) -> None:
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            n=1,
        )
        result = await AishaGenerationProvider._resolve_effective_aspect_ratio(request, None)
        assert result is AspectRatio.RATIO_1_1

    async def test_resolve_effective_aspect_ratio_derives_source_fraction(self) -> None:
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.I2I,
            model=ModelType.AISHA_IMAGE,
            n=1,
            input_image_id=uuid4(),
        )
        result = await AishaGenerationProvider._resolve_effective_aspect_ratio(
            request, (_png_bytes(size=(1600, 900)), "source.png")
        )
        assert result == (16, 9)

    async def test_resolve_effective_aspect_ratio_undecodable_raises_value_error(self) -> None:
        request = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.I2I,
            model=ModelType.AISHA_IMAGE,
            n=1,
            input_image_id=uuid4(),
        )
        with pytest.raises(ValueError, match="not decodable"):
            await AishaGenerationProvider._resolve_effective_aspect_ratio(
                request, (b"garbage bytes, not an image", "broken.png")
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
        from src.api.services.generation.aisha_provider import _sanitize_filename

        result = _sanitize_filename(raw)
        assert result == expected_pattern, (
            f"input={raw!r}: expected {expected_pattern!r}, got {result!r}"
        )

    def test_sanitization_keeps_safe_chars_unchanged(self) -> None:
        """The conservative charset preserves alphanumerics, dot, underscore, hyphen."""
        from src.api.services.generation.aisha_provider import _sanitize_filename

        safe = "Photo_2024-12-25.final.v3.png"
        assert _sanitize_filename(safe) == safe


class TestAishaProviderI2IDefensive:
    """Provider-level defenses against misuse and hostile input."""

    async def test_multiple_sources_raise_feature_not_supported(self) -> None:
        """The provider defensively limits direct callers to one source asset."""
        provider, _ = _make_provider_with_mocks()
        sources = [
            _make_resolved_source_image(
                uuid4(), source=LibraryAssetSource.UPLOAD, storage_key="uploads/one.png"
            ),
            _make_resolved_source_image(
                uuid4(), source=LibraryAssetSource.UPLOAD, storage_key="uploads/two.png"
            ),
        ]

        with pytest.raises(FeatureNotSupportedError, match="exactly one"):
            await provider._resolve_input_image(sources)

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
            return_value=_make_active_gpu_session()
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
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
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
    workflow.load_workflow_from_bundle.return_value = {
        "3": {"inputs": {}},
        "9": {"inputs": {}},
        "2": {"inputs": {}},
    }
    workflow.validate_workflow = MagicMock()
    workflow.inject_checkpoint = MagicMock()
    workflow.apply_parameters.return_value = {"3": {"inputs": {"text": "a cat"}}}

    gpu_session_service = AsyncMock()
    gpu_session_service.get_active_session_for_model = AsyncMock(
        return_value=_make_active_gpu_session()
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
        patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
        patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
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
            return_value=_make_active_gpu_session()
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
            return_value=_make_active_gpu_session()
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
            return_value=_make_active_gpu_session()
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
        workflow.load_workflow_from_bundle.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.inject_checkpoint = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_make_active_gpu_session()
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
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
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
