"""Unit tests for AishaGenerationProvider.

Focused on provider-specific behaviours: session routing, per-session client
lifecycle, and the ProviderResponseError path when the backend returns a
malformed response (missing prompt_id).

The orchestrator-level refund flow is covered by ``test_unified_generation.py``
and is not duplicated here — raising any exception triggers the same code path.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.api.schemas.unified_generation import UnifiedGenerationRequest
from src.api.services.generation.aisha_provider import AishaGenerationProvider
from src.api.services.generation.service import ProviderResponseError
from src.api.services.gpu_session.exceptions import NoActiveSessionError
from src.core.enums import AspectRatio, GenerationType, JobStatus, ModelType


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
    return gs


def _make_provider_with_mocks() -> tuple[AishaGenerationProvider, dict]:
    workflow = MagicMock()
    workflow.load_workflow.return_value = {"3": {"inputs": {}}}
    workflow.validate_workflow = MagicMock()
    workflow.apply_parameters.return_value = {"3": {"inputs": {"text": "a cat"}}}

    gpu_session_service = AsyncMock()
    gpu_session_service.get_active_session_for_model = AsyncMock(
        return_value=_make_active_gpu_session()
    )

    # tunnel_domain matches the mocked hostname suffix so the SSRF allowlist
    # check passes by default; SSRF-specific tests construct their own provider
    # with a deliberately-mismatched tunnel_domain.
    provider = AishaGenerationProvider(
        workflow_service=workflow,
        gpu_session_service=gpu_session_service,
        tunnel_domain="gpu.test",
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
        provider, mocks = _make_provider_with_mocks()

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
            # Uppercase: Cloudflare emits lowercase, anything else is corruption
            "AB123.GPU.TEST",
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
        workflow.load_workflow.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_make_active_gpu_session()
        )

        # R2 returns the source bytes for the configured user image.
        user_image_id = uuid4()
        r2 = AsyncMock()
        r2.download = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n[fake-png-bytes]")

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            r2_storage=r2,
            tunnel_domain="gpu.test",
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))

        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        # User-image repo returns a row with a stable filename for the assertion.
        user_image_row = MagicMock()
        user_image_row.storage_key = "users/abc/uploads/cat.png"
        user_image_row.original_filename = "cat.png"

        with (
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
            patch(
                "src.api.services.generation.aisha_provider.UserImageRepository"
            ) as MockUserImageRepo,
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
            MockUserImageRepo.return_value.get = AsyncMock(return_value=user_image_row)

            await provider.submit(
                _make_i2i_request_with_user_image(user_image_id),
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
            )

        # R2 was hit with the storage key from the user image row.
        r2.download.assert_awaited_once_with("users/abc/uploads/cat.png")

        # ComfyUI received the bytes with the agreed filename template:
        # f"input_{source_filename}_{job_id_short}".
        mock_client.upload_image.assert_awaited_once()
        upload_kwargs = mock_client.upload_image.await_args.kwargs
        assert upload_kwargs["image_data"] == b"\x89PNG\r\n\x1a\n[fake-png-bytes]"
        # source_filename in template; suffix is the 8-char job_id prefix.
        assert upload_kwargs["filename"].startswith("input_cat.png_")
        # 8 hex chars after the underscore.
        suffix = upload_kwargs["filename"].rsplit("_", 1)[1]
        assert len(suffix) == 8

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
        workflow.load_workflow.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_make_active_gpu_session()
        )

        r2 = AsyncMock()
        r2.download = AsyncMock(return_value=b"\xff\xd8\xff[fake-jpg]")

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            r2_storage=r2,
            tunnel_domain="gpu.test",
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        output_row = MagicMock()
        output_row.storage_key = "outputs/job_xyz/result_001.jpg"

        # Single ID flows through both the request body and the kwarg used
        # for the source_jobs.source_output_id FK column on the GenerationJob row.
        source_id = uuid4()

        with (
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
            patch("src.api.services.generation.aisha_provider.OutputRepository") as MockOutputRepo,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            mock_client.upload_image = AsyncMock(
                return_value={"name": "input_result_001.jpg_aaaaaaaa"}
            )
            mock_client.queue_prompt = AsyncMock(return_value={"prompt_id": "queued-456"})
            MockComfyClient.return_value = mock_client
            MockOutputRepo.return_value.get = AsyncMock(return_value=output_row)

            await provider.submit(
                _make_i2i_request_with_source_output(source_id),
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
                source_output_id=source_id,
            )

        # Repo lookup used the request's source_output_id (the request body is
        # authoritative; the submit kwarg is for the GenerationJob FK column).
        MockOutputRepo.return_value.get.assert_awaited_once_with(source_id)

        # Filename derived from storage_key tail (not original_filename).
        upload_kwargs = mock_client.upload_image.await_args.kwargs
        assert upload_kwargs["filename"].startswith("input_result_001.jpg_")

        # Workflow wired with the ComfyUI-returned name.
        ap_kwargs = workflow.apply_parameters.call_args.kwargs
        assert ap_kwargs["input_image_1"] == "input_result_001.jpg_aaaaaaaa"

    async def test_i2i_user_image_not_found_raises(self) -> None:
        """A missing input_image_id row raises BEFORE billing — no debit, no refund needed.

        Resolution happens before reservation so user-error inputs (missing
        rows, mutually-exclusive IDs) don't leave a debit on the ledger
        that requires later refund. This is a fail-fast contract.
        """
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
            r2_storage=r2,
            tunnel_domain="gpu.test",
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        with (
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch(
                "src.api.services.generation.aisha_provider.UserImageRepository"
            ) as MockUserImageRepo,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            MockUserImageRepo.return_value.get = AsyncMock(return_value=None)

            with pytest.raises(ValueError, match="not found"):
                await provider.submit(
                    _make_i2i_request_with_user_image(uuid4()),
                    user_id=uuid4(),
                    session=session,
                    billing_service=billing,
                    account_id=uuid4(),
                    token_cost=50,
                    product_id="vex",
                )

        # R2 download was never reached because the row lookup returned None.
        r2.download.assert_not_called()
        # Billing reservation must NOT have run — that's the whole point of
        # resolving inputs before the ledger write. No debit means no refund
        # responsibility on the orchestrator.
        billing.check_and_reserve.assert_not_called()
        # The DB job row also was never created (resolve runs even before
        # JobRepository.create).
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

    async def test_both_inputs_set_raises_value_error(self) -> None:
        """Defensive guard: orchestrator already enforces mutual exclusion,
        but a future direct caller of the provider deserves a clear local
        error rather than a silently-ignored input_image_id."""
        from src.api.schemas.unified_generation import UnifiedGenerationRequest

        # Construct a request with BOTH set. Bypass the unified schema's own
        # validator if it has one by using model_construct (msgspec equivalent).
        req = UnifiedGenerationRequest(
            prompt="a cat",
            generation_type=GenerationType.I2I,
            model=ModelType.AISHA_IMAGE,
            aspect_ratio=AspectRatio.RATIO_1_1,
            n=1,
            input_image_id=uuid4(),
            source_output_id=uuid4(),
        )

        provider, _ = _make_provider_with_mocks()

        with pytest.raises(ValueError, match="mutually exclusive"):
            await provider._resolve_input_image(req, AsyncMock())

    async def test_filename_with_path_components_is_sanitized(self) -> None:
        """End-to-end: a user image whose original_filename contains path
        traversal yields a sanitized ComfyUI filename. The job_id suffix
        is preserved for uniqueness.
        """
        workflow = MagicMock()
        workflow.load_workflow.return_value = {"3": {"inputs": {}}}
        workflow.validate_workflow = MagicMock()
        workflow.apply_parameters.return_value = {"3": {}}

        gpu_session_service = AsyncMock()
        gpu_session_service.get_active_session_for_model = AsyncMock(
            return_value=_make_active_gpu_session()
        )

        r2 = AsyncMock()
        r2.download = AsyncMock(return_value=b"\x89PNG[bytes]")

        provider = AishaGenerationProvider(
            workflow_service=workflow,
            gpu_session_service=gpu_session_service,
            r2_storage=r2,
            tunnel_domain="gpu.test",
        )

        billing = AsyncMock()
        billing.check_and_reserve = AsyncMock(return_value=MagicMock(id=uuid4()))
        session = AsyncMock()
        db_job = MagicMock(status=JobStatus.PENDING, error_message=None)

        # Hostile filename: tries path traversal AND has shell metacharacters.
        user_image_row = MagicMock()
        user_image_row.storage_key = "users/u_1/uploads/legit.png"
        user_image_row.original_filename = "../../../etc/passwd; rm -rf /.png"

        with (
            patch("src.api.services.generation.aisha_provider.JobRepository") as MockJobRepo,
            patch("src.api.services.generation.aisha_provider.ComfyUIClient") as MockComfyClient,
            patch(
                "src.api.services.generation.aisha_provider.UserImageRepository"
            ) as MockUserImageRepo,
        ):
            MockJobRepo.return_value.create = AsyncMock(return_value=db_job)
            mock_client = AsyncMock()
            mock_client.upload_image = AsyncMock(return_value={"name": "stored.png"})
            mock_client.queue_prompt = AsyncMock(return_value={"prompt_id": "queued-x"})
            MockComfyClient.return_value = mock_client
            MockUserImageRepo.return_value.get = AsyncMock(return_value=user_image_row)

            await provider.submit(
                _make_i2i_request_with_user_image(uuid4()),
                user_id=uuid4(),
                session=session,
                billing_service=billing,
                account_id=uuid4(),
                token_cost=50,
                product_id="vex",
            )

        upload_kwargs = mock_client.upload_image.await_args.kwargs
        # No slashes survived.
        assert "/" not in upload_kwargs["filename"]
        # No shell metacharacters survived (semicolon, space, exclamation etc).
        assert ";" not in upload_kwargs["filename"]
        assert " " not in upload_kwargs["filename"]
        # The job_id suffix is preserved (8 hex chars after the trailing _).
        suffix = upload_kwargs["filename"].rsplit("_", 1)[1]
        assert len(suffix) == 8
        # The "input_" prefix is preserved.
        assert upload_kwargs["filename"].startswith("input_")
