"""Unit tests for AishaGenerationProvider.

Focused on provider-specific behaviours: session routing, per-session client
lifecycle, and the ProviderResponseError path when the backend returns a
malformed response (missing prompt_id).

The orchestrator-level refund flow is covered by ``test_unified_generation.py``
and is not duplicated here — raising any exception triggers the same code path.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

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
        assert "refunded" in msg.lower()

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
