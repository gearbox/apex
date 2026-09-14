"""Unit tests for ProvisioningWebhookService (Change 3 / D7-D9)."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from litestar.status_codes import HTTP_200_OK, HTTP_401_UNAUTHORIZED

from src.api.schemas.provisioning import ProvisionerFailureWebhookBody
from src.api.services.provisioning_webhook import ProvisioningWebhookService
from src.core.enums import GpuSessionStatus

_REPO_PATH = "src.api.services.provisioning_webhook.GpuSessionRepository"


def _make_payload(**overrides: object) -> ProvisionerFailureWebhookBody:
    fields: dict[str, object] = {
        "action": "continue",
        "manifest": "/path/to/manifest.yaml",
        "error": "provisioning failed after all retries",
        "container_id": "50885024",
        "timestamp": "2026-09-13T12:36:41",
    } | overrides
    return ProvisionerFailureWebhookBody(**fields)  # type: ignore[arg-type]


def _make_session_row(
    *,
    token: str = "correct-token",
    status: GpuSessionStatus = GpuSessionStatus.provisioning,
    vastai_instance_id: int | None = 50885024,
) -> MagicMock:
    row = MagicMock()
    row.callback_token_hash = hashlib.sha256(token.encode()).hexdigest()
    row.status = status
    row.vastai_instance_id = vastai_instance_id
    return row


def _make_mock_session_factory() -> MagicMock:
    mock_db = MagicMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=None)
    return MagicMock(return_value=mock_db)


class TestHandleFailure:
    async def test_unknown_session_is_200_noop(self) -> None:
        gpu_session_service = AsyncMock()
        service = ProvisioningWebhookService(
            gpu_session_service=gpu_session_service,
            session_factory=_make_mock_session_factory(),
        )
        with patch(_REPO_PATH) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=None)
            status = await service.handle_failure(
                session_id=uuid4(), token="whatever", payload=_make_payload()
            )

        assert status == HTTP_200_OK
        gpu_session_service.fail_pre_active_session.assert_not_awaited()

    async def test_already_terminal_session_is_200_noop_no_second_refund(self) -> None:
        gpu_session_service = AsyncMock()
        service = ProvisioningWebhookService(
            gpu_session_service=gpu_session_service,
            session_factory=_make_mock_session_factory(),
        )
        row = _make_session_row(status=GpuSessionStatus.failed)
        with patch(_REPO_PATH) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=row)
            status = await service.handle_failure(
                session_id=uuid4(), token="correct-token", payload=_make_payload()
            )

        assert status == HTTP_200_OK
        gpu_session_service.fail_pre_active_session.assert_not_awaited()

    async def test_wrong_token_is_401_session_untouched(self) -> None:
        gpu_session_service = AsyncMock()
        service = ProvisioningWebhookService(
            gpu_session_service=gpu_session_service,
            session_factory=_make_mock_session_factory(),
        )
        row = _make_session_row(token="correct-token")
        with patch(_REPO_PATH) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=row)
            status = await service.handle_failure(
                session_id=uuid4(), token="wrong-token", payload=_make_payload()
            )

        assert status == HTTP_401_UNAUTHORIZED
        gpu_session_service.fail_pre_active_session.assert_not_awaited()

    async def test_missing_token_is_401(self) -> None:
        gpu_session_service = AsyncMock()
        service = ProvisioningWebhookService(
            gpu_session_service=gpu_session_service,
            session_factory=_make_mock_session_factory(),
        )
        row = _make_session_row()
        with patch(_REPO_PATH) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=row)
            status = await service.handle_failure(
                session_id=uuid4(), token=None, payload=_make_payload()
            )

        assert status == HTTP_401_UNAUTHORIZED

    async def test_valid_call_fails_the_session_with_reason(self) -> None:
        gpu_session_service = AsyncMock()
        service = ProvisioningWebhookService(
            gpu_session_service=gpu_session_service,
            session_factory=_make_mock_session_factory(),
        )
        session_id = uuid4()
        row = _make_session_row(token="correct-token")
        with patch(_REPO_PATH) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=row)
            status = await service.handle_failure(
                session_id=session_id,
                token="correct-token",
                payload=_make_payload(error="provisioning failed after all retries"),
            )

        assert status == HTTP_200_OK
        gpu_session_service.fail_pre_active_session.assert_awaited_once()
        call = gpu_session_service.fail_pre_active_session.await_args
        assert call.args[0] == session_id
        assert "node_provision_script_failed" in call.kwargs["reason"]
        assert "provisioning failed after all retries" in call.kwargs["reason"]

    async def test_container_id_mismatch_still_fails_the_session(self) -> None:
        """The token is the authority, not the container id (D9)."""
        gpu_session_service = AsyncMock()
        service = ProvisioningWebhookService(
            gpu_session_service=gpu_session_service,
            session_factory=_make_mock_session_factory(),
        )
        row = _make_session_row(token="correct-token", vastai_instance_id=111)
        with patch(_REPO_PATH) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=row)
            status = await service.handle_failure(
                session_id=uuid4(),
                token="correct-token",
                payload=_make_payload(container_id="999999"),
            )

        assert status == HTTP_200_OK
        gpu_session_service.fail_pre_active_session.assert_awaited_once()

    async def test_matching_container_id_logs_no_mismatch(self) -> None:
        from structlog.testing import capture_logs

        gpu_session_service = AsyncMock()
        service = ProvisioningWebhookService(
            gpu_session_service=gpu_session_service,
            session_factory=_make_mock_session_factory(),
        )
        row = _make_session_row(token="correct-token", vastai_instance_id=50885024)
        with (
            patch(_REPO_PATH) as MockRepo,
            capture_logs() as logs,
        ):
            MockRepo.return_value.get_by_id = AsyncMock(return_value=row)
            await service.handle_failure(
                session_id=uuid4(),
                token="correct-token",
                payload=_make_payload(container_id="50885024"),
            )

        mismatch_logs = [
            log for log in logs if log.get("event") == "provisioning.webhook.instance_mismatch"
        ]
        assert not mismatch_logs

    async def test_reason_is_length_bounded(self) -> None:
        gpu_session_service = AsyncMock()
        service = ProvisioningWebhookService(
            gpu_session_service=gpu_session_service,
            session_factory=_make_mock_session_factory(),
        )
        row = _make_session_row(token="correct-token")
        with patch(_REPO_PATH) as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=row)
            await service.handle_failure(
                session_id=uuid4(),
                token="correct-token",
                payload=_make_payload(error="x" * 10_000),
            )

        reason = gpu_session_service.fail_pre_active_session.await_args.kwargs["reason"]
        assert len(reason) <= 500


class TestMalformedBody:
    def test_unknown_extra_field_decodes_fine(self) -> None:
        """msgspec's default is to ignore unknown fields (upstream schema may grow)."""
        import msgspec

        body = msgspec.json.decode(
            msgspec.json.encode(
                {
                    "action": "continue",
                    "manifest": "/x.yaml",
                    "error": "failed",
                    "container_id": "123",
                    "timestamp": "2026-09-13T12:36:41",
                    "new_upstream_field": "some value apex doesn't know about yet",
                }
            ),
            type=ProvisionerFailureWebhookBody,
        )
        assert body.container_id == "123"

    def test_container_id_is_typed_as_string(self) -> None:
        payload = _make_payload(container_id="50885024")
        assert isinstance(payload.container_id, str)
