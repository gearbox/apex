"""HTTP-level tests for POST .../commands/claim (invariants #8, #9, #10, #11).

The service is stubbed; this file only asserts the wire-level status/body/header
contract the aisha-agent client depends on (see agent_prompts/apex-p3-command-queue-prompt.md,
D29/D30 and the pitfalls section on 204 having no body).
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

from litestar import Litestar
from litestar.di import Provide
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_204_NO_CONTENT,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
)
from litestar.testing import TestClient

from src.api.routes.internal_gpu_session import InternalGpuSessionController
from src.api.services.gpu_session.command_service import GpuSessionCommandService


def _stub_service(result: tuple[int, dict[str, object] | None] = (204, None)) -> AsyncMock:
    service = AsyncMock(spec=GpuSessionCommandService)
    service.claim.return_value = result
    return service


def _app(service: AsyncMock) -> Litestar:
    return Litestar(
        route_handlers=[InternalGpuSessionController],
        dependencies={
            "gpu_session_command_service": Provide(lambda: service, sync_to_thread=False)
        },
    )


def _claim_path(session_id: object) -> str:
    return f"/v1/internal/gpu-sessions/{session_id}/commands/claim"


class TestClaimCommandController:
    def test_missing_bearer_is_401_json(self) -> None:
        session_id = uuid4()
        service = _stub_service()
        with TestClient(app=_app(service)) as client:
            response = client.post(
                _claim_path(session_id), json={"agent_id": "a:h", "schema_version": 2}
            )

        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert response.headers["content-type"].startswith("application/json")
        service.claim.assert_not_awaited()

    def test_empty_bearer_is_401(self) -> None:
        session_id = uuid4()
        service = _stub_service()
        with TestClient(app=_app(service)) as client:
            response = client.post(
                _claim_path(session_id),
                json={"agent_id": "a:h", "schema_version": 2},
                headers={"Authorization": "Bearer "},
            )

        assert response.status_code == HTTP_401_UNAUTHORIZED
        service.claim.assert_not_awaited()

    def test_wrong_schema_version_is_400(self) -> None:
        session_id = uuid4()
        service = _stub_service()
        with TestClient(app=_app(service)) as client:
            response = client.post(
                _claim_path(session_id),
                json={"agent_id": "a:h", "schema_version": 3},
                headers={"Authorization": "Bearer tok"},
            )

        assert response.status_code == HTTP_400_BAD_REQUEST
        assert response.headers["content-type"].startswith("application/json")
        service.claim.assert_not_awaited()

    def test_empty_agent_id_is_400(self) -> None:
        session_id = uuid4()
        service = _stub_service()
        with TestClient(app=_app(service)) as client:
            response = client.post(
                _claim_path(session_id),
                json={"agent_id": "", "schema_version": 2},
                headers={"Authorization": "Bearer tok"},
            )

        assert response.status_code == HTTP_400_BAD_REQUEST
        service.claim.assert_not_awaited()

    def test_service_401_is_passed_through_as_json(self) -> None:
        session_id = uuid4()
        service = _stub_service((401, None))
        with TestClient(app=_app(service)) as client:
            response = client.post(
                _claim_path(session_id),
                json={"agent_id": "a:h", "schema_version": 2},
                headers={"Authorization": "Bearer tok"},
            )

        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert response.headers["content-type"].startswith("application/json")

    def test_no_work_is_204_with_empty_body(self) -> None:
        """Invariant #8: no queued work -> 204 with an empty body."""
        session_id = uuid4()
        service = _stub_service((204, None))
        with TestClient(app=_app(service)) as client:
            response = client.post(
                _claim_path(session_id),
                json={"agent_id": "a:h", "schema_version": 2},
                headers={"Authorization": "Bearer tok"},
            )

        assert response.status_code == HTTP_204_NO_CONTENT
        assert response.content == b""

    def test_successful_claim_is_200_json_envelope(self) -> None:
        envelope = {
            "command_id": "cmd-1",
            "operation_id": "op-1",
            "kind": "bundle_provision",
            "batch": None,
            "payload": {"bundle": "wan_2.2_i2v", "mode": "full", "verify": True},
        }
        session_id = uuid4()
        service = _stub_service((200, envelope))
        with TestClient(app=_app(service)) as client:
            response = client.post(
                _claim_path(session_id),
                json={"agent_id": "a:h", "schema_version": 2},
                headers={"Authorization": "Bearer tok"},
            )

        assert response.status_code == HTTP_200_OK
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == envelope

    def test_service_receives_extracted_bearer_and_agent_id(self) -> None:
        session_id = uuid4()
        service = _stub_service((204, None))
        with TestClient(app=_app(service)) as client:
            client.post(
                _claim_path(session_id),
                json={"agent_id": "sess:host-1", "schema_version": 2},
                headers={"Authorization": "Bearer real-token"},
            )

        service.claim.assert_awaited_once_with(
            session_id=session_id, bearer_token="real-token", agent_id="sess:host-1"
        )
