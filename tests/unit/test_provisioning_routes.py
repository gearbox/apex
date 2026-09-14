"""HTTP-level tests for the provisioning routes (Change 1 GET script + Change 3 webhook).

The services are stubbed; these tests only assert the wire-level status/body/header
contract — see test_provisioning_script_service.py and test_provisioning_webhook_service.py
for the underlying service logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

from litestar import Litestar
from litestar.di import Provide
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_304_NOT_MODIFIED,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
    HTTP_502_BAD_GATEWAY,
)
from litestar.testing import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes.provisioning import ProvisioningController
from src.api.services.provisioning_script import (
    ProvisioningScriptService,
    ResolvedScript,
    ScriptServeResult,
)
from src.api.services.provisioning_webhook import ProvisioningWebhookService
from src.core.config import Settings


def _make_settings(**overrides: object) -> Settings:
    fields = {
        "comfyui_host": "127.0.0.1",
        "comfyui_port": Settings.model_fields["comfyui_port"].default,
        "rate_limit_provisioning_script": "1000/minute",
    } | overrides
    return Settings(**fields)  # type: ignore[arg-type]


def _app(
    script_service: AsyncMock,
    webhook_service: AsyncMock,
    *,
    settings: Settings | None = None,
) -> Litestar:
    return Litestar(
        route_handlers=[ProvisioningController],
        dependencies={
            "provisioning_script_service": Provide(lambda: script_service, sync_to_thread=False),
            "provisioning_webhook_service": Provide(lambda: webhook_service, sync_to_thread=False),
            "settings": Provide(lambda: settings or _make_settings(), sync_to_thread=False),
            "session": Provide(lambda: AsyncMock(spec=AsyncSession), sync_to_thread=False),
        },
    )


class TestGetScriptRoute:
    def test_ok_returns_text_plain_with_etag(self) -> None:
        script_service = AsyncMock(spec=ProvisioningScriptService)
        script_service.serve_for_session.return_value = ScriptServeResult(
            outcome="ok",
            script=ResolvedScript(content="#!/bin/sh\necho hi\n", sha256="a" * 64, cache_hit=True),
        )
        with TestClient(app=_app(script_service, AsyncMock())) as client:
            response = client.get(
                "/v1/provisioning/scripts/comfyui/v1.2.3",
                params={"session": str(uuid4()), "token": "tok"},
            )

        assert response.status_code == HTTP_200_OK
        assert response.headers["content-type"].startswith("text/plain")
        assert response.headers["ETag"] == '"' + "a" * 64 + '"'
        assert response.text == "#!/bin/sh\necho hi\n"

    def test_if_none_match_returns_304_empty_body(self) -> None:
        script_service = AsyncMock(spec=ProvisioningScriptService)
        sha = "b" * 64
        script_service.serve_for_session.return_value = ScriptServeResult(
            outcome="ok",
            script=ResolvedScript(content="body", sha256=sha, cache_hit=True),
        )
        with TestClient(app=_app(script_service, AsyncMock())) as client:
            response = client.get(
                "/v1/provisioning/scripts/comfyui/v1.2.3",
                params={"session": str(uuid4()), "token": "tok"},
                headers={"If-None-Match": f'"{sha}"'},
            )

        assert response.status_code == HTTP_304_NOT_MODIFIED
        assert response.content == b""

    def test_bad_request_outcome_is_400(self) -> None:
        script_service = AsyncMock(spec=ProvisioningScriptService)
        script_service.serve_for_session.return_value = ScriptServeResult(outcome="bad_request")
        with TestClient(app=_app(script_service, AsyncMock())) as client:
            response = client.get(
                "/v1/provisioning/scripts/comfyui/master",
                params={"session": str(uuid4()), "token": "tok"},
            )

        assert response.status_code == HTTP_400_BAD_REQUEST

    def test_unauthorized_outcome_is_401(self) -> None:
        script_service = AsyncMock(spec=ProvisioningScriptService)
        script_service.serve_for_session.return_value = ScriptServeResult(outcome="unauthorized")
        with TestClient(app=_app(script_service, AsyncMock())) as client:
            response = client.get(
                "/v1/provisioning/scripts/comfyui/v1.2.3",
                params={"session": str(uuid4()), "token": "bad"},
            )

        assert response.status_code == HTTP_401_UNAUTHORIZED

    def test_missing_token_is_401(self) -> None:
        script_service = AsyncMock(spec=ProvisioningScriptService)
        script_service.serve_for_session.return_value = ScriptServeResult(outcome="unauthorized")
        with TestClient(app=_app(script_service, AsyncMock())) as client:
            response = client.get(
                "/v1/provisioning/scripts/comfyui/v1.2.3",
                params={"session": str(uuid4())},
            )

        assert response.status_code == HTTP_401_UNAUTHORIZED

    def test_not_found_outcome_is_404_with_code(self) -> None:
        script_service = AsyncMock(spec=ProvisioningScriptService)
        script_service.serve_for_session.return_value = ScriptServeResult(outcome="not_found")
        with TestClient(app=_app(script_service, AsyncMock())) as client:
            response = client.get(
                "/v1/provisioning/scripts/comfyui/v9.9.9",
                params={"session": str(uuid4()), "token": "tok"},
            )

        assert response.status_code == HTTP_404_NOT_FOUND
        assert response.json()["error"] == "provisioning_script_ref_not_found"

    def test_unavailable_outcome_is_502_with_code(self) -> None:
        script_service = AsyncMock(spec=ProvisioningScriptService)
        script_service.serve_for_session.return_value = ScriptServeResult(outcome="unavailable")
        with TestClient(app=_app(script_service, AsyncMock())) as client:
            response = client.get(
                "/v1/provisioning/scripts/comfyui/v1.2.3",
                params={"session": str(uuid4()), "token": "tok"},
            )

        assert response.status_code == HTTP_502_BAD_GATEWAY
        assert response.json()["error"] == "provisioning_script_unavailable"

    def test_query_cannot_influence_which_variant_or_ref_is_served(self) -> None:
        """repo/path are never taken from the request — variant/ref only come from
        the path template; extra query params are simply ignored by the route."""
        script_service = AsyncMock(spec=ProvisioningScriptService)
        script_service.serve_for_session.return_value = ScriptServeResult(
            outcome="ok",
            script=ResolvedScript(content="body", sha256="c" * 64, cache_hit=True),
        )
        with TestClient(app=_app(script_service, AsyncMock())) as client:
            client.get(
                "/v1/provisioning/scripts/comfyui/v1.2.3",
                params={
                    "session": str(uuid4()),
                    "token": "tok",
                    "repo": "some/other-repo",
                    "path": "evil/path.sh",
                },
            )

        call = script_service.serve_for_session.await_args
        assert call.kwargs["variant"] == "comfyui"
        assert call.kwargs["ref"] == "v1.2.3"
        assert "repo" not in call.kwargs
        assert "path" not in call.kwargs


class TestWebhookRoute:
    def _payload(self, **overrides: object) -> dict[str, object]:
        return {
            "action": "continue",
            "manifest": "/path/to/manifest.yaml",
            "error": "provisioning failed after all retries",
            "container_id": "50885024",
            "timestamp": "2026-09-13T12:36:41",
        } | overrides

    def test_valid_call_returns_200_ok_envelope(self) -> None:
        webhook_service = AsyncMock(spec=ProvisioningWebhookService)
        webhook_service.handle_failure.return_value = HTTP_200_OK
        session_id = uuid4()
        with TestClient(app=_app(AsyncMock(), webhook_service)) as client:
            response = client.post(
                f"/v1/provisioning/webhook/{session_id}",
                params={"token": "tok"},
                json=self._payload(),
            )

        assert response.status_code == HTTP_200_OK
        assert response.json() == {"ok": True}
        webhook_service.handle_failure.assert_awaited_once()
        call = webhook_service.handle_failure.await_args
        assert call.kwargs["session_id"] == session_id
        assert call.kwargs["token"] == "tok"

    def test_wrong_token_is_401(self) -> None:
        webhook_service = AsyncMock(spec=ProvisioningWebhookService)
        webhook_service.handle_failure.return_value = HTTP_401_UNAUTHORIZED
        with TestClient(app=_app(AsyncMock(), webhook_service)) as client:
            response = client.post(
                f"/v1/provisioning/webhook/{uuid4()}",
                params={"token": "wrong"},
                json=self._payload(),
            )

        assert response.status_code == HTTP_401_UNAUTHORIZED

    def test_malformed_body_is_400(self) -> None:
        webhook_service = AsyncMock(spec=ProvisioningWebhookService)
        with TestClient(app=_app(AsyncMock(), webhook_service)) as client:
            response = client.post(
                f"/v1/provisioning/webhook/{uuid4()}",
                params={"token": "tok"},
                json={"action": "continue"},  # missing required fields
            )

        assert response.status_code == HTTP_400_BAD_REQUEST
        webhook_service.handle_failure.assert_not_awaited()

    def test_unknown_extra_field_decodes_fine(self) -> None:
        webhook_service = AsyncMock(spec=ProvisioningWebhookService)
        webhook_service.handle_failure.return_value = HTTP_200_OK
        with TestClient(app=_app(AsyncMock(), webhook_service)) as client:
            response = client.post(
                f"/v1/provisioning/webhook/{uuid4()}",
                params={"token": "tok"},
                json=self._payload(new_field_from_upstream="whatever"),
            )

        assert response.status_code == HTTP_200_OK
