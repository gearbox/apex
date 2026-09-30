"""W2-e — forgot-password takes the recorded IP and the product from DI, not from headers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from litestar.testing import RequestFactory

from src.api.dependencies.request_context import provide_request_context
from src.api.routes.auth import AuthController
from src.core.config import Settings
from src.core.product_registry import SYNTHARA_CONFIG

pytestmark = pytest.mark.unit


def _kwargs(send: AsyncMock) -> dict[str, Any]:
    assert send.await_args is not None
    return dict(send.await_args.kwargs)


_SPOOFED = "6.6.6.6"


async def _forgot_password(*, settings: Settings, headers: dict[str, str]) -> tuple[AsyncMock, str]:
    request = RequestFactory().post("/v1/auth/forgot-password", headers=headers)
    assert request.client is not None
    context = provide_request_context(request, settings)
    service = AsyncMock()
    data = MagicMock()
    data.email = "u@example.com"

    response = await AuthController.forgot_password.fn(
        MagicMock(),
        data=data,
        session=AsyncMock(),
        email_verification_service=service,
        product_config=SYNTHARA_CONFIG,
        request_context=context,
    )

    assert response.status_code == 200
    return service.send_password_reset_email, request.client.host


async def test_w2_e_spoofed_forwarded_for_from_an_untrusted_peer_is_not_recorded() -> None:
    send, peer = await _forgot_password(
        settings=Settings(trusted_ip_header="none"),
        headers={"X-Forwarded-For": _SPOOFED},
    )

    send.assert_awaited_once()
    assert _kwargs(send)["ip_address"] == peer
    assert _kwargs(send)["ip_address"] != _SPOOFED


async def test_w2_e_records_the_ip_the_trusted_proxy_header_resolves_to() -> None:
    send, _ = await _forgot_password(
        settings=Settings(trusted_ip_header="cf-connecting-ip"),
        headers={"CF-Connecting-IP": "203.0.113.9", "X-Forwarded-For": _SPOOFED},
    )

    assert _kwargs(send)["ip_address"] == "203.0.113.9"


async def test_w2_e_the_reset_lookup_is_scoped_to_the_requesting_product() -> None:
    send, _ = await _forgot_password(settings=Settings(trusted_ip_header="none"), headers={})

    assert _kwargs(send)["product_id"] == "synthara"
