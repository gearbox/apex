"""Tests for CloudflareTunnelClient."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.api.services.cloudflare.client import CloudflareTunnelClient
from src.api.services.cloudflare.exceptions import (
    CloudflareError,
    DNSRecordError,
    TunnelConfigError,
    TunnelCreationError,
    TunnelDeletionError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int, body: dict[str, object]) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.content = json.dumps(body).encode()
    return resp


def _make_client(mock_http: AsyncMock) -> CloudflareTunnelClient:
    return CloudflareTunnelClient(
        http_client=mock_http,
        api_token="test-token",
        account_id="acc-123",
        zone_id="zone-456",
        tunnel_domain="gpu.cloudin.space",
    )


_CREATE_TUNNEL_OK = {
    "success": True,
    "errors": [],
    "messages": [],
    "result": {"id": "tunnel-abc", "name": "gpu-session-01jf8x3k", "token": "tok-xyz"},
}

_CREATE_DNS_OK = {
    "success": True,
    "errors": [],
    "messages": [],
    "result": {
        "id": "dns-def",
        "name": "01jf8x3k.gpu.cloudin.space",
        "type": "CNAME",
        "content": "tunnel-abc.cfargotunnel.com",
    },
}

_CONFIGURE_OK = {"success": True, "errors": [], "messages": [], "result": {}}

_LIST_OK = {
    "success": True,
    "errors": [],
    "messages": [],
    "result": [
        {"id": "tunnel-abc", "name": "gpu-session-01jf8x3k", "status": "healthy"},
        {"id": "tunnel-def", "name": "gpu-session-02xx", "status": "inactive"},
    ],
}


# ---------------------------------------------------------------------------
# create_tunnel
# ---------------------------------------------------------------------------


class TestCreateTunnel:
    async def test_create_tunnel_success(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_response(200, _CREATE_TUNNEL_OK))
        client = _make_client(mock_http)

        tunnel_id, token = await client.create_tunnel("gpu-session-01jf8x3k")

        assert tunnel_id == "tunnel-abc"
        assert token == "tok-xyz"
        mock_http.post.assert_called_once()
        call_url = mock_http.post.call_args[0][0]
        assert "/accounts/acc-123/cfd_tunnel" in call_url

    async def test_create_tunnel_http_error(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            return_value=_mock_response(
                400, {"success": False, "errors": [{"message": "bad"}], "messages": []}
            )
        )
        client = _make_client(mock_http)

        with pytest.raises(TunnelCreationError) as exc_info:
            await client.create_tunnel("gpu-session-01jf8x3k")

        assert exc_info.value.status_code == 400

    async def test_create_tunnel_api_error_in_body(self) -> None:
        body = {
            "success": False,
            "errors": [{"message": "quota exceeded"}],
            "messages": [],
            "result": None,
        }
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_response(200, body))
        client = _make_client(mock_http)

        with pytest.raises(TunnelCreationError) as exc_info:
            await client.create_tunnel("gpu-session-01jf8x3k")

        # The HTTP status should be preserved even when API-level error has HTTP 2xx
        assert exc_info.value.status_code == 200


# ---------------------------------------------------------------------------
# configure_tunnel_ingress
# ---------------------------------------------------------------------------


class TestConfigureTunnelIngress:
    async def test_configure_tunnel_ingress_success(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.put = AsyncMock(return_value=_mock_response(200, _CONFIGURE_OK))
        client = _make_client(mock_http)

        await client.configure_tunnel_ingress("tunnel-abc", "01jf8x3k.gpu.cloudin.space", 18188)

        mock_http.put.assert_called_once()
        call_url = mock_http.put.call_args[0][0]
        assert "/cfd_tunnel/tunnel-abc/configurations" in call_url

    async def test_configure_tunnel_ingress_error(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.put = AsyncMock(
            return_value=_mock_response(500, {"success": False, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        with pytest.raises(TunnelConfigError) as exc_info:
            await client.configure_tunnel_ingress("tunnel-abc", "host.example.com", 18188)

        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# create_dns_record
# ---------------------------------------------------------------------------


class TestCreateDnsRecord:
    async def test_create_dns_record_success(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_response(200, _CREATE_DNS_OK))
        client = _make_client(mock_http)

        record_id = await client.create_dns_record("01jf8x3k.gpu.cloudin.space", "tunnel-abc")

        assert record_id == "dns-def"
        call_url = mock_http.post.call_args[0][0]
        assert "/zones/zone-456/dns_records" in call_url

    async def test_create_dns_record_http_error(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            return_value=_mock_response(403, {"success": False, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        with pytest.raises(DNSRecordError) as exc_info:
            await client.create_dns_record("host.example.com", "tunnel-abc")

        assert exc_info.value.status_code == 403

    async def test_create_dns_record_api_error_in_body(self) -> None:
        body = {
            "success": False,
            "errors": [{"message": "zone not found"}],
            "messages": [],
            "result": None,
        }
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_response(200, body))
        client = _make_client(mock_http)

        with pytest.raises(DNSRecordError) as exc_info:
            await client.create_dns_record("host.example.com", "tunnel-abc")

        assert exc_info.value.status_code == 200


# ---------------------------------------------------------------------------
# delete_tunnel
# ---------------------------------------------------------------------------


class TestDeleteTunnel:
    async def test_delete_tunnel_success(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.delete = AsyncMock(
            return_value=_mock_response(200, {"success": True, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        await client.delete_tunnel("tunnel-abc")

        mock_http.delete.assert_called_once()
        call_url = mock_http.delete.call_args[0][0]
        assert "/cfd_tunnel/tunnel-abc" in call_url

    async def test_delete_tunnel_idempotent_on_404(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.delete = AsyncMock(
            return_value=_mock_response(404, {"success": False, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        # Should not raise
        await client.delete_tunnel("tunnel-abc")

    async def test_delete_tunnel_error_on_non_404(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.delete = AsyncMock(
            return_value=_mock_response(500, {"success": False, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        with pytest.raises(TunnelDeletionError):
            await client.delete_tunnel("tunnel-abc")


# ---------------------------------------------------------------------------
# delete_dns_record
# ---------------------------------------------------------------------------


class TestDeleteDnsRecord:
    async def test_delete_dns_record_success(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.delete = AsyncMock(
            return_value=_mock_response(200, {"success": True, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        await client.delete_dns_record("dns-def")

        call_url = mock_http.delete.call_args[0][0]
        assert "/dns_records/dns-def" in call_url

    async def test_delete_dns_record_idempotent_on_404(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.delete = AsyncMock(
            return_value=_mock_response(404, {"success": False, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        # Should not raise
        await client.delete_dns_record("dns-def")

    async def test_delete_dns_record_error_on_non_404(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.delete = AsyncMock(
            return_value=_mock_response(403, {"success": False, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        with pytest.raises(DNSRecordError):
            await client.delete_dns_record("dns-def")


# ---------------------------------------------------------------------------
# list_tunnels
# ---------------------------------------------------------------------------


class TestListTunnels:
    async def test_list_tunnels_with_prefix(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=_mock_response(200, _LIST_OK))
        client = _make_client(mock_http)

        entries = await client.list_tunnels(name_prefix="gpu-session-")

        assert len(entries) == 2
        assert entries[0].id == "tunnel-abc"
        assert entries[1].name == "gpu-session-02xx"

        call_kwargs = mock_http.get.call_args[1]
        assert call_kwargs["params"]["name"] == "gpu-session-"
        assert call_kwargs["params"]["is_deleted"] == "false"

    async def test_list_tunnels_without_prefix(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=_mock_response(200, _LIST_OK))
        client = _make_client(mock_http)

        await client.list_tunnels()

        call_kwargs = mock_http.get.call_args[1]
        assert "name" not in call_kwargs["params"]

    async def test_list_tunnels_http_error(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(
            return_value=_mock_response(500, {"success": False, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        with pytest.raises(CloudflareError):
            await client.list_tunnels()

    async def test_list_tunnels_api_error_in_body(self) -> None:
        body = {
            "success": False,
            "errors": [{"message": "permission denied"}],
            "messages": [],
            "result": [],
        }
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=_mock_response(200, body))
        client = _make_client(mock_http)

        with pytest.raises(CloudflareError):
            await client.list_tunnels()

    async def test_list_tunnels_decode_error(self) -> None:
        """Non-JSON response (e.g. proxy HTML error page) raises CloudflareError."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.is_success = True
        resp.content = b"<html><body>502 Bad Gateway</body></html>"

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=resp)
        client = _make_client(mock_http)

        with pytest.raises(CloudflareError):
            await client.list_tunnels()


# ---------------------------------------------------------------------------
# create_session_tunnel (high-level orchestration)
# ---------------------------------------------------------------------------


class TestCreateSessionTunnel:
    async def test_create_session_tunnel_orchestrates_all_steps(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            side_effect=[
                _mock_response(200, _CREATE_TUNNEL_OK),  # create_tunnel
                _mock_response(200, _CREATE_DNS_OK),  # create_dns_record
            ]
        )
        mock_http.put = AsyncMock(return_value=_mock_response(200, _CONFIGURE_OK))
        client = _make_client(mock_http)

        tunnel_id, token, dns_record_id, hostname = await client.create_session_tunnel(
            "01jf8x3k", 18188
        )

        assert tunnel_id == "tunnel-abc"
        assert token == "tok-xyz"
        assert dns_record_id == "dns-def"
        assert hostname == "01jf8x3k.gpu.cloudin.space"

        # Verify call order: post (create) → put (configure) → post (DNS)
        assert mock_http.post.call_count == 2
        assert mock_http.put.call_count == 1

        create_url = mock_http.post.call_args_list[0][0][0]
        configure_url = mock_http.put.call_args[0][0]
        dns_url = mock_http.post.call_args_list[1][0][0]

        assert "/cfd_tunnel" in create_url
        assert "/configurations" in configure_url
        assert "/dns_records" in dns_url

    async def test_create_session_tunnel_rolls_back_on_configure_failure(self) -> None:
        """If configure_tunnel_ingress fails, the created tunnel must be deleted."""
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_response(200, _CREATE_TUNNEL_OK))
        mock_http.put = AsyncMock(
            return_value=_mock_response(500, {"success": False, "errors": [], "messages": []})
        )
        mock_http.delete = AsyncMock(
            return_value=_mock_response(200, {"success": True, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        with pytest.raises(TunnelConfigError):
            await client.create_session_tunnel("01jf8x3k", 18188)

        # Rollback: tunnel should have been deleted
        mock_http.delete.assert_called_once()
        delete_url = mock_http.delete.call_args[0][0]
        assert "/cfd_tunnel/tunnel-abc" in delete_url

    async def test_create_session_tunnel_rolls_back_on_dns_failure(self) -> None:
        """If create_dns_record fails, the created tunnel must be deleted."""
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(
            side_effect=[
                _mock_response(200, _CREATE_TUNNEL_OK),  # create_tunnel OK
                _mock_response(500, {"success": False, "errors": [], "messages": []}),  # DNS fails
            ]
        )
        mock_http.put = AsyncMock(return_value=_mock_response(200, _CONFIGURE_OK))
        mock_http.delete = AsyncMock(
            return_value=_mock_response(200, {"success": True, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        with pytest.raises(DNSRecordError):
            await client.create_session_tunnel("01jf8x3k", 18188)

        mock_http.delete.assert_called_once()
        delete_url = mock_http.delete.call_args[0][0]
        assert "/cfd_tunnel/tunnel-abc" in delete_url

    async def test_create_session_tunnel_rollback_swallows_cleanup_errors(self) -> None:
        """If the rollback delete itself fails, the original error must still propagate."""
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_mock_response(200, _CREATE_TUNNEL_OK))
        mock_http.put = AsyncMock(
            return_value=_mock_response(500, {"success": False, "errors": [], "messages": []})
        )
        # Delete itself fails — should be suppressed so the original error surfaces
        mock_http.delete = AsyncMock(
            return_value=_mock_response(500, {"success": False, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        # Original TunnelConfigError must propagate, not the delete failure
        with pytest.raises(TunnelConfigError):
            await client.create_session_tunnel("01jf8x3k", 18188)

        mock_http.delete.assert_called_once()


# ---------------------------------------------------------------------------
# delete_session_tunnel (high-level)
# ---------------------------------------------------------------------------


class TestDeleteSessionTunnel:
    async def test_delete_session_tunnel_deletes_both(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.delete = AsyncMock(
            return_value=_mock_response(200, {"success": True, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        await client.delete_session_tunnel("tunnel-abc", "dns-def")

        assert mock_http.delete.call_count == 2
        dns_call_url = mock_http.delete.call_args_list[0][0][0]
        tunnel_call_url = mock_http.delete.call_args_list[1][0][0]

        assert "/dns_records/dns-def" in dns_call_url
        assert "/cfd_tunnel/tunnel-abc" in tunnel_call_url

    async def test_delete_session_tunnel_idempotent_on_404(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.delete = AsyncMock(
            return_value=_mock_response(404, {"success": False, "errors": [], "messages": []})
        )
        client = _make_client(mock_http)

        # Should not raise even if both resources are already gone
        await client.delete_session_tunnel("tunnel-abc", "dns-def")
        assert mock_http.delete.call_count == 2
