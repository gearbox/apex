"""Tests for get_real_ip trusted-proxy-aware client IP resolution (C4).

X-Forwarded-For is fully client-controlled unless a trusted proxy is known to
strip/overwrite it. These tests pin down that get_real_ip only trusts the
header configured via settings.trusted_ip_header, and — critically — that the
leftmost X-Forwarded-For entry (the client-supplied one) never wins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from litestar.testing import RequestFactory

from src.api.middleware.rate_limit import get_real_ip
from src.core.config import Settings

if TYPE_CHECKING:
    from litestar import Request

_factory = RequestFactory()


def _connection_host(request: Request[Any, Any, Any]) -> str:
    """RequestFactory-built requests always populate client — narrow for mypy."""
    assert request.client is not None
    return request.client.host


class TestGetRealIpNoneMode:
    """trusted_ip_header='none' — the default; only the ASGI connection address."""

    def test_ignores_xff_and_uses_connection_address(self) -> None:
        settings = Settings(trusted_ip_header="none")
        request = _factory.get("/", headers={"X-Forwarded-For": "9.9.9.9"})
        assert get_real_ip(request, settings) == _connection_host(request)

    def test_spoofed_xff_does_not_change_result(self) -> None:
        settings = Settings(trusted_ip_header="none")
        baseline = get_real_ip(_factory.get("/"), settings)
        spoofed = get_real_ip(_factory.get("/", headers={"X-Forwarded-For": "1.2.3.4"}), settings)
        assert baseline == spoofed


class TestGetRealIpCloudflareMode:
    """trusted_ip_header='cf-connecting-ip' — Cloudflare strips/overwrites this header."""

    def test_uses_cf_connecting_ip(self) -> None:
        settings = Settings(trusted_ip_header="cf-connecting-ip")
        request = _factory.get("/", headers={"CF-Connecting-IP": "203.0.113.9"})
        assert get_real_ip(request, settings) == "203.0.113.9"

    def test_falls_back_to_connection_address_when_header_absent(self) -> None:
        settings = Settings(trusted_ip_header="cf-connecting-ip")
        request = _factory.get("/")
        assert get_real_ip(request, settings) == _connection_host(request)

    def test_ignores_xff_entirely(self) -> None:
        settings = Settings(trusted_ip_header="cf-connecting-ip")
        request = _factory.get(
            "/",
            headers={"X-Forwarded-For": "9.9.9.9", "CF-Connecting-IP": "203.0.113.9"},
        )
        assert get_real_ip(request, settings) == "203.0.113.9"


class TestGetRealIpXForwardedForMode:
    """trusted_ip_header='x-forwarded-for' — generic reverse proxy, rightmost-minus-hops."""

    def test_zero_hops_uses_rightmost_entry(self) -> None:
        settings = Settings(trusted_ip_header="x-forwarded-for", trusted_proxy_hops=0)
        request = _factory.get(
            "/", headers={"X-Forwarded-For": "client-spoofed, 10.0.0.1, 203.0.113.5"}
        )
        assert get_real_ip(request, settings) == "203.0.113.5"

    def test_spoofed_leftmost_entry_does_not_change_result(self) -> None:
        """The leftmost entry is client-controlled — only the rightmost (the
        entry appended by our own trusted proxy) may determine the result."""
        settings = Settings(trusted_ip_header="x-forwarded-for", trusted_proxy_hops=0)
        base = get_real_ip(
            _factory.get("/", headers={"X-Forwarded-For": "1.1.1.1, 203.0.113.5"}),
            settings,
        )
        spoofed = get_real_ip(
            _factory.get("/", headers={"X-Forwarded-For": "9.9.9.9, 203.0.113.5"}),
            settings,
        )
        assert base == spoofed == "203.0.113.5"

    def test_one_hop_selects_second_from_right(self) -> None:
        settings = Settings(trusted_ip_header="x-forwarded-for", trusted_proxy_hops=1)
        request = _factory.get(
            "/", headers={"X-Forwarded-For": "client, real-client-ip, edge-proxy"}
        )
        assert get_real_ip(request, settings) == "real-client-ip"

    def test_hops_exceeding_entry_count_clamps_to_leftmost(self) -> None:
        settings = Settings(trusted_ip_header="x-forwarded-for", trusted_proxy_hops=5)
        request = _factory.get("/", headers={"X-Forwarded-For": "a, b, c"})
        assert get_real_ip(request, settings) == "a"

    def test_falls_back_to_connection_address_when_header_absent(self) -> None:
        settings = Settings(trusted_ip_header="x-forwarded-for")
        request = _factory.get("/")
        assert get_real_ip(request, settings) == _connection_host(request)
