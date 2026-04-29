"""Tests for tunnel hostname SSRF validation."""

from __future__ import annotations

import pytest

from src.api.services.generation.tunnel_validation import (
    InvalidTunnelHostnameError,
    validate_tunnel_hostname,
)

SUFFIX = ".gpu.cloudin.space"


def _ok(hostname: str) -> None:
    validate_tunnel_hostname(hostname, allowed_suffix=SUFFIX)


def _bad(hostname: str) -> None:
    with pytest.raises(InvalidTunnelHostnameError):
        validate_tunnel_hostname(hostname, allowed_suffix=SUFFIX)


class TestValidHosts:
    def test_standard_subdomain(self) -> None:
        _ok("abc123.gpu.cloudin.space")

    def test_hyphenated_subdomain(self) -> None:
        _ok("my-node-01.gpu.cloudin.space")

    def test_deep_subdomain(self) -> None:
        _ok("a.b.gpu.cloudin.space")


class TestInvalidHosts:
    def test_empty_hostname(self) -> None:
        _bad("")

    def test_wrong_suffix(self) -> None:
        _bad("abc123.evil.com")

    def test_uppercase_letter(self) -> None:
        _bad("ABC.gpu.cloudin.space")

    def test_url_meta_question_mark(self) -> None:
        # endswith check would pass but charset check must catch this
        _bad("evil.com?.gpu.cloudin.space")

    def test_url_meta_at_sign(self) -> None:
        _bad("user@evil.com.gpu.cloudin.space")

    def test_url_meta_hash(self) -> None:
        _bad("evil.com#.gpu.cloudin.space")

    def test_consecutive_dots(self) -> None:
        _bad("abc..gpu.cloudin.space")

    def test_empty_allowed_suffix(self) -> None:
        with pytest.raises(InvalidTunnelHostnameError):
            validate_tunnel_hostname("abc123.gpu.cloudin.space", allowed_suffix="")

    def test_bare_ip_address(self) -> None:
        _bad("192.168.1.1")

    def test_localhost(self) -> None:
        _bad("localhost")
