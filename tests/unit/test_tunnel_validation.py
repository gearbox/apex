"""Tests for tunnel hostname SSRF validation."""

from __future__ import annotations

import pytest

from src.api.services.generation.tunnel_validation import (
    InvalidTunnelHostnameError,
    validate_tunnel_hostname,
)

SUFFIX = ".gpu-domain.com"
PREFIX = "gpu-"


def _ok(hostname: str) -> None:
    validate_tunnel_hostname(hostname, allowed_suffix=SUFFIX, allowed_prefix=PREFIX)


def _bad(hostname: str) -> None:
    with pytest.raises(InvalidTunnelHostnameError):
        validate_tunnel_hostname(hostname, allowed_suffix=SUFFIX, allowed_prefix=PREFIX)


class TestValidHosts:
    def test_standard_gpu_prefix_hostname(self) -> None:
        _ok("gpu-abc123.gpu-domain.com")

    def test_hyphenated_session_id(self) -> None:
        _ok("gpu-my-node-01.gpu-domain.com")

    def test_realistic_session_id(self) -> None:
        _ok("gpu-01jf8x3k.gpu-domain.com")


class TestInvalidHosts:
    def test_empty_hostname(self) -> None:
        _bad("")

    def test_wrong_suffix(self) -> None:
        _bad("abc123.evil.com")

    def test_uppercase_letter(self) -> None:
        _bad("ABC.gpu-domain.com")

    def test_url_meta_question_mark(self) -> None:
        # endswith check would pass but charset check must catch this
        _bad("evil.com?.gpu-domain.com")

    def test_url_meta_at_sign(self) -> None:
        _bad("user@evil.com.gpu-domain.com")

    def test_url_meta_hash(self) -> None:
        _bad("evil.com#.gpu-domain.com")

    def test_consecutive_dots(self) -> None:
        _bad("abc..gpu-domain.com")

    def test_empty_allowed_suffix(self) -> None:
        with pytest.raises(InvalidTunnelHostnameError):
            validate_tunnel_hostname("gpu-abc123.gpu-domain.com", allowed_suffix="")

    def test_bare_ip_address(self) -> None:
        _bad("192.168.1.1")

    def test_localhost(self) -> None:
        _bad("localhost")

    def test_old_two_level_format_rejected(self) -> None:
        """Regression: old format {id}.gpu.gpu-domain.com must be rejected — it lacks the gpu- prefix."""
        _bad("01jf8x3k.gpu.gpu-domain.com")

    def test_missing_gpu_prefix_rejected(self) -> None:
        """Hostname ending correctly but missing gpu- prefix must be rejected."""
        _bad("01jf8x3k.gpu-domain.com")

    def test_wrong_prefix_rejected(self) -> None:
        """Prefix other than gpu- must be rejected."""
        _bad("evil-01jf8x3k.gpu-domain.com")

    def test_wrong_suffix_with_correct_prefix_rejected(self) -> None:
        """Correct gpu- prefix but wrong suffix (cross-tenant SSRF guard)."""
        _bad("gpu-01jf8x3k.attacker.com")
