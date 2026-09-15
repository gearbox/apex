"""Tests for safe logging of untrusted upstream details."""

from __future__ import annotations

import pytest

from src.api.utils.redaction import redact_secrets, redact_secrets_mapping


@pytest.mark.parametrize(
    ("value", "forbidden"),
    [
        ("fetch https://apex.test/script?token=secret&session=session-id", "secret"),
        ("callback token=secret session=session-id", "secret"),
        ("fetch https://user:password@apex.test/script", "user:password@"),
    ],
)
def test_redact_secrets_removes_credentials_and_query_values(value: str, forbidden: str) -> None:
    redacted = redact_secrets(value, max_length=500)

    assert forbidden not in redacted
    assert "?" not in redacted


def test_redact_secrets_preserves_unicode_and_bounds_result() -> None:
    redacted = redact_secrets("ошибка " + "x" * 1_000, max_length=20)

    assert redacted.startswith("ошибка ")
    assert len(redacted) == 20


@pytest.mark.parametrize(
    "env_line",
    [
        "CF_TUNNEL_TOKEN=abc123secret",
        "ACS_GITHUB_TOKEN=ghp_abc123secret",
        "ACS_HF_TOKEN=hf_abc123secret",
        "ACS_CIVITAI_API_TOKEN=civ_abc123secret",
        "some_api_key=abc123secret",
        "DB_PASSWORD=abc123secret",
        "CLIENT_SECRET=abc123secret",
    ],
)
def test_redact_secrets_covers_any_token_secret_key_password_env_form(env_line: str) -> None:
    """S1: not just literal token=/session= — any KEY=value where KEY
    case-insensitively contains token/secret/key/password."""
    redacted = redact_secrets(env_line, max_length=500)

    assert "abc123secret" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_secrets_leaves_unrelated_key_value_pairs_alone() -> None:
    redacted = redact_secrets("model=wan_2.2_i2v mode=full", max_length=500)

    assert redacted == "model=wan_2.2_i2v mode=full"


class TestRedactSecretsMapping:
    def test_none_passes_through(self) -> None:
        assert redact_secrets_mapping(None, max_length=500) is None

    def test_redacts_a_top_level_string_leaf(self) -> None:
        result = redact_secrets_mapping(
            {"message": "fetch https://apex.test/x?token=secret"}, max_length=500
        )
        assert result is not None
        assert "secret" not in result["message"]

    def test_redacts_a_deeply_nested_string_leaf(self) -> None:
        """A URL/token can hide at any depth in an open-ended progress/plan/summary body."""
        result = redact_secrets_mapping(
            {"detail": {"last_error": {"cause": "token=nested-secret"}}}, max_length=500
        )
        assert result is not None
        assert "nested-secret" not in str(result)

    def test_redacts_string_leaves_inside_lists(self) -> None:
        result = redact_secrets_mapping({"errors": ["ok", "token=list-secret"]}, max_length=500)
        assert result is not None
        assert "list-secret" not in str(result)

    def test_non_string_leaves_pass_through_unchanged(self) -> None:
        result = redact_secrets_mapping(
            {"count": 3, "done": True, "ratio": 0.5, "missing": None}, max_length=500
        )
        assert result == {"count": 3, "done": True, "ratio": 0.5, "missing": None}
