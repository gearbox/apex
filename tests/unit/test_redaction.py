"""Tests for safe logging of untrusted upstream details."""

from __future__ import annotations

import pytest

from src.api.utils.redaction import redact_secrets


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
