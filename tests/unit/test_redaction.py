"""Tests for safe logging of untrusted upstream details.

T1 (round-3 remediation): the redactor is a tokenizer + key-based structural walk,
not a regex — see src/api/utils/redaction.py's module docstring for the three
layers. Every row of the T1 finding's leak-form table has a dedicated test below
with its expected output spelled out.
"""

from __future__ import annotations

import pytest

from src.api.utils.redaction import (
    redact_known_secrets,
    redact_secrets,
    redact_secrets_mapping,
)

# ---------------------------------------------------------------------------
# T1 table: every leak form the round-3 review demonstrated against the old
# regex-based redactor, with the exact expected output under the new design.
# ---------------------------------------------------------------------------


class TestT1LeakFormsTable:
    def test_single_quoted_env_assignment(self) -> None:
        assert (
            redact_secrets("ACS_GITHUB_TOKEN='ghp_SECRETVALUE123'", max_length=500)
            == "ACS_GITHUB_TOKEN='[REDACTED]'"
        )

    def test_declare_dash_x_double_quoted_env_assignment(self) -> None:
        assert (
            redact_secrets('declare -x ACS_GITHUB_TOKEN="ghp_SECRETVALUE123"', max_length=500)
            == 'declare -x ACS_GITHUB_TOKEN="[REDACTED]"'
        )

    def test_authorization_bearer_header(self) -> None:
        assert (
            redact_secrets("Authorization: Bearer ghp_SECRETVALUE123", max_length=500)
            == "Authorization: Bearer [REDACTED]"
        )

    def test_authorization_token_scheme_header(self) -> None:
        assert (
            redact_secrets("Authorization: token ghp_SECRET", max_length=500)
            == "Authorization: token [REDACTED]"
        )

    def test_json_token_field_as_text(self) -> None:
        assert (
            redact_secrets('{"token": "ghp_SECRETVALUE123"}', max_length=500)
            == '{"token": "[REDACTED]"}'
        )

    def test_json_env_mapping_key_redacted_wholesale(self) -> None:
        result = redact_secrets_mapping({"env": {"ACS_GITHUB_TOKEN": "ghp_SECRET"}}, max_length=500)
        assert result == {"env": {"ACS_GITHUB_TOKEN": "[REDACTED]"}}

    def test_cf_tunnel_token_env_form_still_covered(self) -> None:
        assert (
            redact_secrets("CF_TUNNEL_TOKEN=eyJhSECRET", max_length=500)
            == "CF_TUNNEL_TOKEN=[REDACTED]"
        )

    def test_url_query_token_and_session_stripped(self) -> None:
        redacted = redact_secrets("https://a.test/v1/x?session=abc&token=SECRET", max_length=500)
        assert redacted == "https://a.test/v1/x"

    def test_newline_does_not_leak_the_next_line_as_a_value(self) -> None:
        """The old regex's `\\s*` after `=` spanned newlines and redacted an
        unrelated value on the next line. Whitespace is a hard token separator
        now, so the next line is untouched and the empty same-line value is left
        as-is rather than inventing a redaction for nothing."""
        assert redact_secrets("my_key=\nvalue", max_length=500) == "my_key=\nvalue"


class TestAuthorizationHeaderLineBoundaries:
    """Branch coverage for _redact_remainder_of_line's line-boundary handling."""

    def test_marker_at_end_of_line_has_nothing_to_redact(self) -> None:
        assert (
            redact_secrets("Authorization:\nunrelated next line", max_length=500)
            == "Authorization:\nunrelated next line"
        )

    def test_scheme_word_at_end_of_line_has_nothing_after_it(self) -> None:
        assert (
            redact_secrets("Authorization: Bearer\nunrelated next line", max_length=500)
            == "Authorization: Bearer\nunrelated next line"
        )

    def test_multiple_tokens_on_the_same_line_are_all_redacted(self) -> None:
        redacted = redact_secrets("Authorization: Bearer secret-one secret-two", max_length=500)
        assert redacted == "Authorization: Bearer [REDACTED] [REDACTED]"


# ---------------------------------------------------------------------------
# redact_secrets() — general behavior, including cases that must keep passing.
# ---------------------------------------------------------------------------


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
    """Not just literal token=/session= — any KEY=value where KEY
    case-insensitively contains token/secret/key/password/etc."""
    redacted = redact_secrets(env_line, max_length=500)

    assert "abc123secret" not in redacted
    assert "hf_abc123secret" not in redacted
    assert "civ_abc123secret" not in redacted
    assert "ghp_abc123secret" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_secrets_leaves_unrelated_key_value_pairs_alone() -> None:
    redacted = redact_secrets("model=wan_2.2_i2v mode=full", max_length=500)

    assert redacted == "model=wan_2.2_i2v mode=full"


def test_redact_secrets_leaves_a_punctuation_only_value_alone() -> None:
    """A value that strips down to nothing (all punctuation, no core) is left
    as-is rather than inventing a redaction for something that isn't there."""
    assert redact_secrets("KEY=,", max_length=500) == "KEY=,"


def test_redact_secrets_handles_key_space_equals_space_value() -> None:
    """`KEY = value` split across three whitespace-separated tokens."""
    assert redact_secrets("token = abc123secret", max_length=500) == "token = [REDACTED]"


def test_redact_secrets_handles_quoted_value_spanning_two_tokens() -> None:
    """`KEY="a b"` — the closing quote is in a later whitespace-split token."""
    assert redact_secrets('KEY="a secretvalue"', max_length=500) == 'KEY="[REDACTED]"'


def test_redact_secrets_is_idempotent() -> None:
    value = (
        'Authorization: Bearer ghp_X\nCF_TUNNEL_TOKEN="abc123secret"\nmodel=wan_2.2_i2v mode=full'
    )
    once = redact_secrets(value, max_length=500)
    twice = redact_secrets(once, max_length=500)

    assert once == twice


def test_no_regex_module_used() -> None:
    """T1 guard: this module must not use `re` at all — see the finding's
    explanation of why shape-matching via regex is not tunable."""
    import ast
    from pathlib import Path

    source = Path("src/api/utils/redaction.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "re" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "re"


# ---------------------------------------------------------------------------
# Layer 1 — known-secret exact replacement.
# ---------------------------------------------------------------------------


class TestKnownSecrets:
    def test_catches_a_token_that_matches_no_shape_at_all(self) -> None:
        """A bare secret value with no KEY=, no quotes, no URL — only exact
        known-value matching (Layer 1) can catch this; Layers 2/3 cannot."""
        secret = "ghp_totallyBareSecretNoShapeAtAll"
        redacted = redact_secrets(
            f"the node said: {secret} and gave up",
            max_length=500,
            known_secrets=frozenset({secret}),
        )
        assert secret not in redacted
        assert "[REDACTED]" in redacted

    def test_value_shorter_than_minimum_length_is_left_alone(self) -> None:
        short = "abc123"  # < 8 chars
        text = f"value is {short} here"
        redacted = redact_secrets(text, max_length=500, known_secrets=frozenset({short}))
        assert redacted == text

    def test_redact_known_secrets_only_does_layer_1(self) -> None:
        secret = "ghp_exactSecretValue123"
        result = redact_known_secrets(f"prefix-{secret}-suffix", frozenset({secret}))
        assert secret not in result
        assert "[REDACTED]" in result

    def test_redact_known_secrets_leaves_unrelated_text_untouched(self) -> None:
        result = redact_known_secrets("bundle_version=260101-01", frozenset({"unrelated-secret"}))
        assert result == "bundle_version=260101-01"


# ---------------------------------------------------------------------------
# redact_secrets_mapping() — Layer 2 (structural) + depth/node bounds (T2/T10).
# ---------------------------------------------------------------------------


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

    def test_mapping_key_sensitive_at_depth_three(self) -> None:
        result = redact_secrets_mapping(
            {"a": {"b": {"c": {"api_key": "sk-abc123secret"}}}}, max_length=500
        )
        assert result == {"a": {"b": {"c": {"api_key": "[REDACTED]"}}}}

    def test_list_of_mappings(self) -> None:
        result = redact_secrets_mapping(
            {"errors": [{"token": "abc123secret"}, {"note": "ok"}]}, max_length=500
        )
        assert result == {"errors": [{"token": "[REDACTED]"}, {"note": "ok"}]}

    def test_sensitive_key_whose_value_is_a_dict_is_replaced_wholesale(self) -> None:
        """A sensitive key's value is redacted wholesale, never descended into —
        even when the value is itself a nested structure."""
        result = redact_secrets_mapping(
            {"credentials": {"user": "alice", "password": "hunter2"}}, max_length=500
        )
        assert result == {"credentials": "[REDACTED]"}

    def test_known_secrets_reach_nested_string_leaves(self) -> None:
        secret = "ghp_deeplyNestedBareSecret"
        result = redact_secrets_mapping(
            {"detail": {"cause": f"curl failed: {secret}"}},
            max_length=500,
            known_secrets=frozenset({secret}),
        )
        assert result is not None
        assert secret not in str(result)

    def test_idempotent(self) -> None:
        payload = {"detail": {"last_error": 'Authorization: Bearer ghp_X\ntoken="abc123secret"'}}
        once = redact_secrets_mapping(payload, max_length=500)
        twice = redact_secrets_mapping(once, max_length=500)
        assert once == twice


class TestDepthAndNodeBounds:
    def _nested(self, depth: int) -> dict[str, object]:
        value: dict[str, object] = {"leaf": "token=abc123secret"}
        for _ in range(depth):
            value = {"nested": value}
        return value

    def test_depth_1000_does_not_raise_and_truncates(self) -> None:
        payload = self._nested(1000)
        result = redact_secrets_mapping(payload, max_length=500)
        assert result is not None
        assert "[TRUNCATED: max depth]" in str(result)

    def test_depth_5000_does_not_raise_and_truncates(self) -> None:
        payload = self._nested(5000)
        result = redact_secrets_mapping(payload, max_length=500)
        assert result is not None
        assert "[TRUNCATED: max depth]" in str(result)

    def test_payload_exceeding_node_budget_is_bounded(self) -> None:
        wide = {f"key_{i}": "value" for i in range(10_000)}
        result = redact_secrets_mapping(wide, max_length=500, max_nodes=100)
        assert result is not None
        assert "[TRUNCATED: max nodes]" in result.values()

    def test_custom_max_depth_is_respected(self) -> None:
        payload = self._nested(5)
        result = redact_secrets_mapping(payload, max_length=500, max_depth=2)
        assert result is not None
        assert "[TRUNCATED: max depth]" in str(result)

    def test_list_exceeding_node_budget_is_bounded(self) -> None:
        wide_list = {"items": [f"value_{i}" for i in range(10_000)]}
        result = redact_secrets_mapping(wide_list, max_length=500, max_nodes=100)
        assert result is not None
        assert "[TRUNCATED: max nodes]" in result["items"]

    def test_zero_node_budget_truncates_the_top_level_value_itself(self) -> None:
        result = redact_secrets_mapping({"a": 1}, max_length=500, max_nodes=0)
        assert result == "[TRUNCATED: max nodes]"
