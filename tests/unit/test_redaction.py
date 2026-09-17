"""Tests for safe logging of untrusted upstream details.

T1 (round-3 remediation): the redactor is a tokenizer + key-based structural walk,
not a regex — see src/api/utils/redaction.py's module docstring for the three
layers. Every row of the T1 finding's leak-form table has a dedicated test below
with its expected output spelled out.
"""

from __future__ import annotations

import json

import msgspec
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

    def test_cookie_header_still_fully_redacted(self) -> None:
        """U3: Cookie is on the exact-match rest-of-line allowlist, so unlike
        a general sensitive `KEY:` marker its whole remainder is redacted."""
        redacted = redact_secrets("Cookie: a=1; b=2", max_length=500)
        assert "1" not in redacted
        assert "2" not in redacted
        assert redacted.startswith("Cookie:")


# ---------------------------------------------------------------------------
# U1 (round-4): a URL is redacted wherever it sits in a token, not only when
# the whole token starts with http(s):// — PROVISIONING_SCRIPT and
# PROVISIONER_WEBHOOK_URL are exactly the KEY=<url> shape this closes.
# ---------------------------------------------------------------------------


class TestU1UrlInsideKeyValueAssignment:
    def test_provisioning_script_env_form_is_stripped(self) -> None:
        redacted = redact_secrets(
            "PROVISIONING_SCRIPT=https://apex.test/scripts/comfyui/v1.0.0"
            "?session=01a09ac1&token=REALSECRETTOKEN",
            max_length=500,
        )
        assert redacted == "PROVISIONING_SCRIPT=https://apex.test/scripts/comfyui/v1.0.0"
        assert "REALSECRETTOKEN" not in redacted

    def test_provisioner_webhook_url_env_form_is_stripped(self) -> None:
        redacted = redact_secrets(
            "PROVISIONER_WEBHOOK_URL=https://apex.test/webhook/01a09ac1?token=REALSECRETTOKEN",
            max_length=500,
        )
        assert redacted == "PROVISIONER_WEBHOOK_URL=https://apex.test/webhook/01a09ac1"
        assert "REALSECRETTOKEN" not in redacted

    def test_declare_dash_x_double_quoted_url_value_is_stripped_quotes_kept(self) -> None:
        redacted = redact_secrets(
            'declare -x PROVISIONING_SCRIPT="https://apex.test/?session=x&token=REALSECRETTOKEN"',
            max_length=500,
        )
        assert redacted == 'declare -x PROVISIONING_SCRIPT="https://apex.test/"'
        assert "REALSECRETTOKEN" not in redacted

    def test_env_prefixed_line_is_stripped(self) -> None:
        redacted = redact_secrets(
            "env: PROVISIONING_SCRIPT=https://a.test/x?token=REALSECRET", max_length=500
        )
        assert redacted == "env: PROVISIONING_SCRIPT=https://a.test/x"
        assert "REALSECRET" not in redacted

    def test_curl_command_bare_url_still_stripped(self) -> None:
        """Regression: this shape already worked before U1 — must keep working."""
        redacted = redact_secrets(
            "curl -fsSL https://a.test/x?token=REALSECRET -o /provisioning.sh", max_length=500
        )
        assert redacted == "curl -fsSL https://a.test/x -o /provisioning.sh"

    def test_single_quoted_url_value_is_stripped_quotes_kept(self) -> None:
        redacted = redact_secrets("PROVISIONING_SCRIPT='https://a.test/x?token=S'", max_length=500)
        assert redacted == "PROVISIONING_SCRIPT='https://a.test/x'"

    def test_url_value_with_no_query_is_unchanged(self) -> None:
        redacted = redact_secrets("PROVISIONING_SCRIPT=https://a.test/x", max_length=500)
        assert redacted == "PROVISIONING_SCRIPT=https://a.test/x"

    def test_two_url_tokens_in_free_text_are_both_stripped(self) -> None:
        redacted = redact_secrets(
            "see http://a.test/x?token=1 and http://b.test/y?token=2", max_length=500
        )
        assert "token=1" not in redacted
        assert "token=2" not in redacted
        assert redacted == "see http://a.test/x and http://b.test/y"

    def test_url_query_with_no_sensitive_parameter_still_fully_stripped(self) -> None:
        """The helper drops the whole query regardless of param names — kept."""
        redacted = redact_secrets("https://a.test/x?foo=bar", max_length=500)
        assert redacted == "https://a.test/x"

    def test_sensitive_key_url_value_is_redacted_wholesale_not_just_query(self) -> None:
        """A sensitive left-hand key still wins over the URL-only path: the
        whole value is redacted, not merely its query string."""
        redacted = redact_secrets("TOKEN=https://a.test/x?y=1", max_length=500)
        assert redacted == "TOKEN=[REDACTED]"

    def test_trailing_separator_after_url_is_preserved(self) -> None:
        """U8: a trailing ':' right after a redacted URL (e.g. a log line
        that quotes a failing URL before ': HTTP 404') must survive — losing
        it along with the query makes the redacted line harder to read."""
        redacted = redact_secrets(
            "fetch failed for https://a.test/x?token=X: HTTP 404", max_length=500
        )
        assert redacted == "fetch failed for https://a.test/x: HTTP 404"


# ---------------------------------------------------------------------------
# X3 (round-5): boundary-aware key matching lost glued compound keys —
# _GLUED_KEY_SUFFIXES only covered "apikey"/"authtoken"; every other
# separator-less compound (mytoken=, githubtoken=, clientsecret=) leaked.
# ---------------------------------------------------------------------------


class TestX3GluedCompoundKeys:
    @pytest.mark.parametrize(
        ("line", "secret"),
        [
            ("ACS_GITHUB_TOKEN=S1", "S1"),
            ("api_key=S3", "S3"),
            ("apikey=S4", "S4"),
            ("access_token=S10", "S10"),
            ("mytoken=S7", "S7"),
            ("githubtoken=S8", "S8"),
            ("clientsecret=S9", "S9"),
            ("authtoken=S11", "S11"),
        ],
    )
    def test_glued_and_separated_keys_all_redact(self, line: str, secret: str) -> None:
        redacted = redact_secrets(line, max_length=500)
        assert secret not in redacted
        assert "[REDACTED]" in redacted

    def test_monkey_is_an_accepted_false_positive(self) -> None:
        """The suffix rule costs one accepted false positive — 'monkey' ends
        with 'key' — in exchange for never missing an un-anticipated glued
        spelling. Recorded here so the trade-off is a documented decision,
        not a surprise rediscovered later."""
        redacted = redact_secrets("monkey=x", max_length=500)
        assert redacted == "monkey=[REDACTED]"

    def test_keyframes_and_author_still_survive_intact(self) -> None:
        """Layer-3 preservation regression guard (U3): neither ends with a
        marker (keyframes ends with 'frames', author ends with 'thor')."""
        assert redact_secrets("keyframes: 12", max_length=500) == "keyframes: 12"
        assert redact_secrets("author: alice", max_length=500) == "author: alice"

    def test_session_id_still_survives_intact(self) -> None:
        """U3: 'session' was already dropped from the Layer-3 marker set, and
        neither 'session' nor 'id' ends with a remaining marker."""
        redacted = redact_secrets("session_id=abc123", max_length=500)
        assert redacted == "session_id=abc123"


# ---------------------------------------------------------------------------
# X4 (round-5): a URL preceded by a non-strippable prefix still leaked its
# query string, because _redact_url_in_token required the URL at offset 0.
# ---------------------------------------------------------------------------


class TestX4UrlAtAnyOffsetInToken:
    def test_url_prefixed_lowercase_key_style_marker(self) -> None:
        redacted = redact_secrets("url:https://a.test/x?token=S", max_length=500)
        assert redacted == "url:https://a.test/x"

    def test_url_inside_function_call_parens(self) -> None:
        redacted = redact_secrets("fetch(https://a.test/x?token=S)", max_length=500)
        assert redacted == "fetch(https://a.test/x)"

    def test_url_inside_markdown_link(self) -> None:
        redacted = redact_secrets("[link](https://a.test/x?token=S)", max_length=500)
        assert redacted == "[link](https://a.test/x)"

    def test_url_after_non_strippable_message_prefix(self) -> None:
        redacted = redact_secrets("msg=failed:https://a.test/x?token=S", max_length=500)
        assert redacted == "msg=failed:https://a.test/x"

    def test_href_double_quoted_url_still_works(self) -> None:
        """Already worked before X4 — regression guard."""
        redacted = redact_secrets('href="https://a.test/x?token=S"', max_length=500)
        assert redacted == 'href="https://a.test/x"'

    def test_angle_bracketed_url_still_works(self) -> None:
        """Already worked before X4 — regression guard."""
        redacted = redact_secrets("<https://a.test/x?token=S>", max_length=500)
        assert redacted == "<https://a.test/x>"

    def test_two_urls_in_a_single_token_leak_neither(self) -> None:
        """The second URL lands inside the first URL's query and is stripped
        along with it — no separate handling needed, and nothing survives."""
        redacted = redact_secrets(
            "https://a.test/x?token=1,https://b.test/y?token=2", max_length=500
        )
        assert "token=1" not in redacted
        assert "token=2" not in redacted
        assert "b.test" not in redacted
        assert redacted == "https://a.test/x"

    def test_url_with_no_query_after_a_prefix_is_unchanged(self) -> None:
        redacted = redact_secrets("url:https://a.test/x", max_length=500)
        assert redacted == "url:https://a.test/x"

    def test_bare_http_prefix_with_no_host_does_not_raise(self) -> None:
        """U2's no-raise property must still hold after X4's offset scan.

        `urlsplit("http://")` doesn't raise (empty netloc/path are valid), so
        this degrades to itself rather than [REDACTED] — the point is only
        that it never raises out of the redaction boundary."""
        redacted = redact_secrets("see http:// for details", max_length=500)
        assert redacted == "see http:// for details"

    def test_sensitive_key_still_wins_when_url_is_found_at_an_offset(self) -> None:
        """X4 must not weaken the existing sensitive-key-wins guarantee
        (test_sensitive_key_url_value_is_redacted_wholesale_not_just_query)
        merely because the URL can now be found at a non-zero offset."""
        redacted = redact_secrets('TOKEN="https://a.test/x?y=1"', max_length=500)
        assert redacted == 'TOKEN="[REDACTED]"'
        assert "a.test" not in redacted


# ---------------------------------------------------------------------------
# Y1 (round-6): a compact token can hold many KEY=value / KEY:value pairs.
# msgspec emits this no-whitespace JSON form in production, so every observed
# leak shape and the nearby preservation boundary has an exact-output test.
# ---------------------------------------------------------------------------


_Y1_COMPACT_LEAK_CASES: tuple[tuple[str, str], ...] = (
    ("Authorization:Bearer CALLBACKTOK", "Authorization:Bearer [REDACTED]"),
    ("authorization:CALLBACKTOK", "authorization:[REDACTED]"),
    ("x-auth-token:CALLBACKTOK", "x-auth-token:[REDACTED]"),
    ("X-Api-Key:CALLBACKTOK", "X-Api-Key:[REDACTED]"),
    ("cookie:a=CALLBACKTOK", "cookie:[REDACTED]"),
    ("api_key:CALLBACKTOK", "api_key:[REDACTED]"),
    ("token:'CALLBACKTOK'", "token:'[REDACTED]'"),
    ('token:"CALLBACKTOK"', 'token:"[REDACTED]"'),
    ('{"token":"CALLBACKTOK"}', '{"token":"[REDACTED]"}'),
    ('[{"secret":"CALLBACKTOK"}]', '[{"secret":"[REDACTED]"}]'),
    ("a=1;token=CALLBACKTOK;b=2", "a=1;token=[REDACTED];b=2"),
    ("env=prod&api_key=CALLBACKTOK", "env=prod&api_key=[REDACTED]"),
    ("opts=--token=CALLBACKTOK", "opts=--token=[REDACTED]"),
    ("x=1,secret=CALLBACKTOK", "x=1,secret=[REDACTED]"),
    (
        'headers={"Authorization":"Bearer CALLBACKTOK"}',
        'headers={"Authorization":"Bearer [REDACTED]"}',
    ),
)
_Y1_SPACED_REGRESSION_CASES: tuple[tuple[str, str], ...] = (
    ("Authorization: Bearer CALLBACKTOK", "Authorization: Bearer [REDACTED]"),
    ('"token": "CALLBACKTOK"', '"token": "[REDACTED]"'),
)
_Y1_OVER_REDACTION_GUARDS: tuple[str, ...] = (
    "Error: no space left on device",
    "elapsed 12:30:05",
    "https://a/b:8080/x",
    "phase: download, progress: 42",
    "keyframes: 12 rendered, 4 pending",
    "author: gearbox, revision: 3",
    "session_id: abc-123",
)


class TestY1IntraTokenPairScanner:
    @pytest.mark.parametrize(
        ("line", "expected"),
        _Y1_COMPACT_LEAK_CASES,
        ids=(
            "authorization-bearer",
            "authorization-lowercase",
            "x-auth-token",
            "x-api-key",
            "cookie-with-nested-pair",
            "api-key-colon",
            "single-quoted-value",
            "double-quoted-value",
            "compact-json-object",
            "compact-json-list",
            "semicolon-separated-pairs",
            "ampersand-separated-pairs",
            "nested-assignment",
            "comma-separated-pairs",
            "embedded-compact-header-json",
        ),
    )
    def test_compact_leak_forms_are_redacted(self, line: str, expected: str) -> None:
        assert redact_secrets(line, max_length=500) == expected

    @pytest.mark.parametrize(
        ("line", "expected"),
        _Y1_SPACED_REGRESSION_CASES,
        ids=("authorization", "json-key"),
    )
    def test_spaced_forms_keep_their_existing_output(self, line: str, expected: str) -> None:
        assert redact_secrets(line, max_length=500) == expected

    @pytest.mark.parametrize("line", _Y1_OVER_REDACTION_GUARDS)
    def test_benign_colon_forms_are_preserved(self, line: str) -> None:
        assert redact_secrets(line, max_length=500) == line

    def test_msgspec_compact_json_redacts_only_the_token_field(self) -> None:
        line = msgspec.json.encode({"token": "x", "message": "boom"}).decode()
        assert redact_secrets(line, max_length=500) == '{"token":"[REDACTED]","message":"boom"}'

    def test_multiple_sensitive_compact_json_pairs_leave_benign_field_intact(self) -> None:
        line = '{"token":"A","api_key":"B","phase":"download"}'
        assert redact_secrets(line, max_length=500) == (
            '{"token":"[REDACTED]","api_key":"[REDACTED]","phase":"download"}'
        )

    @pytest.mark.parametrize(
        "source",
        tuple(line for line, _ in _Y1_COMPACT_LEAK_CASES + _Y1_SPACED_REGRESSION_CASES)
        + _Y1_OVER_REDACTION_GUARDS,
    )
    def test_new_pair_scanner_table_is_idempotent(self, source: str) -> None:
        once = redact_secrets(source, max_length=500)
        assert redact_secrets(once, max_length=500) == once


# ---------------------------------------------------------------------------
# U2 (round-4): a malformed URL must never raise out of the redaction
# boundary — it degrades to a wholesale [REDACTED] instead.
# ---------------------------------------------------------------------------


class TestU2MalformedUrlNeverRaises:
    @pytest.mark.parametrize(
        "malformed",
        [
            "http://[",
            "https://[::1",
            "http://[bad]:notaport/x",
        ],
    )
    def test_malformed_url_as_the_whole_value_does_not_raise(self, malformed: str) -> None:
        redacted = redact_secrets(malformed, max_length=500)
        assert redacted == "[REDACTED]"

    def test_malformed_url_embedded_in_a_sentence_does_not_raise(self) -> None:
        redacted = redact_secrets("see http://[ for details", max_length=500)
        assert redacted == "see [REDACTED] for details"

    def test_malformed_url_in_key_value_assignment_does_not_raise(self) -> None:
        redacted = redact_secrets("PROVISIONING_SCRIPT=http://[", max_length=500)
        assert redacted == "PROVISIONING_SCRIPT=[REDACTED]"

    def test_malformed_url_in_mapping_does_not_raise(self) -> None:
        result = redact_secrets_mapping({"error": "see http://[ for details"}, max_length=500)
        assert result is not None
        assert result["error"] == "see [REDACTED] for details"


class TestNoRaiseInvariant:
    """U2: redact_secrets / redact_secrets_mapping / redact_known_secrets must
    never raise for any str input — a property test over a corpus of hostile
    strings, standing in for the property-based test the finding asks for
    (this repo has no hypothesis dependency to reach for)."""

    _HOSTILE_STRINGS: tuple[str, ...] = (
        "",
        "http://",
        "https://",
        "://x",
        "http://[",
        "https://[::1",
        "http://[bad]:notaport/x",
        "http://[[[[[",
        "http://]]]]]",
        "http://[" * 50,
        "\x00\x01\x02control\x03chars",
        "\ud800lone-high-surrogate",
        "\udcffanother-lone-surrogate",
        "'\"({[<>]})'\"'",
        "'''\"\"\"((()))",
        "KEY=" + '"' * 200,
        "KEY='" + "x" * 5000,
        "=" * 500,
        "token=" * 200,
        "a" * 100_000,
        "http://[" + "a" * 10_000,
        "session: " + "x" * 10_000,
        "Authorization:" + "\n" * 100,
        "\n\n\n\n\n",
        " " * 1000,
    )

    @pytest.mark.parametrize("hostile", _HOSTILE_STRINGS)
    def test_redact_secrets_never_raises(self, hostile: str) -> None:
        redact_secrets(hostile, max_length=500, known_secrets=frozenset({"ghp_someSecret123"}))

    @pytest.mark.parametrize("hostile", _HOSTILE_STRINGS)
    def test_redact_secrets_mapping_never_raises(self, hostile: str) -> None:
        redact_secrets_mapping(
            {"message": hostile, "nested": {"error": hostile}, hostile: "value"},
            max_length=500,
        )

    @pytest.mark.parametrize("hostile", _HOSTILE_STRINGS)
    def test_redact_known_secrets_never_raises(self, hostile: str) -> None:
        redact_known_secrets(hostile, frozenset({"ghp_someSecret123"}))

    def test_combinatorial_punctuation_corpus_never_raises(self) -> None:
        """Deeply mixed punctuation/quote/bracket combinations, generated
        rather than hand-written, as an extra hostile-input sweep."""
        punctuation = "\"'(){}[]<>,;:=?&%#"
        for a in punctuation:
            for b in punctuation:
                for c in punctuation:
                    sample = f"KEY{a}{b}{c}http://{a}{b}{c}value"
                    redact_secrets(sample, max_length=200)


# ---------------------------------------------------------------------------
# U3 (round-4): Layer 3 key matching is boundary-aware and a non-allowlisted
# `KEY:` marker only costs the next value token, not the rest of the line.
# ---------------------------------------------------------------------------


class TestU3BoundaryAwareFreeTextRedaction:
    def test_session_prefixed_progress_line_is_fully_preserved(self) -> None:
        line = "session: starting phase 3 of 7"
        assert redact_secrets(line, max_length=500) == line

    def test_keyframes_line_is_fully_preserved(self) -> None:
        line = "keyframes: 12 rendered, 4 pending"
        assert redact_secrets(line, max_length=500) == line

    def test_author_and_revision_line_is_fully_preserved(self) -> None:
        line = "author: gearbox, revision: 3"
        assert redact_secrets(line, max_length=500) == line

    def test_json_as_text_with_only_benign_keys_is_fully_preserved(self) -> None:
        line = '{"session_id": "abc-123", "message": "disk full", "phase": "download"}'
        assert redact_secrets(line, max_length=500) == line

    def test_json_as_text_token_field_redacts_only_that_value(self) -> None:
        redacted = redact_secrets('{"token": "x", "other": "y"}', max_length=500)
        assert redacted == '{"token": "[REDACTED]", "other": "y"}'

    @pytest.mark.parametrize(
        "env_line",
        [
            "ACS_GITHUB_TOKEN=ghp_abc123secret",
            "CF_TUNNEL_TOKEN=abc123secret",
            "HF_TOKEN=hf_abc123secret",
            "CIVITAI_API_TOKEN=civ_abc123secret",
            "api_key=abc123secret",
            "secret_key=abc123secret",
            "password=abc123secret",
        ],
    )
    def test_known_marker_names_still_match_under_boundary_rules(self, env_line: str) -> None:
        redacted = redact_secrets(env_line, max_length=500)
        assert "abc123secret" not in redacted
        assert "[REDACTED]" in redacted

    def test_authorization_header_full_line_still_redacted(self) -> None:
        """Regression: "authorization" has no separator to split on, so it
        cannot boundary-match "auth" — it must be reached via the exact-match
        allowlist instead, not the segment test."""
        redacted = redact_secrets("Authorization: Bearer ghp_SECRETVALUE123", max_length=500)
        assert redacted == "Authorization: Bearer [REDACTED]"

    def test_glued_apikey_spelling_matches_via_endswith_fallback(self) -> None:
        redacted = redact_secrets("MY_APIKEY=abc123secret", max_length=500)
        assert "abc123secret" not in redacted
        assert "[REDACTED]" in redacted

    def test_glued_authtoken_spelling_matches_via_endswith_fallback(self) -> None:
        redacted = redact_secrets("AUTHTOKEN=abc123secret", max_length=500)
        assert "abc123secret" not in redacted
        assert "[REDACTED]" in redacted


class TestScopedKeyMarkerLineBoundaries:
    """Branch coverage for _redact_next_value_token's line-boundary handling —
    the scoped, non-allowlisted counterpart to
    TestAuthorizationHeaderLineBoundaries (which covers the same shapes for
    the allowlisted _redact_remainder_of_line)."""

    def test_marker_at_end_of_line_has_nothing_to_redact(self) -> None:
        assert (
            redact_secrets("token:\nunrelated next line", max_length=500)
            == "token:\nunrelated next line"
        )

    def test_scheme_word_at_end_of_line_has_nothing_after_it(self) -> None:
        assert (
            redact_secrets("token: Bearer\nunrelated next line", max_length=500)
            == "token: Bearer\nunrelated next line"
        )

    def test_scheme_word_followed_by_a_value_redacts_only_that_value(self) -> None:
        redacted = redact_secrets("token: Bearer secretvalue", max_length=500)
        assert redacted == "token: Bearer [REDACTED]"

    def test_marker_with_only_trailing_whitespace_has_nothing_to_redact(self) -> None:
        assert redact_secrets("token: ", max_length=500) == "token: "


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
    """Not just a literal token= form — any KEY=value where KEY
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
        # U4: the walker must stop, not just mark — 10,000 input keys must not
        # produce 10,000 output keys. Bounded to roughly the node budget,
        # plus exactly one truncation marker entry.
        assert len(result) <= 101
        assert sum(v == "[TRUNCATED: max nodes]" for v in result.values()) == 1

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
        # U4: bounded to roughly the node budget, not the full 10,000 items.
        assert len(result["items"]) <= 100
        assert result["items"].count("[TRUNCATED: max nodes]") == 1

    def test_zero_node_budget_truncates_the_top_level_value_itself(self) -> None:
        result = redact_secrets_mapping({"a": 1}, max_length=500, max_nodes=0)
        assert result == "[TRUNCATED: max nodes]"

    def test_worst_case_serialized_output_size_is_bounded(self) -> None:
        """U4: the node budget must bound total output size, not just how
        many entries get replaced with a marker — serialize the worst case
        (a 10,000-key wide payload under the default budget, non-sensitive
        key names so the ordinary recursive path is exercised rather than
        Layer 2's wholesale sensitive-key shortcut) and assert it stays well
        under what an unbounded walk would have produced."""
        wide = {f"k{i}": "v" for i in range(10_000)}
        result = redact_secrets_mapping(wide, max_length=500)
        assert result is not None
        assert len(result) <= 2001
        serialized = json.dumps(result)
        # An unbounded walk would serialize to ~90,000+ characters (10,000
        # entries); the default 2000-node budget must keep this to a small
        # fraction of that.
        assert len(serialized) < 40_000

    def test_payload_under_budget_is_unchanged_apart_from_redaction(self) -> None:
        """No spurious truncation marker when the payload never exhausts the
        budget."""
        payload = {"a": 1, "b": "safe value", "c": {"d": "also safe"}}
        result = redact_secrets_mapping(payload, max_length=500, max_nodes=1000)
        assert result == payload
        assert "[TRUNCATED: max nodes]" not in str(result)
