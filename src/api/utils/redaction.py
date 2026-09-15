"""Redaction helpers for untrusted text that may be written to logs, persisted, or
published — see each call site for which of those apply.

No node-supplied string reaches a database row, an SSE payload, or a billing record
without passing through one of these first. That boundary rule lives here, once,
rather than being re-derived per call site (see the 2026-09 incident where a fix
applied at one boundary — the provisioner failure webhook — did not cover a second,
newer boundary — telemetry operation events — that carried the same token).
"""

from __future__ import annotations

import re
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

_URL_RE = re.compile(r"https?://[^\s<>\"']+", flags=re.IGNORECASE)

# Matches `token=`/`session=` (kept for backward compat with existing log
# assertions) as well as any `KEY=value` where KEY case-insensitively contains
# "token", "secret", "key", or "password" — e.g. ACS_GITHUB_TOKEN=, ACS_HF_TOKEN=,
# ACS_CIVITAI_API_TOKEN=, and the node's own tunnel-token env vars. Deliberately
# broad: over-redacting a benign param costs nothing, missing a real secret does.
_SECRET_PARAMETER_RE = re.compile(
    r"(?i)\b((?:[a-z0-9_]*(?:token|secret|key|password)[a-z0-9_]*|session)\s*=\s*)"
    r"([^\s&#'\")\]}>,;]+)"
)


def _redact_url(match: re.Match[str]) -> str:
    """Remove a URL's query and userinfo while retaining safe routing context."""
    value = match.group(0)
    parsed = urlsplit(value)
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def redact_secrets(text: str, *, max_length: int) -> str:
    """Strip URL queries, URL userinfo, and token-like params from untrusted text."""
    redacted = _URL_RE.sub(_redact_url, text)
    redacted = _SECRET_PARAMETER_RE.sub(r"\1[REDACTED]", redacted)
    return redacted[:max_length]


def _redact_json_value(value: Any, *, max_length: int) -> Any:  # noqa: ANN401
    """Recursively redact string leaves in a JSON-like structure; other types pass through."""
    if isinstance(value, str):
        return redact_secrets(value, max_length=max_length)
    if isinstance(value, dict):
        return {key: _redact_json_value(v, max_length=max_length) for key, v in value.items()}
    if isinstance(value, list):
        return [_redact_json_value(v, max_length=max_length) for v in value]
    return value


def redact_secrets_mapping(
    value: dict[str, Any] | None, *, max_length: int
) -> dict[str, Any] | None:
    """Redact every string leaf, at any depth, in a JSON-like mapping.

    Used for node-supplied ``progress``/``plan``/``summary`` payloads, which are
    open-ended ``dict[str, Any]`` bodies where a secret could hide at any depth —
    a flat top-level-only redaction would miss one nested under a key like
    ``{"detail": {"last_error": "...token=...\"}}``.
    """
    if value is None:
        return None
    return cast("dict[str, Any]", _redact_json_value(value, max_length=max_length))
