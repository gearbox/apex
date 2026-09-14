"""Redaction helpers for untrusted text that may be written to logs."""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

_URL_RE = re.compile(r"https?://[^\s<>\"']+", flags=re.IGNORECASE)
_SECRET_PARAMETER_RE = re.compile(r"(?i)(\b(?:token|session)\s*=\s*)([^\s&#'\")\]}>,;]+)")


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
