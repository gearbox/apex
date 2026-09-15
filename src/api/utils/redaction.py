"""Redaction helpers for untrusted text that may be written to logs, persisted, or
published — see each call site for which of those apply.

No node-supplied string reaches a database row, an SSE payload, or a billing record
without passing through one of these first. That boundary rule lives here, once,
rather than being re-derived per call site (see the 2026-09 incident where a fix
applied at one boundary — the provisioner failure webhook — did not cover a second,
newer boundary — telemetry operation events — that carried the same token).

Three layers, applied in order (round-3 remediation, T1):

1. **Exact known-value replacement** (``known_secrets``) — a plain ``str.replace``
   pass against every plaintext secret apex actually holds at call time
   (``settings.github_content_token``/``hf_token``/``civitai_api_token``, and the
   per-session ``tunnel_token`` where it is in scope). This is the only layer that
   is a *guarantee*: it has no false negatives for those exact values regardless of
   how they are quoted, prefixed, or embedded, because it never tries to recognize
   a shape at all.
2. **Key-based structural redaction** — in the mapping walker, a dict key is tested
   *before* its value is visited; a sensitive key's value is replaced wholesale
   (never descended into), so a secret hiding under an unanticipated key name is
   still caught as long as the key name itself is recognizable.
3. **A tokenizer, best-effort, for free text** — walks the string once on
   whitespace boundaries and inspects each non-whitespace token for URL query
   strings, ``KEY=value`` assignment shapes (quoted or not, split across tokens as
   ``KEY = value``), and ``Authorization:``-style / JSON ``"key":``-style markers.
   Whitespace is a hard separator and never treated as part of a value, so this
   layer cannot reach across a line and consume an unrelated token the way a regex
   with a bare ``\\s*`` could.

Layers 2 and 3 are shape-matching and therefore best-effort — see T8 in the round-3
remediation notes: the per-session callback token is stored only as a hash, so
Layer 1 can never cover it, and it is the one secret this module cannot guarantee
to catch. Do not represent redaction as a complete defence for that value.

The mapping walker also bounds recursion depth and total node count (T2/T10): a
node-supplied ``progress``/``plan``/``summary`` body is open-ended JSON, and an
unbounded recursive walk over a maliciously (or just buggily) deep or wide payload
is a crash (``RecursionError``) or unbounded-write vector on a hot path every node
hits on every tick. Exceeding either bound truncates the subtree with a marker
string; it never raises.
"""

from __future__ import annotations

from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

_REDACTED = "[REDACTED]"

# Layer 2/3 sensitivity test: a lowercase substring check, deliberately broad —
# over-redacting a benign field costs nothing, missing a real secret does.
_SENSITIVE_KEY_MARKERS = (
    "token",
    "secret",
    "key",
    "password",
    "passwd",
    "credential",
    "auth",
    "cookie",
    "session",
)

# Recognized auth schemes kept as-is right after a redacted `Authorization:`-style
# marker (e.g. "Authorization: Bearer [REDACTED]", not "Authorization: [REDACTED]").
_AUTH_SCHEME_WORDS = frozenset({"bearer", "basic", "token", "digest"})

# Punctuation stripped from a token's edges before it is inspected for shape —
# quotes, braces, and the trailing ':' that marks a header/JSON-key token. Square
# brackets are deliberately excluded: they wrap our own [REDACTED] marker, and
# stripping them would let a second redaction pass re-wrap an already-redacted
# value (breaking idempotency) instead of treating it as stable output.
_STRIP_CHARS = "\"'(){}<>,;:"
_QUOTE_CHARS = "\"'"

# Layer 1: ignore settings values shorter than this so an empty/trivial config
# value can never blank out unrelated text.
_MIN_KNOWN_SECRET_LENGTH = 8

# T2/T10: bounds for the mapping walker. 16 is generous for real telemetry nesting;
# 2000 nodes bounds total walk work (and therefore JSONB write size) for a
# wide-but-shallow payload that depth alone wouldn't catch.
_DEFAULT_MAX_DEPTH = 16
_DEFAULT_MAX_NODES = 2000
_MAX_DEPTH_MARKER = "[TRUNCATED: max depth]"
_MAX_NODES_MARKER = "[TRUNCATED: max nodes]"


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS)


def _strip_wrapping(token: str) -> tuple[str, str, str]:
    """Split a token into (leading punctuation, core, trailing punctuation)."""
    start, end = 0, len(token)
    while start < end and token[start] in _STRIP_CHARS:
        start += 1
    while end > start and token[end - 1] in _STRIP_CHARS:
        end -= 1
    return token[:start], token[start:end], token[end:]


def _redact_value_token(token: str) -> str:
    """Replace a token's core with [REDACTED], preserving surrounding punctuation."""
    prefix, core, suffix = _strip_wrapping(token)
    if not core:
        # Nothing but punctuation (or truly empty) — leave alone rather than
        # inventing a redaction for a value that isn't there.
        return token
    return f"{prefix}{_REDACTED}{suffix}"


def _redact_url(url: str) -> str:
    """Remove a URL's query and userinfo while retaining safe routing context."""
    parsed = urlsplit(url)
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _tokenize(text: str) -> list[tuple[bool, str]]:
    """Split `text` into (is_whitespace, chunk) segments that concatenate back to it."""
    segments: list[tuple[bool, str]] = []
    i, n = 0, len(text)
    while i < n:
        start = i
        is_ws = text[i].isspace()
        while i < n and text[i].isspace() == is_ws:
            i += 1
        segments.append((is_ws, text[start:i]))
    return segments


def _consume_quoted_value(
    segments: list[tuple[bool, str]], out: list[str], start: int, left: str, quote: str
) -> int:
    """Handle `KEY="a b"` where the value's closing quote is in a later token.

    Redacts everything from the opening quote (in the `start` token) through the
    token that closes it (or through the end of the string if it is never closed),
    and returns the index to resume scanning from.
    """
    n = len(segments)
    j = start + 1
    closed_at: int | None = None
    while j < n:
        if not segments[j][0] and segments[j][1].endswith(quote):
            closed_at = j
            break
        j += 1
    end = closed_at if closed_at is not None else n - 1
    out[start] = f"{left}={quote}{_REDACTED}{quote}"
    for k in range(start + 1, end + 1):
        out[k] = ""
    return end + 1


def _redact_remainder_of_line(segments: list[tuple[bool, str]], out: list[str], start: int) -> int:
    """Redact every token on the same line after a sensitive `KEY:` marker token.

    Keeps one recognized auth-scheme word (Bearer/token/basic/digest) immediately
    after the marker, if present, so e.g. "Authorization: Bearer [REDACTED]"
    reads naturally rather than eating the scheme word too.
    """
    n = len(segments)
    j = start + 1
    if j < n and segments[j][0]:
        if "\n" in segments[j][1]:
            return j
        j += 1
    if j < n and not segments[j][0]:
        _, core, _ = _strip_wrapping(segments[j][1])
        if core.lower() in _AUTH_SCHEME_WORDS:
            j += 1
            if j < n and segments[j][0]:
                if "\n" in segments[j][1]:
                    return j
                j += 1
    k = j
    while k < n:
        is_ws, chunk = segments[k]
        if is_ws:
            if "\n" in chunk:
                break
            k += 1
            continue
        out[k] = _redact_value_token(chunk)
        k += 1
    return k


def _redact_free_text(text: str) -> str:
    """Layer 3: best-effort shape-based redaction over whitespace-delimited tokens."""
    segments = _tokenize(text)
    out = [chunk for _, chunk in segments]
    n = len(segments)
    prev_bare_sensitive_key = False
    i = 0
    while i < n:
        is_ws, token = segments[i]
        if is_ws:
            i += 1
            continue

        lowered = token.lower()
        if lowered.startswith(("http://", "https://")):
            out[i] = _redact_url(token)
            prev_bare_sensitive_key = False
            i += 1
            continue

        if token == "=" and prev_bare_sensitive_key:  # noqa: S105 -- text token, not a credential
            # "KEY = value" split across three tokens by the spaces around '='.
            prev_bare_sensitive_key = False
            j = i + 1
            if j < n and segments[j][0] and "\n" not in segments[j][1]:
                j += 1
            if j < n and not segments[j][0]:
                out[j] = _redact_value_token(segments[j][1])
            i += 1
            continue

        if "=" in token:
            left, _, right = token.partition("=")
            _, left_core, _ = _strip_wrapping(left)
            if _is_sensitive_key(left_core):
                quote = right[0] if right and right[0] in _QUOTE_CHARS else None
                unclosed = quote is not None and not (len(right) >= 2 and right.endswith(quote))
                if quote is not None and unclosed:
                    i = _consume_quoted_value(segments, out, i, left, quote)
                    prev_bare_sensitive_key = False
                    continue
                out[i] = f"{left}={_redact_value_token(right)}" if right else token
            prev_bare_sensitive_key = False
            i += 1
            continue

        _, core, suffix = _strip_wrapping(token)
        if ":" in suffix and core and _is_sensitive_key(core):
            i = _redact_remainder_of_line(segments, out, i)
            prev_bare_sensitive_key = False
            continue

        prev_bare_sensitive_key = bool(core) and _is_sensitive_key(core)
        i += 1

    return "".join(out)


def _apply_known_secrets(text: str, known_secrets: frozenset[str]) -> str:
    """Layer 1: exact replacement of every plaintext secret apex currently holds."""
    for secret in known_secrets:
        if len(secret) >= _MIN_KNOWN_SECRET_LENGTH and secret in text:
            text = text.replace(secret, _REDACTED)
    return text


def redact_known_secrets(text: str, known_secrets: frozenset[str]) -> str:
    """Layer-1-only redaction for structurally simple fields (T11).

    For values like an event id or a bundle version — not free text, so the full
    tokenizer pass is unneeded overhead — but an exact known-secret match must
    still never survive into a persisted row.
    """
    return _apply_known_secrets(text, known_secrets)


def redact_secrets(
    text: str, *, max_length: int, known_secrets: frozenset[str] = frozenset()
) -> str:
    """Redact `text` through all three layers, then bound its length.

    `known_secrets` should be every plaintext secret apex holds at the call site
    (see the module docstring's Layer 1). Values under 8 characters are ignored.
    """
    redacted = _apply_known_secrets(text, known_secrets)
    redacted = _redact_free_text(redacted)
    return redacted[:max_length]


class _NodeBudget:
    """Mutable shared counter so sibling subtrees see each other's node spend."""

    __slots__ = ("remaining",)

    def __init__(self, remaining: int) -> None:
        self.remaining = remaining


def _redact_json_value(
    value: Any,  # noqa: ANN401
    *,
    max_length: int,
    known_secrets: frozenset[str],
    depth: int,
    budget: _NodeBudget,
    max_depth: int,
) -> Any:  # noqa: ANN401
    """Recursively redact string leaves in a JSON-like structure; other types pass through.

    Bounded on two axes (T2/T10): `depth` caps recursion so a maliciously or
    buggily deep payload truncates instead of raising `RecursionError`, and
    `budget` caps total nodes visited so a wide-but-shallow payload can't produce
    an unbounded JSONB write either. Both bounds truncate with a marker; neither
    ever raises.
    """
    if budget.remaining <= 0:
        return _MAX_NODES_MARKER
    budget.remaining -= 1

    if depth > max_depth:
        return _MAX_DEPTH_MARKER

    if isinstance(value, str):
        return redact_secrets(value, max_length=max_length, known_secrets=known_secrets)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, v in value.items():
            if budget.remaining <= 0:
                result[key] = _MAX_NODES_MARKER
                continue
            if _is_sensitive_key(str(key)):
                # Layer 2: sensitive key -> whole value redacted wholesale,
                # never descended into (even if it's itself a dict/list).
                budget.remaining -= 1
                result[key] = _REDACTED
                continue
            result[key] = _redact_json_value(
                v,
                max_length=max_length,
                known_secrets=known_secrets,
                depth=depth + 1,
                budget=budget,
                max_depth=max_depth,
            )
        return result
    if isinstance(value, list):
        items: list[Any] = []
        for item in value:
            if budget.remaining <= 0:
                items.append(_MAX_NODES_MARKER)
                continue
            items.append(
                _redact_json_value(
                    item,
                    max_length=max_length,
                    known_secrets=known_secrets,
                    depth=depth + 1,
                    budget=budget,
                    max_depth=max_depth,
                )
            )
        return items
    return value


def redact_secrets_mapping(
    value: dict[str, Any] | None,
    *,
    max_length: int,
    known_secrets: frozenset[str] = frozenset(),
    max_depth: int = _DEFAULT_MAX_DEPTH,
    max_nodes: int = _DEFAULT_MAX_NODES,
) -> dict[str, Any] | None:
    """Redact every string leaf, at any depth, in a JSON-like mapping.

    Used for node-supplied ``progress``/``plan``/``summary`` payloads, which are
    open-ended ``dict[str, Any]`` bodies where a secret could hide at any depth —
    a flat top-level-only redaction would miss one nested under a key like
    ``{"detail": {"last_error": "...token=...\"}}``. `known_secrets` should be
    every plaintext secret apex holds at the call site (see the module docstring).
    """
    if value is None:
        return None
    budget = _NodeBudget(max_nodes)
    return cast(
        "dict[str, Any]",
        _redact_json_value(
            value,
            max_length=max_length,
            known_secrets=known_secrets,
            depth=0,
            budget=budget,
            max_depth=max_depth,
        ),
    )
