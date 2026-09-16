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
   with a bare ``\\s*`` could. A URL is redacted wherever it appears — as a bare
   token, or as the value on the right of ``KEY=`` — not only when the whole
   token happens to start with ``http(s)://`` (U1, round-4): ``PROVISIONING_SCRIPT``
   and ``PROVISIONER_WEBHOOK_URL`` are exactly this shape and contain none of the
   sensitive-key markers, so without this a query string carrying the callback
   token survived untouched.

Layers 2 and 3 are shape-matching and therefore best-effort — see T8 in the round-3
remediation notes: the per-session callback token is stored only as a hash, so
Layer 1 can never cover it, and it is the one secret this module cannot guarantee
to catch. Do not represent redaction as a complete defence for that value.

**Layer 2 vs Layer 3 key matching are deliberately different tests (U3, round-4).**
Layer 2 (a mapping key, tested in the dict walker) uses a broad, lowercase
*substring* match: over-redacting costs exactly one field's value, so the risk
worth avoiding is a false negative — a secret hiding under a key name nobody
anticipated. Layer 3 (free text — a ``KEY=value`` shape, or a ``KEY:``/JSON
``"key":`` marker) uses a boundary-aware test instead: the key is split on
``_``/``-``/``.`` and a segment must *equal* a marker (plus an ``endswith``
fallback for glued spellings like ``apikey``/``authtoken``), because a Layer-3
false positive does not cost one field — a marker matching as a bare substring
(``session`` inside ``session_id``, ``key`` inside ``keyframes``, ``auth`` inside
``author``) used to redact to the end of the line, destroying an unbounded run of
ordinary telemetry text around it. For the same reason, a Layer-3 ``KEY:``-style
match only redacts the one value token that follows (plus a recognized
auth-scheme word, e.g. ``Bearer``) rather than the rest of the line — full
rest-of-line redaction is now reserved for an explicit, exact-match allowlist of
header names (``Authorization``, ``Cookie``, etc.) whose entire remainder really
is credential material. Layer 3 also drops ``session`` from its marker set
entirely (Layer 2 keeps it): a session id is not a credential, ``session=`` in a
URL query is already stripped by the URL-handling above, and it was the single
noisiest word in ordinary node telemetry.

The mapping walker also bounds recursion depth and total node count (T2/T10): a
node-supplied ``progress``/``plan``/``summary`` body is open-ended JSON, and an
unbounded recursive walk over a maliciously (or just buggily) deep or wide payload
is a crash (``RecursionError``) or unbounded-write vector on a hot path every node
hits on every tick. Exceeding the depth bound truncates the subtree with a marker
string. Exceeding the node budget (U4, round-4) stops iterating a dict/list
entirely and appends a single truncation marker for the remainder, rather than
continuing to visit and copy every remaining entry with a marker value — the
budget bounds total walk work (and therefore output size), not just how many
entries get replaced.

**Invariant: this module's public entry points never raise for any ``str``
input.** ``redact_secrets``, ``redact_secrets_mapping``, and
``redact_known_secrets`` are the trust boundary for untrusted node text on hot
paths (a webhook, a per-tick telemetry POST) — an input that crashes the
redactor crashes the boundary, not just the field being redacted (U2, round-4:
``urlsplit`` raises ``ValueError`` on inputs like ``"http://["``, which ordinary
log text can produce without anyone trying, e.g. a URL that wraps or truncates
mid-token, or bracketed IPv6). A malformed URL that cannot be safely parsed
degrades to a wholesale ``[REDACTED]`` rather than propagating an exception or
returning the unparsed (possibly secret-bearing) original — a URL this module
cannot parse is a URL whose query it cannot safely strip, so the fail-safe
output is the redaction marker, not the input.
"""

from __future__ import annotations

from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

_REDACTED = "[REDACTED]"

# Layer 2 (mapping-key) sensitivity test: a lowercase substring check,
# deliberately broad — over-redacting a benign field costs nothing, missing a
# real secret does. See _SENSITIVE_KEY_MARKERS_FREE_TEXT for why Layer 3 does
# not reuse this set unchanged (U3, round-4).
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

# Layer 3 (free-text tokenizer) marker set — same as Layer 2 minus "session"
# (U3, round-4): a session id is not a credential, `session=` in a URL query is
# already stripped by _redact_url, and it is the single noisiest word in
# ordinary node telemetry. Layer 2 keeps "session" since over-redacting a whole
# mapping value there costs exactly one field, a cost Layer 3 does not share.
_SENSITIVE_KEY_MARKERS_FREE_TEXT = tuple(
    marker for marker in _SENSITIVE_KEY_MARKERS if marker != "session"
)

# Layer 3 glued-spelling fallback (U3): compound names with no separator to
# split on, e.g. `APIKEY=`/`authtoken=`. Checked via `str.endswith`, not
# segment equality.
_GLUED_KEY_SUFFIXES = ("apikey", "authtoken")

# Layer 3 exact-match allowlist (U3): header names whose entire remainder of
# the line really is credential material, so rest-of-line redaction is safe
# there specifically — unlike the general `KEY:` marker match, which now
# redacts only the next value token (see _redact_next_value_token). Spelled
# out in full because boundary-aware matching means `authorization` no longer
# matches the substring marker `auth`.
_REST_OF_LINE_HEADER_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
    }
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
# U4: sentinel key for the single truncation entry appended when a dict's node
# budget runs out mid-walk — arbitrary but chosen not to collide with an
# ordinary telemetry key name.
_MAX_NODES_TRUNCATION_KEY = "__truncated__"


def _is_sensitive_key(key: str) -> bool:
    """Layer 2 (mapping-key) sensitivity test — broad substring match.

    Used only by the JSON/dict walker (_redact_json_value). See the module
    docstring's "Layer 2 vs Layer 3" note for why this is intentionally
    broader than _is_sensitive_key_segment.
    """
    lowered = key.lower()
    return any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS)


def _is_sensitive_key_segment(key: str) -> bool:
    """Layer 3 (free-text tokenizer) sensitivity test — boundary-aware (U3).

    Splits `key` on `_`/`-`/`.` and requires an exact segment match against a
    Layer-3 marker, with an `endswith` fallback for glued spellings that have
    no separator to split on (`apikey`, `authtoken`). `session_id`,
    `keyframes`, `author`, and `monkey` must not match here — none of them
    contain a marker as a whole `_`/`-`/`.`-delimited segment — see the module
    docstring for why a Layer-3 false positive is expensive in a way a
    Layer-2 one is not.
    """
    lowered = key.lower()
    if lowered.endswith(_GLUED_KEY_SUFFIXES):
        return True
    normalized = lowered.replace("-", "_").replace(".", "_")
    segments = normalized.split("_")
    return any(segment in _SENSITIVE_KEY_MARKERS_FREE_TEXT for segment in segments)


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
    # Nothing but punctuation (or truly empty) — leave alone rather than
    # inventing a redaction for a value that isn't there.
    return f"{prefix}{_REDACTED}{suffix}" if core else token


def _redact_url(url: str) -> str:
    """Remove a URL's query and userinfo while retaining safe routing context.

    On a parse failure the URL is treated as opaque and redacted wholesale
    (U2, round-4) — see the module docstring's no-raise invariant. `urlsplit`
    raises `ValueError` on shapes like `"http://["` (unbalanced IPv6 brackets),
    which ordinary log text produces without anyone trying; a URL this
    function cannot parse is a URL whose query it cannot safely strip, so the
    fail-safe output is the redaction marker, not the unparsed original.
    """
    try:
        parsed = urlsplit(url)
        netloc = parsed.netloc.rsplit("@", 1)[-1]
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        return _REDACTED


def _redact_url_in_token(token: str) -> str | None:
    """If `token`'s core (after stripping wrapping punctuation/quotes) is a
    URL, redact it and reassemble with the stripped punctuation preserved.
    Returns `None` when the core is not a URL, so callers can leave
    non-URL tokens untouched.

    Shared by both the bare-URL branch and the `KEY=<url>` branch of
    `_redact_free_text` (U1, round-4) so a URL is redacted wherever it
    appears in a token, not only when the *whole* token happens to start with
    `http(s)://` — `PROVISIONING_SCRIPT=<url>` and
    `PROVISIONER_WEBHOOK_URL=<url>` are exactly the shape that fell through
    before this existed. Stripping wrapping punctuation first (rather than
    only quotes) also preserves a trailing separator like the `:` in
    `...token=X: HTTP 404` instead of losing it along with the query (U8).
    """
    prefix, core, suffix = _strip_wrapping(token)
    if not core.lower().startswith(("http://", "https://")):
        return None
    return f"{prefix}{_redact_url(core)}{suffix}"


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


def _redact_next_value_token(segments: list[tuple[bool, str]], out: list[str], start: int) -> int:
    """Redact only the next value token after a sensitive `KEY:` marker token.

    U3, round-4: the general (non-allowlisted) `KEY:` marker match must not
    cost more than the one value that follows it — unlike
    `_redact_remainder_of_line`, which the caller now reserves for an
    explicit exact-match header allowlist. Mirrors that function's
    line-boundary and auth-scheme-word handling so `token: Bearer secret`
    still reads naturally, but stops after a single value token instead of
    consuming the rest of the line.
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
    if j < n and not segments[j][0]:
        out[j] = _redact_value_token(segments[j][1])
        return j + 1
    return j


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

        redacted_url = _redact_url_in_token(token)
        if redacted_url is not None:
            out[i] = redacted_url
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
            if _is_sensitive_key_segment(left_core):
                quote = right[0] if right and right[0] in _QUOTE_CHARS else None
                unclosed = quote is not None and not (len(right) >= 2 and right.endswith(quote))
                if quote is not None and unclosed:
                    i = _consume_quoted_value(segments, out, i, left, quote)
                    prev_bare_sensitive_key = False
                    continue
                out[i] = f"{left}={_redact_value_token(right)}" if right else token
            elif right:
                # U1: the key itself isn't recognized as sensitive, but the
                # value may still be a URL carrying a secret in its query
                # string (e.g. PROVISIONING_SCRIPT=<url>) — strip that.
                redacted_right = _redact_url_in_token(right)
                if redacted_right is not None:
                    out[i] = f"{left}={redacted_right}"
            prev_bare_sensitive_key = False
            i += 1
            continue

        _, core, suffix = _strip_wrapping(token)
        if ":" in suffix and core:
            # U3: the exact-match header allowlist is checked independently
            # of the boundary-aware segment test below — "authorization" no
            # longer matches the substring marker "auth" under boundary
            # rules (it has no `_`/`-`/`.` to split on), which is exactly why
            # it must be named in the allowlist rather than relying on that
            # test. Anything in the allowlist gets full rest-of-line
            # redaction; any other sensitive `KEY:` marker only costs the one
            # value token that follows it.
            if core.lower() in _REST_OF_LINE_HEADER_NAMES:
                i = _redact_remainder_of_line(segments, out, i)
                prev_bare_sensitive_key = False
                continue
            if _is_sensitive_key_segment(core):
                i = _redact_next_value_token(segments, out, i)
                prev_bare_sensitive_key = False
                continue

        prev_bare_sensitive_key = bool(core) and _is_sensitive_key_segment(core)
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
    `budget` caps total nodes visited. Depth exhaustion truncates the current
    subtree with a marker. Node-budget exhaustion (U4, round-4) *stops*
    iterating the current dict/list entirely and appends a single truncation
    marker for the remainder, rather than continuing to visit and copy every
    remaining entry with a marker value — the budget bounds total walk work
    (and therefore output size), not just how many entries get replaced with
    a marker. Neither bound ever raises.
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
                result[_MAX_NODES_TRUNCATION_KEY] = _MAX_NODES_MARKER
                return result
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
                return items
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
