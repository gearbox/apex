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
   strings and one or more ``KEY=value`` / ``KEY:value`` pairs (quoted or not,
   split across tokens as ``KEY = value``). Within a token, pair values stop at
   a structural delimiter (comma, semicolon, ampersand, closing brace/bracket/
   parenthesis) so redacting ``{"token":"x","message":"keep"}`` cannot consume
   the unrelated ``message`` field. Whitespace is a hard separator and never
   treated as part of a value, so this layer cannot reach across a line and
   consume an unrelated token the way a regex with a bare ``\\s*`` could. A URL is
   redacted wherever it appears **at any
   offset** within a token (X4, round-5 remediation — a plain ``str.find``, no
   regex) — as a bare token, preceded by a non-strippable prefix
   (``fetch(``, ``url:``, ``msg=failed:``), or as the value on the right of
   ``KEY=`` — not only when the whole token happens to start with
   ``http(s)://`` (U1, round-4): ``PROVISIONING_SCRIPT`` and
   ``PROVISIONER_WEBHOOK_URL`` are exactly this shape and contain none of the
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
``_``/``-``/``.`` and a segment must *end with* a marker (X3, round-5
remediation — a plain ``segment.endswith(marker)`` test, which subsumes exact
equality; it replaced a two-item curated list of specific glued spellings
that was always one spelling behind — ``mytoken``, ``githubtoken``, and
``clientsecret`` all fell through it and leaked before this change), because
a Layer-3 false positive does not cost one field — a marker matching as a bare substring
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

# Layer 3 can contain several assignment-like pairs without whitespace, for
# example msgspec's compact JSON or shell/query-string fragments. A value ends
# at a structural delimiter rather than at every following separator: that is
# what lets the scanner keep examining ``a=1;token=x`` while preserving the
# unrelated fields around a redacted JSON value.
_PAIR_SEPARATORS = frozenset({"=", ":"})
_VALUE_TERMINATORS = frozenset({",", ";", "&", "}", "]", ")"})

# Punctuation stripped from a token's edges before it is inspected for shape —
# quotes, braces, and the trailing ':' that marks a header/JSON-key token. Square
# brackets are deliberately excluded: they wrap our own [REDACTED] marker, and
# stripping them would let a second redaction pass re-wrap an already-redacted
# value (breaking idempotency) instead of treating it as stable output.
_STRIP_CHARS = "\"'(){}<>,;:"
_QUOTE_CHARS = "\"'"

# Square brackets are intentionally not in _STRIP_CHARS: a second pass must
# leave [REDACTED] stable. They *are* safe to strip while recognizing a pair
# key, where they can only be JSON/list wrapping syntax (e.g. [{"token":...]).
_PAIR_KEY_STRIP_CHARS = f"{_STRIP_CHARS}[]"

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
    """Layer 3 (free-text tokenizer) sensitivity test — boundary-aware (U3, X3).

    Splits `key` on `_`/`-`/`.` and matches if any segment either *equals* a
    Layer-3 marker, or *ends with* one — the `endswith` arm is what catches
    glued compound spellings with no separator to split on at all
    (`mytoken`, `githubtoken`, `clientsecret`, `apikey`, `authtoken`, ...).
    X3, round-5 remediation: a curated list of specific glued spellings
    (formerly `_GLUED_KEY_SUFFIXES = ("apikey", "authtoken")`) is always one
    spelling behind — `mytoken=`/`githubtoken=`/`clientsecret=` all fell
    through it and leaked. A general suffix test costs one accepted false
    positive (`monkey` ends with `key`) in exchange for never missing an
    un-anticipated glued spelling — the same false-negative-averse trade U3
    already made for this layer (see the module docstring); `keyframes` and
    `author` still don't match (neither *ends with* a marker: `keyframes`
    ends with `frames`, `author` ends with `thor`), so they remain
    unaffected. `session_id` still doesn't match either — "session" was
    already dropped from `_SENSITIVE_KEY_MARKERS_FREE_TEXT` by U3.
    """
    lowered = key.lower()
    normalized = lowered.replace("-", "_").replace(".", "_")
    segments = normalized.split("_")
    # `str.endswith` against an equal string is True, so this single test
    # subsumes the old exact-match check — no separate equality branch needed.
    return any(segment.endswith(_SENSITIVE_KEY_MARKERS_FREE_TEXT) for segment in segments)


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


def _find_url_offset(lowered_core: str) -> int:
    """Index of the first `http://` or `https://` in `lowered_core`, or -1."""
    positions = [i for i in (lowered_core.find("http://"), lowered_core.find("https://")) if i >= 0]
    return min(positions, default=-1)


def _redact_url_in_token(token: str) -> str | None:
    """If a URL appears anywhere in `token`'s core (after stripping wrapping
    punctuation/quotes), redact from that point to the end of the core and
    reassemble with the stripped punctuation and any text before the URL
    preserved verbatim. Returns `None` when no URL is found, so callers can
    leave the token untouched.

    Shared by both the bare-URL branch and the `KEY=<url>` branch of
    `_redact_free_text` (U1, round-4) so a URL is redacted wherever it
    appears in a token, not only when the *whole* token happens to start with
    `http(s)://` — `PROVISIONING_SCRIPT=<url>` and
    `PROVISIONER_WEBHOOK_URL=<url>` are exactly the shape that fell through
    before U1. X4, round-5 remediation: a URL preceded by a prefix that isn't
    pure wrapping punctuation (`fetch(`, `url:`, `msg=failed:`) still fell
    through U1's `core.startswith(...)` check, because the URL wasn't at
    offset 0 even after stripping. Scanning for the URL at any offset (via
    `str.find`, no regex) subsumes the old strip-then-check logic — this is
    the only place either branch needs to call.

    If everything before the URL reduces (after stripping any quote) to a
    sensitive `KEY=`, the whole value is redacted wholesale instead of just
    the URL's query — matching the `KEY=value` sensitive-key path elsewhere
    in `_redact_free_text` (see `test_sensitive_key_url_value_is_redacted_
    wholesale_not_just_query`). Without this, scanning at any offset would
    let a call on the *whole* token (e.g. `TOKEN=https://...`) hijack that
    case ahead of the dedicated sensitive-key branch and under-redact it to
    only the query string.

    Stripping wrapping punctuation first (rather than only quotes) also
    preserves a trailing separator like the `:` in `...token=X: HTTP 404`
    instead of losing it along with the query (U8).
    """
    prefix, core, suffix = _strip_wrapping(token)
    idx = _find_url_offset(core.lower())
    if idx < 0:
        return None
    before, url_part = core[:idx], core[idx:]
    key_candidate = before.rstrip(_QUOTE_CHARS)
    if key_candidate.endswith("="):
        _, key_core, _ = _strip_wrapping(key_candidate[:-1])
        if _is_sensitive_key_segment(key_core):
            return f"{prefix}{before}{_REDACTED}{suffix}"
    return f"{prefix}{before}{_redact_url(url_part)}{suffix}"


def _strip_pair_key(key: str) -> str:
    """Remove punctuation that can wrap a key inside a compact token.

    This is intentionally narrower in scope than _strip_wrapping: square
    brackets stay significant for ordinary value redaction so ``[REDACTED]``
    is idempotent, but are only JSON/list wrapping syntax when recognizing the
    key portion of ``[{\"token\":...}]``.
    """
    return key.strip(_PAIR_KEY_STRIP_CHARS)


def _is_rest_of_line_header(key: str) -> bool:
    """Whether a compact pair key names a credential-only HTTP header."""
    return _strip_pair_key(key).lower() in _REST_OF_LINE_HEADER_NAMES


def _is_sensitive_pair_key(key: str) -> bool:
    """Layer-3 pair-key test, including exact credential-header names."""
    stripped_key = _strip_pair_key(key)
    return stripped_key.lower() in _REST_OF_LINE_HEADER_NAMES or _is_sensitive_key_segment(
        stripped_key
    )


def _pair_key_before_separator(
    core: str, *, separator_at: int, last_terminator: int, last_separator: int
) -> str:
    """Return the most specific plausible key before a pair separator.

    Structural terminators define the normal start of a pair. Looking back to
    the preceding pair separator as a fallback also catches compact nested
    forms such as ``opts=--token=X`` and ``headers={\"Authorization\":X}``,
    whose sensitive pair begins inside the preceding value.
    """
    structural_start = last_terminator + 1
    key = core[structural_start:separator_at]
    if _is_sensitive_pair_key(key) or last_separator <= last_terminator:
        return key
    return core[last_separator + 1 : separator_at]


def _is_auth_scheme_value(value: str) -> bool:
    """Whether a pair's complete in-token value is an auth scheme word."""
    _, core, _ = _strip_wrapping(value)
    return core.lower() in _AUTH_SCHEME_WORDS


def _replace_spans(text: str, replacements: list[tuple[int, int, str]]) -> str:
    """Apply ordered, non-overlapping replacements without using a regex."""
    if not replacements:
        return text

    pieces: list[str] = []
    previous_end = 0
    for start, end, replacement in replacements:
        pieces.extend((text[previous_end:start], replacement))
        previous_end = end
    pieces.append(text[previous_end:])
    return "".join(pieces)


def _find_pair_value_end(core: str, value_start: int) -> int:
    """Find a bounded pair value, preserving an existing redaction marker.

    ``]`` is deliberately a structural terminator, but it is also part of our
    stable ``[REDACTED]`` output. Treating that marker as one existing value
    prevents a second pass from mistaking its closing bracket for a malformed,
    unclosed quote and consuming the rest of the line.
    """
    marker_start = value_start
    if marker_start < len(core) and core[marker_start] in _QUOTE_CHARS:
        marker_start += 1
    if core.startswith(_REDACTED, marker_start):
        value_end = marker_start + len(_REDACTED)
        if (
            value_start < marker_start
            and value_end < len(core)
            and core[value_end] == core[value_start]
        ):
            value_end += 1
        return value_end

    value_end = value_start
    while value_end < len(core) and core[value_end] not in _VALUE_TERMINATORS:
        value_end += 1
    return value_end


def _redact_intra_token_pairs(
    token: str,
) -> tuple[str, str | None, tuple[str, str, str] | None]:
    """Redact every sensitive ``key=value`` / ``key:value`` pair in ``token``.

    The scanner is deliberately structural rather than regex-based. A pair
    value extends to the first member of _VALUE_TERMINATORS or the token end;
    the bound preserves sibling JSON/query fields while letting the scan resume
    after the delimiter. It returns a follow-up mode for a marker whose value
    starts in the next whitespace token, plus the parts needed to redact an
    unclosed quoted value spanning whitespace.
    """
    prefix, core, suffix = _strip_wrapping(token)
    # _strip_wrapping treats quotes/braces/colons at the edge as punctuation.
    # They are meaningful to this scanner: appending them makes `token:`,
    # `{"token":"x"}`, and similar compact forms visible as pairs while the
    # prefix remains available for exact reassembly.
    core = f"{core}{suffix}"

    replacements: list[tuple[int, int, str]] = []
    follow_up: str | None = None
    last_terminator = -1
    last_separator = -1
    i = 0
    while i < len(core):
        char = core[i]
        if char in _VALUE_TERMINATORS:
            last_terminator = i
            i += 1
            continue
        if char not in _PAIR_SEPARATORS:
            i += 1
            continue

        key = _pair_key_before_separator(
            core,
            separator_at=i,
            last_terminator=last_terminator,
            last_separator=last_separator,
        )
        is_header = _is_rest_of_line_header(key)
        if _is_sensitive_pair_key(key):
            value_start = i + 1
            value_end = _find_pair_value_end(core, value_start)
            value = core[value_start:value_end]

            if (not value and value_start == len(core)) or (value and _is_auth_scheme_value(value)):
                follow_up = "remainder" if is_header else "next"
            elif value:
                quote = value[0] if value[0] in _QUOTE_CHARS else None
                if quote is not None and quote not in value[1:]:
                    # Preserve the existing KEY="a b" behavior for a quoted
                    # value whose closing quote falls in a later whitespace
                    # token. The caller consumes those later chunks safely.
                    return token, None, (f"{prefix}{core[:i]}", char, quote)
                replacements.append((value_start, value_end, _redact_value_token(value)))
                # A credential-only header is safe to redact through the rest
                # of its line even when its first value is compactly attached.
                if is_header:
                    follow_up = "remainder"

        last_separator = i
        i += 1

    return f"{prefix}{_replace_spans(core, replacements)}", follow_up, None


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
    segments: list[tuple[bool, str]],
    out: list[str],
    start: int,
    left: str,
    separator: str,
    quote: str,
) -> int:
    """Handle `KEY="a b"` / `KEY:"a b"` values spanning later tokens.

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
    out[start] = f"{left}{separator}{quote}{_REDACTED}{quote}"
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

        redacted_pairs, follow_up, unclosed_quote = _redact_intra_token_pairs(token)
        if unclosed_quote is not None:
            left, separator, quote = unclosed_quote
            i = _consume_quoted_value(segments, out, i, left, separator, quote)
            prev_bare_sensitive_key = False
            continue

        # A preceding split `KEY = value` or header helper can already have
        # redacted this segment. Do not overwrite that result while walking
        # the original token list.
        if out[i] == token:
            out[i] = redacted_pairs
        if follow_up == "remainder":
            i = _redact_remainder_of_line(segments, out, i)
            prev_bare_sensitive_key = False
            continue
        if follow_up == "next":
            i = _redact_next_value_token(segments, out, i)
            prev_bare_sensitive_key = False
            continue

        _, core, _ = _strip_wrapping(token)
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
