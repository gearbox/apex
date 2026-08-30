"""Format-agnostic canonical serialization for NowPayments IPN bodies.

NowPayments signs a JSON object with keys sorted at every level and no
whitespace. Their dashboard can deliver that object with numbers as JSON
number literals ("Classic"), as quoted strings ("All-Strings"), or a mix of
both — but never changes how a given leaf itself is rendered. ``canonical_bytes``
reproduces their signed bytes for any of these by re-serializing with sorted
keys while leaving every leaf's original lexeme untouched: numbers are
re-emitted verbatim (not reformatted through a float round-trip, which risks
drift on exotic lexemes like large integers or exponent notation), and
strings are re-quoted byte-for-byte equivalent to their input.
"""

from __future__ import annotations

import json


class RawNumber(str):
    """A JSON number literal preserved as its exact original lexeme.

    Behaves as a plain ``str`` everywhere (equality, ``Decimal(str(x))``,
    JSON re-encoding as a quoted string) except in :func:`canonical_bytes`,
    which detects this subclass and emits it unquoted.
    """

    __slots__ = ()


type JSONValue = bool | RawNumber | str | list[JSONValue] | dict[str, JSONValue] | None


def parse_ipn_body(raw: bytes) -> JSONValue:
    """Parse an IPN body, preserving every number's original lexeme as a ``RawNumber``."""
    return json.loads(raw, parse_float=RawNumber, parse_int=RawNumber)  # type: ignore[no-any-return]


def canonical_bytes(parsed: JSONValue) -> bytes:
    """Re-serialize a ``parse_ipn_body`` result with keys sorted at every level.

    Byte-identical to NowPayments' signed form regardless of whether the
    wire body used Classic, All-Strings, or mixed number formatting — no
    leaf's representation changes, only key order and whitespace.
    """
    return _canonical(parsed).encode()


def _canonical(value: JSONValue) -> str:
    if isinstance(value, RawNumber):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: kv[0])
        body = ",".join(f"{json.dumps(k, ensure_ascii=False)}:{_canonical(v)}" for k, v in items)
        return "{" + body + "}"
    if isinstance(value, list):
        return "[" + ",".join(_canonical(item) for item in value) + "]"
    raise TypeError(f"Unsupported JSON value type in IPN body: {type(value)!r}")
