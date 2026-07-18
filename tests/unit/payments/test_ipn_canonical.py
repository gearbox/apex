"""Golden tests for the format-agnostic IPN canonicalizer (D1).

Fixtures below are (wire_body, expected_canonical_bytes) pairs generated and
hand-verified against NowPayments' actual signer shape:
``JSON.stringify(sortObject(obj))`` — recursively sort keys, then compact
`JSON.stringify`. The JS used to produce them:

    function sortObject(obj) {
      if (Array.isArray(obj)) return obj.map(sortObject);
      if (obj !== null && typeof obj === "object") {
        return Object.keys(obj).sort().reduce((acc, k) => {
          acc[k] = sortObject(obj[k]);
          return acc;
        }, {});
      }
      return obj;
    }
    const canonical = (o) => JSON.stringify(sortObject(o));
    // WIRE  = JSON.stringify(obj)     (insertion/wire key order, unsorted)
    // CANON = canonical(obj)          (expected canonicalizer output)

Each fixture covers: unsorted wire key order, exotic float lexemes (`0.1`,
`4.98842014`, exponent form `1e-7`, a large integer), a `null` field, and a
non-ascii character in `order_description` (locks the `ensure_ascii=False`
choice — Node's `JSON.stringify` never escapes non-ASCII by default).

Note: in the "classic"/"mixed" fixtures, the wire body's `large_field`
already reads `123456789012345680`, not the originally-intended
`...678` — JS numbers are float64, so NowPayments' own signer already lost
that precision by the time it serialized the wire body. The canonicalizer
only has to preserve whatever lexeme is actually on the wire, not recover
a "true" value that was never transmitted — and it does, because it never
reparses a number through a float.
"""

from __future__ import annotations

from src.api.services.payments.ipn_canonical import RawNumber, canonical_bytes, parse_ipn_body

# All-Strings: every leaf value is a quoted JSON string.
_ALL_STRINGS_WIRE = (
    b'{"order_description":"caf\xc3\xa9 order \xe2\x80\x94 5 tokens",'
    b'"payment_status":"finished","actually_paid":"0.1","pay_amount":"4.98842014",'
    b'"large_field":"123456789012345678","exp_field":"1e-7","extra":null,"aaa_first":"z"}'
)
_ALL_STRINGS_CANON = (
    b'{"aaa_first":"z","actually_paid":"0.1","exp_field":"1e-7","extra":null,'
    b'"large_field":"123456789012345678",'
    b'"order_description":"caf\xc3\xa9 order \xe2\x80\x94 5 tokens",'
    b'"pay_amount":"4.98842014","payment_status":"finished"}'
)

# Classic: bare JSON number literals throughout, including a nested object
# (mirrors the real incident's nested `fee` object).
_CLASSIC_WIRE = (
    b'{"order_description":"caf\xc3\xa9 order \xe2\x80\x94 5 tokens",'
    b'"payment_status":"finished","actually_paid":0.1,"pay_amount":4.98842014,'
    b'"large_field":123456789012345680,"exp_field":1e-7,"extra":null,"aaa_first":"z",'
    b'"fee":{"withdrawalFee":0.068969,"depositFee":0.034133,"serviceFee":0.04883,'
    b'"currency":"usdtmatic"}}'
)
_CLASSIC_CANON = (
    b'{"aaa_first":"z","actually_paid":0.1,"exp_field":1e-7,"extra":null,'
    b'"fee":{"currency":"usdtmatic","depositFee":0.034133,"serviceFee":0.04883,'
    b'"withdrawalFee":0.068969},"large_field":123456789012345680,'
    b'"order_description":"caf\xc3\xa9 order \xe2\x80\x94 5 tokens",'
    b'"pay_amount":4.98842014,"payment_status":"finished"}'
)

# Mixed: top-level strings, nested bare numbers — the hypothesized real-world
# case (the incident's own `fee` object was exactly this shape).
_MIXED_WIRE = (
    b'{"order_description":"caf\xc3\xa9 order \xe2\x80\x94 5 tokens",'
    b'"payment_status":"finished","actually_paid":"0.1","pay_amount":"4.98842014",'
    b'"extra":null,"aaa_first":"z","fee":{"withdrawalFee":0.068969,"depositFee":0.034133,'
    b'"serviceFee":0.04883,"large_field":123456789012345680,"exp_field":1e-7,'
    b'"currency":"usdtmatic"}}'
)
_MIXED_CANON = (
    b'{"aaa_first":"z","actually_paid":"0.1","extra":null,'
    b'"fee":{"currency":"usdtmatic","depositFee":0.034133,"exp_field":1e-7,'
    b'"large_field":123456789012345680,"serviceFee":0.04883,"withdrawalFee":0.068969},'
    b'"order_description":"caf\xc3\xa9 order \xe2\x80\x94 5 tokens",'
    b'"pay_amount":"4.98842014","payment_status":"finished"}'
)

_FIXTURES = [
    ("all_strings", _ALL_STRINGS_WIRE, _ALL_STRINGS_CANON),
    ("classic", _CLASSIC_WIRE, _CLASSIC_CANON),
    ("mixed", _MIXED_WIRE, _MIXED_CANON),
]


def test_fixtures_are_valid_utf8_and_distinct_from_canon() -> None:
    """Sanity check on the fixtures themselves before trusting them as golden data."""
    for name, wire, canon in _FIXTURES:
        wire.decode("utf-8")
        canon.decode("utf-8")
        assert wire != canon, f"{name}: wire and canon fixtures must differ (unsorted vs sorted)"


def test_canonical_bytes_matches_js_stringify_sortobject() -> None:
    for name, wire, expected_canon in _FIXTURES:
        parsed = parse_ipn_body(wire)
        assert canonical_bytes(parsed) == expected_canon, name


def test_canonicalizer_is_idempotent_on_already_canonical_input() -> None:
    """Re-canonicalizing a canonical body must be a byte-identical no-op."""
    for name, _wire, canon in _FIXTURES:
        assert canonical_bytes(parse_ipn_body(canon)) == canon, name


def test_raw_number_preserves_exact_lexeme() -> None:
    parsed = parse_ipn_body(b'{"a": 4.98842014, "b": 5, "c": 1e-7, "d": 123456789012345680}')
    assert isinstance(parsed, dict)
    assert isinstance(parsed["a"], RawNumber)
    assert str(parsed["a"]) == "4.98842014"
    assert str(parsed["b"]) == "5"
    assert str(parsed["c"]) == "1e-7"
    assert str(parsed["d"]) == "123456789012345680"


def test_raw_number_is_emitted_unquoted_string_is_quoted() -> None:
    parsed = parse_ipn_body(b'{"n": 5, "s": "5"}')
    assert canonical_bytes(parsed) == b'{"n":5,"s":"5"}'


def test_booleans_and_lists_round_trip() -> None:
    parsed = parse_ipn_body(b'{"b": [true, false, null, 1, "x"]}')
    assert canonical_bytes(parsed) == b'{"b":[true,false,null,1,"x"]}'


def test_nested_object_keys_sorted_at_every_level() -> None:
    parsed = parse_ipn_body(b'{"z": {"b": 1, "a": 2}, "a": {"z": 1}}')
    assert canonical_bytes(parsed) == b'{"a":{"z":1},"z":{"a":2,"b":1}}'
