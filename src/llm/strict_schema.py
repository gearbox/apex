"""JSON Schema for a draft ``msgspec.Struct`` in the strict form structured-output APIs accept.

Strict mode wants a root object, every property required and ``additionalProperties: false`` on
every object. A draft Struct therefore declares **no defaults** (a defaulted field would be
optional on the wire); the converter raises on one instead of silently changing the contract.

Length, size, pattern and numeric-bound keywords are dropped: some providers reject them in strict
mode, and the server validates those limits itself after the reply arrives (P2a-7), then asks the
model to fix what is wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import msgspec

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["DROPPED_KEYWORDS", "to_strict_schema"]

DROPPED_KEYWORDS: Final = frozenset(
    {
        "default",
        "title",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "format",
    }
)
"""Annotation / validation keywords removed from every *schema* (never from a property map)."""

_REF_TEMPLATE: Final = "#/$defs/{name}"


def to_strict_schema(draft: type[msgspec.Struct]) -> dict[str, Any]:
    """The strict JSON Schema of ``draft``.

    Args:
        draft: A ``msgspec.Struct`` whose fields all lack defaults.

    Returns:
        A root object schema; the other definitions live under ``$defs``.

    Raises:
        ValueError: A field is not required (it declares a default) - the message names the field.
    """
    schemas, components = msgspec.json.schema_components((draft,), ref_template=_REF_TEMPLATE)
    root = schemas[0]
    if "$ref" in root:
        name = str(root["$ref"]).rsplit("/", 1)[-1]
        root = components[name]
        components = {key: value for key, value in components.items() if key != name}
    result = _strictify(root)
    if components:
        result["$defs"] = {name: _strictify(schema) for name, schema in components.items()}
    return result


def _strictify(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Copy ``schema`` as a strict schema (recursing into every sub-schema position)."""
    out: dict[str, Any] = {}
    for keyword, value in schema.items():
        if keyword in DROPPED_KEYWORDS:
            continue
        if keyword == "properties":
            # Property NAMES are data: a field called ``title`` must survive. Only the schemas
            # *under* the names are cleaned.
            out[keyword] = {name: _strictify(sub) for name, sub in value.items()}
        elif keyword in ("items", "additionalProperties") and isinstance(value, dict):
            out[keyword] = _strictify(value)
        elif keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
            out[keyword] = [_strictify(sub) for sub in value]
        elif keyword == "$defs":
            out[keyword] = {name: _strictify(sub) for name, sub in value.items()}
        else:
            out[keyword] = value
    if "properties" in out:  # a Struct; a ``dict`` field has none and is left as it is
        properties = out["properties"]
        required = set(out.get("required", ()))
        optional = [name for name in properties if name not in required]
        if optional:
            raise ValueError(
                "a draft struct must not declare defaults; not required: " + ", ".join(optional)
            )
        out["required"] = list(properties)
        out["additionalProperties"] = False
    return out
