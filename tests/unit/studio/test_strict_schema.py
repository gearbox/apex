"""``to_strict_schema``: the strict JSON schema of a draft Struct (invariant 14)."""

from __future__ import annotations

from typing import Annotated, Any

import msgspec
import pytest

from src.studio.llm.strict_schema import DROPPED_KEYWORDS, to_strict_schema
from src.studio.planner.drafts import PlanDraft, PlanSectionDraft

pytestmark = pytest.mark.unit


def _objects(schema: Any) -> list[dict[str, Any]]:
    """Every object schema (a dict with ``properties``) anywhere in ``schema``."""
    found: list[dict[str, Any]] = []
    if isinstance(schema, dict):
        if "properties" in schema:
            found.append(schema)
        for value in schema.values():
            found.extend(_objects(value))
    elif isinstance(schema, list):
        for value in schema:
            found.extend(_objects(value))
    return found


def _schemas(schema: Any) -> list[dict[str, Any]]:
    """Every sub-schema, never descending *into* a ``properties`` map's keys (those are names)."""
    found: list[dict[str, Any]] = []
    if isinstance(schema, dict):
        found.append(schema)
        for keyword, value in schema.items():
            if keyword == "properties" or keyword == "$defs":
                for sub in value.values():
                    found.extend(_schemas(sub))
            else:
                found.extend(_schemas(value))
    elif isinstance(schema, list):
        for value in schema:
            found.extend(_schemas(value))
    return found


@pytest.mark.parametrize("draft", [PlanDraft, PlanSectionDraft], ids=lambda d: d.__name__)
def test_every_draft_struct_converts(draft: type[msgspec.Struct]) -> None:
    assert to_strict_schema(draft)["type"] == "object"


class TestPlanDraft:
    schema = to_strict_schema(PlanDraft)

    def test_root_is_an_object_not_a_reference(self) -> None:
        assert self.schema["type"] == "object"
        assert "$ref" not in self.schema

    def test_a_property_named_title_survives(self) -> None:
        assert "title" in self.schema["properties"]
        assert "title" in self.schema["required"]

    def test_every_property_of_every_object_is_required(self) -> None:
        objects = _objects(self.schema)
        assert len(objects) >= 2  # the root and the section definition
        for obj in objects:
            assert sorted(obj["required"]) == sorted(obj["properties"])

    def test_additional_properties_are_forbidden_everywhere(self) -> None:
        for obj in _objects(self.schema):
            assert obj["additionalProperties"] is False

    def test_no_dropped_keyword_survives_as_a_schema_keyword(self) -> None:
        for sub in _schemas(self.schema):
            assert not DROPPED_KEYWORDS & sub.keys() - {"properties"}, sub.keys()

    def test_definitions_stay_under_defs(self) -> None:
        assert "StudioSectionKind" in self.schema["$defs"]
        assert "PlanSectionDraft" in self.schema["$defs"]
        assert "PlanDraft" not in self.schema["$defs"]

    def test_a_sample_reply_decodes(self) -> None:
        reply = {
            "refusal": None,
            "title": "T",
            "tempo_bpm": None,
            "instrumental": False,
            "tags": [],
            "avoid": [],
            "sections": [
                {
                    "key": "s1",
                    "kind": "verse",
                    "label": None,
                    "duration_seconds": None,
                    "tags": [],
                    "delivery": [],
                    "instrumental": False,
                    "notes": None,
                }
            ],
        }
        decoded = msgspec.convert(reply, PlanDraft)
        assert decoded.sections[0].key == "s1"


class WithDefault(msgspec.Struct):
    name: str
    flag: bool = False


class Limited(msgspec.Struct):
    text: Annotated[str, msgspec.Meta(min_length=1, max_length=5, pattern="^a")]
    count: Annotated[int, msgspec.Meta(ge=1, le=9)]
    items: Annotated[list[int], msgspec.Meta(max_length=3)]


class Odd(msgspec.Struct):
    pattern: str
    default: int


def test_a_struct_with_a_default_is_refused_naming_the_field() -> None:
    with pytest.raises(ValueError, match="flag"):
        to_strict_schema(WithDefault)


def test_length_and_pattern_keywords_are_dropped() -> None:
    schema = to_strict_schema(Limited)
    assert schema["properties"]["text"] == {"type": "string"}
    assert schema["properties"]["count"] == {"type": "integer"}
    assert schema["properties"]["items"] == {"type": "array", "items": {"type": "integer"}}


def test_a_field_named_like_a_keyword_is_kept_next_to_a_dropped_keyword() -> None:
    schema = to_strict_schema(Odd)
    assert set(schema["properties"]) == {"pattern", "default"}
    assert sorted(schema["required"]) == ["default", "pattern"]
