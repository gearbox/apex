"""The lyrics draft Structs are strict-schema-clean (invariant 19): no defaults, all required."""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from src.llm.strict_schema import to_strict_schema
from src.studio.planner.drafts import (
    LinePatchDraft,
    LyricLineDraft,
    LyricsDraft,
    LyricsPatchDraft,
    LyricsRevisionDraft,
    LyricsSectionDraft,
)

pytestmark = pytest.mark.unit

DRAFTS = [
    LyricLineDraft,
    LyricsSectionDraft,
    LyricsDraft,
    LinePatchDraft,
    LyricsPatchDraft,
    LyricsRevisionDraft,
]
ROOTS = [LyricsDraft, LyricsPatchDraft, LyricsRevisionDraft]


def _objects(schema: Any) -> list[dict[str, Any]]:
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


@pytest.mark.parametrize("draft", DRAFTS, ids=lambda d: d.__name__)
def test_every_new_draft_converts_and_declares_no_defaults(draft: type[msgspec.Struct]) -> None:
    assert to_strict_schema(draft)["type"] == "object"
    assert all(
        field.default is msgspec.NODEFAULT and field.default_factory is msgspec.NODEFAULT
        for field in msgspec.structs.fields(draft)
    )


@pytest.mark.parametrize("draft", ROOTS, ids=lambda d: d.__name__)
def test_every_property_of_every_object_is_required_and_nothing_extra_is_allowed(
    draft: type[msgspec.Struct],
) -> None:
    schema = to_strict_schema(draft)
    objects = _objects(schema)
    assert objects
    for obj in objects:
        assert sorted(obj["required"]) == sorted(obj["properties"])
        assert obj["additionalProperties"] is False
    assert "refusal" in schema["required"]


def test_sample_replies_decode_into_the_drafts() -> None:
    line = {"text": "la la", "delivery": []}
    patch_line = {"key": "s1", "line": 1, "text": "la", "delivery": ["soft"]}
    assert (
        msgspec.convert(
            {"refusal": None, "sections": [{"key": "s1", "lines": [line]}]}, LyricsDraft
        )
        .sections[0]
        .lines[0]
        .text
        == "la la"
    )
    assert (
        msgspec.convert({"refusal": None, "lines": [patch_line]}, LyricsPatchDraft).lines[0].line
        == 1
    )
    revision = msgspec.convert(
        {
            "refusal": "content_policy",
            "sections": [{"key": "s1", "lines": [line]}],
            "lines": [patch_line],
        },
        LyricsRevisionDraft,
    )
    assert revision.refusal is not None


@pytest.mark.parametrize(
    ("draft", "reply"),
    [
        (LyricsDraft, {"refusal": None}),  # a missing field is not defaulted
        (LyricsDraft, {"refusal": None, "sections": [], "extra": 1}),
        (LyricsPatchDraft, {"refusal": None, "lines": [{"key": "s1", "line": 1, "text": "x"}]}),
        (LyricLineDraft, {"text": "x"}),
        (LyricsRevisionDraft, {"refusal": None, "sections": []}),
    ],
)
def test_a_reply_that_leaves_a_field_out_or_adds_one_does_not_decode(
    draft: type[msgspec.Struct], reply: dict[str, object]
) -> None:
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert(reply, draft)
