"""A memory draft is checked like a memory: the same normalisers, the same limits (C8, RD6)."""

from __future__ import annotations

import msgspec.inspect as mi
import pytest

from src.api.schemas.studio import MemoryCreate
from src.llm.strict_schema import to_strict_schema
from src.studio.chat.draft import (
    MemoryContent,
    MemoryDraftDraft,
    MemoryDraftRefusal,
    check_memory_draft,
)
from src.studio.constraints import (
    MEMORY_BODY_MAX,
    MEMORY_IMAGE_MAX,
    MEMORY_IMAGERY_MAX,
    MEMORY_THEME_MAX,
    MEMORY_THEMES_MAX,
    MEMORY_TITLE_MAX,
)

pytestmark = pytest.mark.unit


def draft(**overrides: object) -> MemoryDraftDraft:
    base: dict[str, object] = {
        "refusal": None,
        "title": "Rooftop night",
        "body": "We sang until dawn.",
        "themes": ["home"],
        "imagery": ["wet asphalt"],
    }
    return MemoryDraftDraft(**{**base, **overrides})  # type: ignore[arg-type]


def paths(draft_: MemoryDraftDraft) -> set[tuple[str, str]]:
    _, issues = check_memory_draft(draft_)
    return {(i.path, i.code) for i in issues}


class TestAValidDraft:
    def test_it_becomes_memory_content(self) -> None:
        content, issues = check_memory_draft(draft())
        assert issues == ()
        assert content is not None
        assert content.to_json() == {
            "title": "Rooftop night",
            "body": "We sang until dawn.",
            "themes": ["home"],
            "imagery": ["wet asphalt"],
        }

    def test_it_is_normalized_like_a_memory(self) -> None:
        content, _ = check_memory_draft(
            draft(
                title="  Rooftop \n night  ", body="  line one\r\nline two  \n", themes=["  home "]
            )
        )
        assert content is not None
        assert content.title == "Rooftop night"
        assert content.body == "line one\nline two"
        assert content.themes == ["home"]

    def test_empty_lists_are_allowed(self) -> None:
        content, issues = check_memory_draft(draft(themes=[], imagery=[]))
        assert issues == ()
        assert content is not None


class TestAnInvalidDraft:
    def test_every_issue_is_reported_with_its_path_never_its_text(self) -> None:
        found = paths(draft(title="  ", body="", themes=["ok", "OK", "a[b]", " "], imagery=[""]))
        assert found == {
            ("title", "empty_after_normalization"),
            ("body", "empty_after_normalization"),
            ("themes[1]", "duplicate_tag"),
            ("themes[2]", "reserved_characters"),
            ("themes[3]", "empty_after_normalization"),
            ("imagery[0]", "empty_after_normalization"),
        }

    def test_a_title_past_its_limit(self) -> None:
        assert ("title", "too_long_after_normalization") in paths(
            draft(title="t" * (MEMORY_TITLE_MAX + 1))
        )

    def test_a_body_past_its_limit(self) -> None:
        assert ("body", "too_long_after_normalization") in paths(
            draft(body="b" * (MEMORY_BODY_MAX + 1))
        )

    def test_too_many_themes_and_too_long_a_theme(self) -> None:
        many = [f"theme {i}" for i in range(MEMORY_THEMES_MAX + 1)]
        assert ("themes", "too_many_items") in paths(draft(themes=many))
        assert ("themes[0]", "too_long_after_normalization") in paths(
            draft(themes=["x" * (MEMORY_THEME_MAX + 1)])
        )

    def test_too_many_imagery_items_and_too_long_an_item(self) -> None:
        many = [f"image {i}" for i in range(MEMORY_IMAGERY_MAX + 1)]
        assert ("imagery", "too_many_items") in paths(draft(imagery=many))
        assert ("imagery[0]", "too_long_after_normalization") in paths(
            draft(imagery=["x" * (MEMORY_IMAGE_MAX + 1)])
        )

    def test_nfc_expansion_is_caught(self) -> None:
        # U+0958 decomposes to two code points under NFC: 60 of them are 120 characters
        assert ("themes[0]", "too_long_after_normalization") in paths(
            draft(themes=["क़" * MEMORY_THEME_MAX])
        )


def _limits(cls: type, field: str) -> tuple[object, ...]:
    """The declared limits of one field, as ``msgspec.inspect`` reports them."""
    info = mi.type_info(cls)
    assert isinstance(info, mi.StructType)
    (found,) = [f for f in info.fields if f.name == field]
    node = found.type
    out: list[object] = []
    while True:
        if isinstance(node, mi.StrType):
            out.append(("str", node.min_length, node.max_length))
            break
        if isinstance(node, mi.ListType):
            out.append(("list", node.min_length, node.max_length))
            node = node.item_type
            continue
        break
    return tuple(out)


@pytest.mark.parametrize("field", ["title", "body", "themes", "imagery"])
def test_the_draft_limits_are_the_memorys_own(field: str) -> None:
    """``MemoryContent`` declares what ``MemoryCreate`` does, so a draft never passes a limit the
    memory it becomes would refuse."""
    assert _limits(MemoryContent, field) == _limits(MemoryCreate, field)


def test_the_draft_struct_is_a_strict_schema_with_everything_required() -> None:
    schema = to_strict_schema(MemoryDraftDraft)
    assert set(schema["required"]) == {"refusal", "title", "body", "themes", "imagery"}
    assert schema["additionalProperties"] is False


def test_the_refusals_are_the_two_the_prompt_names() -> None:
    assert {r.value for r in MemoryDraftRefusal} == {"content_policy", "nothing_to_remember"}
