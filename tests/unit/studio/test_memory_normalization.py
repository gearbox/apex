"""Artist-memory fields: the wire limits, the one source of limits, and RD6 after NFC.

The service normalizes (NFC, single-line title, block body, tokens) and then re-checks every length
on the normalized values with ``collect_text_issues`` - the limits declared on ``MemoryCreate`` /
``MemoryPatch`` are the only source. Varchar arrays truncate silently on the asyncpg bind cast, so
this re-check is the only guard for ``themes`` / ``imagery``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import pytest

from src.api.schemas.studio import MemoryCreate, MemoryPatch
from src.studio.constraints import (
    MAX_ACTIVE_MEMORIES_PER_ARTIST,
    MEMORY_BODY_MAX,
    MEMORY_CONTEXT_BUDGET,
    MEMORY_IMAGE_MAX,
    MEMORY_IMAGERY_MAX,
    MEMORY_THEME_MAX,
    MEMORY_THEMES_MAX,
    MEMORY_TITLE_MAX,
    TOO_LONG,
    collect_text_issues,
)
from src.studio.validation import normalize_block, normalize_text, normalize_tokens

if TYPE_CHECKING:
    from src.studio.errors import DocumentIssue

pytestmark = pytest.mark.unit

DEVANAGARI_QA = "क़"  # NFC decomposes it to two code points and never recomposes it


def _decode(body: dict[str, object]) -> MemoryCreate:
    return msgspec.json.decode(msgspec.json.encode(body), type=MemoryCreate)


def test_the_documented_limits() -> None:
    assert (MEMORY_TITLE_MAX, MEMORY_BODY_MAX) == (120, 2000)
    assert (MEMORY_THEMES_MAX, MEMORY_THEME_MAX) == (12, 60)
    assert (MEMORY_IMAGERY_MAX, MEMORY_IMAGE_MAX) == (20, 120)
    assert (MAX_ACTIVE_MEMORIES_PER_ARTIST, MEMORY_CONTEXT_BUDGET) == (200, 12_000)


# -- the wire ------------------------------------------------------------------------------------


def test_the_defaults() -> None:
    memory = _decode({"title": "t", "body": "b"})
    assert (memory.salience, memory.themes, memory.imagery, memory.occurred_on) == (3, [], [], None)


@pytest.mark.parametrize(
    "body",
    [
        {"title": "", "body": "b"},
        {"title": "t", "body": ""},
        {"title": "t" * (MEMORY_TITLE_MAX + 1), "body": "b"},
        {"title": "t", "body": "b" * (MEMORY_BODY_MAX + 1)},
        {"title": "t", "body": "b", "themes": ["x"] * (MEMORY_THEMES_MAX + 1)},
        {"title": "t", "body": "b", "themes": ["x" * (MEMORY_THEME_MAX + 1)]},
        {"title": "t", "body": "b", "imagery": ["x"] * (MEMORY_IMAGERY_MAX + 1)},
        {"title": "t", "body": "b", "imagery": ["x" * (MEMORY_IMAGE_MAX + 1)]},
        {"title": "t", "body": "b", "salience": 0},
        {"title": "t", "body": "b", "salience": 6},
        {"title": "t", "body": "b", "occurred_on": "not a date"},
        {"title": "t", "body": "b", "origin": "chat"},  # never accepted from the client
        {"title": "t", "body": "b", "status": "archived"},  # a new memory is always active
        {"title": "t", "body": "b", "artist_id": "x"},
    ],
)
def test_the_wire_refuses(body: dict[str, object]) -> None:
    with pytest.raises(msgspec.ValidationError):
        _decode(body)


def test_the_wire_accepts_the_limits_exactly() -> None:
    memory = _decode(
        {
            "title": "t" * MEMORY_TITLE_MAX,
            "body": "b" * MEMORY_BODY_MAX,
            "themes": ["x" * MEMORY_THEME_MAX] * MEMORY_THEMES_MAX,
            "imagery": ["y" * MEMORY_IMAGE_MAX] * MEMORY_IMAGERY_MAX,
            "salience": 5,
            "occurred_on": "2019-06-04",
        }
    )
    assert memory.salience == 5
    assert memory.occurred_on is not None


def test_a_patch_is_tri_state() -> None:
    patch = msgspec.json.decode(b'{"occurred_on": null, "salience": 2}', type=MemoryPatch)
    assert patch.occurred_on is None  # cleared
    assert patch.salience == 2
    assert patch.title is msgspec.UNSET  # kept
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(b'{"origin": "chat"}', type=MemoryPatch)
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(b'{"status": "deleted"}', type=MemoryPatch)


# -- RD6: the length re-check after normalization ----------------------------------------------


def _issues(value: msgspec.Struct) -> list[tuple[str, str]]:
    issues: list[DocumentIssue] = []
    collect_text_issues(value, issues)
    return [(issue.path, issue.code) for issue in issues]


def test_nfc_expansion_pushes_every_field_past_its_limit_and_is_reported_at_its_wire_path() -> None:
    expanded = DEVANAGARI_QA * (MEMORY_TITLE_MAX // 2 + 1)  # within 120 raw, past it normalized
    assert len(expanded) <= MEMORY_TITLE_MAX
    normalized = normalize_text(expanded)
    assert len(normalized) > MEMORY_TITLE_MAX

    theme = normalize_text(DEVANAGARI_QA * (MEMORY_THEME_MAX // 2 + 1))
    image = normalize_text(DEVANAGARI_QA * (MEMORY_IMAGE_MAX // 2 + 1))
    body = normalize_block(DEVANAGARI_QA * (MEMORY_BODY_MAX // 2 + 1))
    assert body is not None
    memory = MemoryCreate(title=normalized, body=body, themes=["ok", theme], imagery=[image])

    assert _issues(memory) == [
        ("title", TOO_LONG),
        ("body", TOO_LONG),
        ("themes[1]", TOO_LONG),
        ("imagery[0]", TOO_LONG),
    ]


def test_a_value_that_stays_within_its_limit_is_not_reported() -> None:
    memory = MemoryCreate(
        title="t" * MEMORY_TITLE_MAX,
        body="b" * MEMORY_BODY_MAX,
        themes=["x" * MEMORY_THEME_MAX],
        imagery=["y" * MEMORY_IMAGE_MAX],
    )
    assert _issues(memory) == []


def test_only_the_fields_a_patch_sent_are_rechecked() -> None:
    patch = MemoryPatch(themes=[normalize_text(DEVANAGARI_QA * (MEMORY_THEME_MAX // 2 + 1))])
    assert _issues(patch) == [("themes[0]", TOO_LONG)]
    assert _issues(MemoryPatch()) == []  # nothing sent: nothing to check


# -- the token and text normalizers the service uses ----------------------------------------------


def test_themes_and_imagery_are_trimmed_and_checked_for_empty_duplicates_and_brackets() -> None:
    issues: list[DocumentIssue] = []
    result = normalize_tokens(
        ["  Home ", "", "home", "a [b]"], "themes", issues, check_reserved=True
    )
    assert result == ("Home", "", "home", "a [b]")
    assert [(i.path, i.code) for i in issues] == [
        ("themes[1]", "empty_after_normalization"),
        ("themes[2]", "duplicate_tag"),  # case-insensitive
        ("themes[3]", "reserved_characters"),
    ]


def test_the_title_is_one_line_and_the_body_keeps_its_paragraphs() -> None:
    assert normalize_text("  A\n title \t here ") == "A title here"
    assert normalize_block("  one\r\n\r\ntwo \n") == "one\n\ntwo"
    assert normalize_block("  \n ") is None
