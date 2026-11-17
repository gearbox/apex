"""``POST .../runs`` bodies: the tagged union and the brief's normalization (RD6)."""

from __future__ import annotations

import json
from uuid import uuid4

import msgspec
import pytest

from src.studio.errors import StudioDocumentInvalid
from src.studio.run_requests import (
    LyricsRunRequest,
    PlanRunRequest,
    ReviseLyricsRunRequest,
    RunRequest,
)
from src.studio.run_requests import normalize_run_request as normalize_plan_run_request

pytestmark = pytest.mark.unit


def _decode(body: str) -> RunRequest:
    return msgspec.json.decode(body, type=RunRequest)


def test_a_plan_request_decodes_with_defaults() -> None:
    request = _decode('{"kind": "plan", "brief": "a song"}')
    assert request == PlanRunRequest(brief="a song")
    assert request.language is None  # type: ignore[union-attr]
    assert request.memory_ids == ()  # type: ignore[union-attr]
    assert request.auto_fill is True  # type: ignore[union-attr]


def test_the_union_is_tagged_on_kind() -> None:
    assert isinstance(_decode('{"kind": "lyrics"}'), LyricsRunRequest)
    revise = '{"kind": "revise_lyrics", "instruction": "x", "targets": {}}'
    assert isinstance(_decode(revise), ReviseLyricsRunRequest)


@pytest.mark.parametrize(
    "body",
    [
        '{"kind": "plan"}',
        '{"kind": "plan", "brief": ""}',
        '{"kind": "plan", "brief": "x", "extra": 1}',
        '{"kind": "plan", "brief": "x", "language": "NOT A TAG"}',
        '{"kind": "nope", "brief": "x"}',
        '{"brief": "x"}',
    ],
)
def test_malformed_plan_bodies_are_refused(body: str) -> None:
    with pytest.raises(msgspec.ValidationError):
        _decode(body)


def test_the_brief_is_limited_to_4000_characters() -> None:
    with pytest.raises(msgspec.ValidationError):
        _decode('{"kind": "plan", "brief": "' + "x" * 4001 + '"}')
    assert _decode('{"kind": "plan", "brief": "' + "x" * 4000 + '"}')


def test_memory_ids_are_decoded_as_uuids() -> None:
    memory_id = uuid4()
    request = _decode(f'{{"kind": "plan", "brief": "x", "memory_ids": ["{memory_id}"]}}')
    assert request.memory_ids == (memory_id,)  # type: ignore[union-attr]


def test_the_brief_is_normalized() -> None:
    normalized = normalize_plan_run_request(PlanRunRequest(brief="  Café\r\nbar  "))
    assert normalized.brief == "Café\nbar"


def test_a_blank_brief_is_refused() -> None:
    with pytest.raises(StudioDocumentInvalid) as caught:
        normalize_plan_run_request(PlanRunRequest(brief=" \n\t "))
    assert [(i.path, i.code) for i in caught.value.issues] == [
        ("brief", "empty_after_normalization")
    ]


def test_a_brief_that_grows_past_the_limit_under_nfc_is_refused() -> None:
    # U+0958 expands to two code points: 2001 of them are 4002 after NFC.
    request = PlanRunRequest(brief="क़" * 2001)
    with pytest.raises(StudioDocumentInvalid) as caught:
        normalize_plan_run_request(request)
    assert [(i.path, i.code) for i in caught.value.issues] == [
        ("brief", "too_long_after_normalization")
    ]


# -- S1-P2a-B: the lyrics and revise_lyrics bodies ------------------------------------------------

from uuid import UUID  # noqa: E402

from src.studio.documents import SyllableRange  # noqa: E402
from src.studio.run_requests import LineTarget, ReviseTargets  # noqa: E402

SECTION = UUID(int=1)
OTHER = UUID(int=2)


def _revise_body(targets: dict[str, object]) -> str:
    return json.dumps({"kind": "revise_lyrics", "instruction": "x", "targets": targets})


def _issues(request: RunRequest) -> list[tuple[str, str]]:
    with pytest.raises(StudioDocumentInvalid) as caught:
        normalize_plan_run_request(request)
    return [(i.path, i.code) for i in caught.value.issues]


def _revise(**kwargs: object) -> ReviseLyricsRunRequest:
    kwargs.setdefault("instruction", "warmer")
    kwargs.setdefault("targets", ReviseTargets(sections=(SECTION,)))
    return ReviseLyricsRunRequest(**kwargs)  # type: ignore[arg-type]


def test_a_lyrics_request_decodes_with_defaults() -> None:
    request = _decode('{"kind": "lyrics"}')
    assert request == LyricsRunRequest()
    assert isinstance(request, LyricsRunRequest)
    assert (request.brief, request.plan_version_id, request.language) == (None, None, None)
    assert (request.line_syllables, request.avoid_words, request.auto_fill) == (None, (), True)


def test_a_full_lyrics_request_decodes() -> None:
    plan_id = uuid4()
    request = _decode(
        json.dumps(
            {
                "kind": "lyrics",
                "brief": "b",
                "plan_version_id": str(plan_id),
                "language": "ru",
                "line_syllables": {"min": 6, "max": 10},
                "avoid_words": ["a", "b c"],
            }
        )
    )
    assert isinstance(request, LyricsRunRequest)
    assert request.plan_version_id == plan_id
    assert request.line_syllables is not None
    assert (request.line_syllables.min, request.line_syllables.max) == (6, 10)
    assert request.avoid_words == ("a", "b c")


@pytest.mark.parametrize(
    "body",
    [
        '{"kind": "lyrics", "whatever": 1}',
        '{"kind": "lyrics", "language": "NOT A TAG"}',
        '{"kind": "lyrics", "line_syllables": {"min": 0, "max": 5}}',
        '{"kind": "lyrics", "line_syllables": {"min": 1, "max": 33}}',
        '{"kind": "lyrics", "line_syllables": {"min": 1}}',
        '{"kind": "lyrics", "avoid_words": [""]}',
        '{"kind": "lyrics", "avoid_words": ["' + "x" * 41 + '"]}',
        json.dumps({"kind": "lyrics", "avoid_words": ["x"] * 31}),
        '{"kind": "lyrics", "brief": "' + "x" * 4001 + '"}',
        '{"kind": "lyrics", "plan_version_id": "not-a-uuid"}',
    ],
)
def test_malformed_lyrics_bodies_are_refused(body: str) -> None:
    with pytest.raises(msgspec.ValidationError):
        _decode(body)


@pytest.mark.parametrize(
    "body",
    [
        '{"kind": "revise_lyrics", "targets": {}}',  # no instruction
        '{"kind": "revise_lyrics", "instruction": "x"}',  # no targets
        '{"kind": "revise_lyrics", "instruction": "", "targets": {}}',
        '{"kind": "revise_lyrics", "instruction": "' + "x" * 1001 + '", "targets": {}}',
        '{"kind": "revise_lyrics", "instruction": "x", "targets": {"extra": 1}}',
        '{"kind": "revise_lyrics", "instruction": "x", "extra": 1, "targets": {}}',
        _revise_body({"lines": [{"section_id": str(SECTION), "line": -1}]}),
        _revise_body({"lines": [{"section_id": str(SECTION), "line": 64}]}),
        _revise_body({"lines": [{"section_id": str(SECTION)}]}),
    ],
)
def test_malformed_revise_bodies_are_refused(body: str) -> None:
    with pytest.raises(msgspec.ValidationError):
        _decode(body)


def test_the_targets_limits_are_64_each() -> None:
    sections = [str(UUID(int=i + 1)) for i in range(65)]
    with pytest.raises(msgspec.ValidationError):
        _decode(_revise_body({"sections": sections}))
    assert _decode(_revise_body({"sections": sections[:64]}))


def test_a_revise_body_decodes_with_zero_based_lines() -> None:
    request = _decode(
        _revise_body({"sections": [str(SECTION)], "lines": [{"section_id": str(OTHER), "line": 0}]})
    )
    assert isinstance(request, ReviseLyricsRunRequest)
    assert request.targets.sections == (SECTION,)
    assert request.targets.lines == (LineTarget(section_id=OTHER, line=0),)


def test_a_lyrics_request_is_normalized() -> None:
    request = LyricsRunRequest(
        brief="  Café\r\nbar  ",
        avoid_words=("  Baby  ", "my   heart"),
    )
    normalized = normalize_plan_run_request(request)
    assert isinstance(normalized, LyricsRunRequest)
    assert normalized.brief == "Café\nbar"
    assert normalized.avoid_words == ("Baby", "my heart")


def test_a_blank_optional_brief_becomes_none() -> None:
    normalized = normalize_plan_run_request(LyricsRunRequest(brief=" \n\t "))
    assert isinstance(normalized, LyricsRunRequest)
    assert normalized.brief is None


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        (("a", "A"), [("avoid_words[1]", "duplicate_tag")]),  # case-insensitive repeat
        (("a", "  "), [("avoid_words[1]", "empty_after_normalization")]),
        (("a[b]",), [("avoid_words[0]", "reserved_characters")]),
        (("Straße", "STRASSE"), [("avoid_words[1]", "duplicate_tag")]),  # casefold: ß is ss
        (("Straße", "Strasse2"), []),
    ],
)
def test_avoid_words_follow_the_token_rules(
    words: tuple[str, ...], expected: list[tuple[str, str]]
) -> None:
    request = LyricsRunRequest(avoid_words=words)
    if expected:
        assert _issues(request) == expected
    else:
        assert normalize_plan_run_request(request).avoid_words == words  # type: ignore[union-attr]


def test_an_inverted_syllable_range_is_refused_in_both_kinds() -> None:
    inverted = SyllableRange(min=9, max=3)
    expected = [("line_syllables", "syllable_range_inverted")]
    assert _issues(LyricsRunRequest(line_syllables=inverted)) == expected
    assert _issues(_revise(line_syllables=inverted)) == expected


def test_an_avoid_word_that_grows_past_forty_under_nfc_is_refused() -> None:
    # U+0958 expands to two code points: 21 of them are 42 after NFC
    request = LyricsRunRequest(avoid_words=("\N{DEVANAGARI LETTER QA}" * 21,))
    assert _issues(request) == [("avoid_words[0]", "too_long_after_normalization")]


def test_an_instruction_is_normalized_and_must_not_be_blank() -> None:
    normalized = normalize_plan_run_request(_revise(instruction="  Café\r\nwarmer  "))
    assert normalized.instruction == "Café\nwarmer"  # type: ignore[union-attr]
    assert _issues(_revise(instruction=" \n\t ")) == [("instruction", "empty_after_normalization")]


def test_an_instruction_that_grows_past_the_limit_under_nfc_is_refused() -> None:
    request = _revise(instruction="\N{DEVANAGARI LETTER QA}" * 501)
    assert _issues(request) == [("instruction", "too_long_after_normalization")]


def test_targets_must_name_something() -> None:
    assert _issues(_revise(targets=ReviseTargets())) == [("targets", "targets_required")]


def test_a_repeated_target_is_refused_at_its_own_path() -> None:
    targets = ReviseTargets(
        sections=(SECTION, SECTION),
        lines=(
            LineTarget(section_id=OTHER, line=1),
            LineTarget(section_id=OTHER, line=2),
            LineTarget(section_id=OTHER, line=1),
        ),
    )
    assert _issues(_revise(targets=targets)) == [
        ("targets.sections[1]", "duplicate_target"),
        ("targets.lines[2]", "duplicate_target"),
    ]


def test_a_line_inside_a_section_that_is_rewritten_anyway_is_refused() -> None:
    targets = ReviseTargets(
        sections=(SECTION,),
        lines=(LineTarget(section_id=OTHER, line=0), LineTarget(section_id=SECTION, line=3)),
    )
    assert _issues(_revise(targets=targets)) == [("targets.lines[1]", "overlapping_targets")]


def test_every_problem_is_reported_together() -> None:
    request = _revise(
        instruction=" ",
        avoid_words=("a", "a"),
        line_syllables=__import__("src.studio.documents", fromlist=["SyllableRange"]).SyllableRange(
            min=5, max=2
        ),
        targets=ReviseTargets(),
    )
    assert sorted(_issues(request)) == sorted(
        [
            ("instruction", "empty_after_normalization"),
            ("avoid_words[1]", "duplicate_tag"),
            ("line_syllables", "syllable_range_inverted"),
            ("targets", "targets_required"),
        ]
    )


def test_normalization_keeps_the_kind_and_is_idempotent() -> None:
    request = _revise(
        avoid_words=("baby",),
        targets=ReviseTargets(lines=(LineTarget(section_id=SECTION, line=2),)),
    )
    once = normalize_plan_run_request(request)
    twice = normalize_plan_run_request(once)
    assert type(once) is ReviseLyricsRunRequest
    assert once == twice


def test_the_stored_form_decodes_back_to_the_same_request() -> None:
    request = _revise(
        avoid_words=("baby",),
        targets=ReviseTargets(sections=(SECTION,), lines=(LineTarget(section_id=OTHER, line=2),)),
    )
    stored = msgspec.to_builtins(request)
    assert stored["kind"] == "revise_lyrics"
    assert msgspec.convert(stored, RunRequest) == request
    lyrics = LyricsRunRequest(plan_version_id=SECTION, avoid_words=("a",))
    assert msgspec.convert(msgspec.to_builtins(lyrics), RunRequest) == lyrics


def test_a_plan_request_normalizes_byte_for_byte_as_before() -> None:
    """The plan request's stored/hashed form must not change with the new union members."""
    normalized = normalize_plan_run_request(PlanRunRequest(brief="  a song  "))
    assert msgspec.json.encode(normalized) == (
        b'{"kind":"plan","brief":"a song","language":null,"memory_ids":[],"auto_fill":true}'
    )


def test_run_inputs_carry_ids_and_the_language_only() -> None:
    from src.studio.run_requests import RunInputs

    inputs = RunInputs(plan_version_id=SECTION, language="en")
    assert msgspec.to_builtins(inputs) == {
        "plan_version_id": str(SECTION),
        "language": "en",
        "lyrics_version_id": None,
    }
    assert msgspec.convert(msgspec.to_builtins(inputs), RunInputs) == inputs
