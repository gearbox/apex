"""The pure realized-timeline module (S3a W1): validation, completeness, hash and limits.

The song under test: a verse and a chorus with two lyric lines each, then an instrumental
section, over 60 s of audio. :func:`complete_timeline` times all of it, with touching spans on
purpose (half-open intervals), so every negative test changes exactly one thing and asserts the
**whole** issue list.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any
from uuid import UUID

import msgspec
import pytest

from src.core.enums import StudioSectionKind as K
from src.core.media_limits import AUDIO_MAX_MS, AUDIO_MAX_SECONDS
from src.studio.errors import StudioDocumentSchemaError
from src.studio.timeline import (
    TIMELINE_SCHEMA_VERSION,
    LineSpan,
    SectionSpan,
    TimelineDocument,
    WordSpan,
    check_timeline_limits,
    decode_stored_timeline,
    is_complete,
    normalize_timeline,
    timeline_content,
    timeline_hash,
    validate_timeline,
)
from src.studio.validation import MAX_DOCUMENT_BYTES
from tests.studio_support import make_lyrics, make_plan, make_section

if TYPE_CHECKING:
    from src.studio.documents import LyricsDocument, PlanDocument
    from src.studio.errors import DocumentIssue

pytestmark = pytest.mark.unit

VERSE = UUID("00000000-0000-7000-8000-00000000000a")
CHORUS = UUID("00000000-0000-7000-8000-00000000000b")
SOLO = UUID("00000000-0000-7000-8000-00000000000c")
STRANGER = UUID("00000000-0000-7000-8000-0000000000ff")
DURATION_MS = 60_000


def plan() -> PlanDocument:
    return make_plan(
        make_section(K.VERSE, section_id=VERSE),
        make_section(K.CHORUS, section_id=CHORUS),
        make_section(K.INSTRUMENTAL, section_id=SOLO),
    )


def lyrics() -> LyricsDocument:
    return make_lyrics((VERSE, ["first line", "second line"]), (CHORUS, ["hook one", "hook two"]))


def span(section_id: UUID, start_ms: int, end_ms: int) -> SectionSpan:
    return SectionSpan(section_id=section_id, start_ms=start_ms, end_ms=end_ms)


def line(
    section_id: UUID, line_index: int, start_ms: int, end_ms: int, *words: WordSpan
) -> LineSpan:
    return LineSpan(
        section_id=section_id,
        line_index=line_index,
        start_ms=start_ms,
        end_ms=end_ms,
        words=words,
    )


def word(text: str, start_ms: int, end_ms: int, *, inferred: bool = False) -> WordSpan:
    return WordSpan(text=text, start_ms=start_ms, end_ms=end_ms, inferred=inferred)


SECTIONS = (span(VERSE, 0, 20_000), span(CHORUS, 20_000, 40_000), span(SOLO, 40_000, 60_000))
LINES = (
    line(VERSE, 0, 1_000, 5_000, word("first", 1_000, 2_000), word("line", 2_000, 5_000)),
    line(VERSE, 1, 5_000, 9_000),
    line(CHORUS, 0, 21_000, 25_000),
    line(CHORUS, 1, 25_000, 30_000),
)


def complete_timeline(
    *, sections: tuple[SectionSpan, ...] = SECTIONS, lines: tuple[LineSpan, ...] = LINES
) -> TimelineDocument:
    return TimelineDocument(sections=sections, lines=lines)


def replaced[T](items: tuple[T, ...], index: int, item: T) -> tuple[T, ...]:
    return (*items[:index], item, *items[index + 1 :])


def check(
    document: TimelineDocument,
    *,
    with_lyrics: bool = True,
    duration_ms: int = DURATION_MS,
) -> list[tuple[str, str]]:
    issues = validate_timeline(
        document,
        duration_ms=duration_ms,
        plan=plan(),
        lyrics=lyrics() if with_lyrics else None,
    )
    return pairs(issues)


def pairs(issues: list[DocumentIssue]) -> list[tuple[str, str]]:
    return [(issue.path, issue.code) for issue in issues]


# -- valid timelines ---------------------------------------------------------------------------


class TestValid:
    def test_a_complete_timeline_with_touching_spans_has_no_issues(self) -> None:
        assert check(complete_timeline()) == []

    def test_a_partial_timeline_is_valid(self) -> None:
        document = TimelineDocument(
            sections=(span(CHORUS, 20_000, 40_000),), lines=(line(CHORUS, 1, 25_000, 30_000),)
        )

        assert check(document) == []

    def test_an_empty_timeline_is_valid(self) -> None:
        assert check(TimelineDocument(sections=())) == []

    def test_gaps_between_sections_lines_and_words_are_allowed(self) -> None:
        document = TimelineDocument(
            sections=(span(VERSE, 3_000, 10_000), span(SOLO, 50_000, 55_000)),
            lines=(
                line(VERSE, 0, 4_000, 5_000, word("a", 4_100, 4_200), word("b", 4_800, 4_900)),
                line(VERSE, 1, 7_000, 8_000),
            ),
        )

        assert check(document) == []

    def test_a_span_may_reach_the_very_end_of_the_audio(self) -> None:
        assert check(TimelineDocument(sections=(span(SOLO, 0, DURATION_MS),))) == []

    def test_word_text_is_never_compared_with_the_line_text(self) -> None:
        document = complete_timeline(
            lines=replaced(
                LINES, 0, line(VERSE, 0, 1_000, 5_000, word("completely different", 1_000, 2_000))
            )
        )

        assert check(document) == []

    def test_an_inferred_word_is_validated_like_any_other(self) -> None:
        good = complete_timeline(
            lines=replaced(
                LINES, 0, line(VERSE, 0, 1_000, 5_000, word("x", 1_000, 2_000, inferred=True))
            )
        )
        bad = complete_timeline(
            lines=replaced(
                LINES, 0, line(VERSE, 0, 1_000, 5_000, word("x", 900, 2_000, inferred=True))
            )
        )

        assert check(good) == []
        assert check(bad) == [("lines[0].words[0]", "word_outside_line")]


# -- one named test per code (S3-4) --------------------------------------------------------------


class TestEveryCode:
    @pytest.mark.parametrize(
        ("document", "path"),
        [
            pytest.param(
                complete_timeline(sections=replaced(SECTIONS, 2, span(SOLO, 40_000, 60_001))),
                "sections[2]",
                id="section-past-the-end",
            ),
            pytest.param(
                complete_timeline(sections=replaced(SECTIONS, 2, span(SOLO, 60_000, 40_000))),
                "sections[2]",
                id="section-inverted",
            ),
            pytest.param(
                complete_timeline(lines=replaced(LINES, 1, line(VERSE, 1, 5_000, 5_000))),
                "lines[1]",
                id="line-empty",
            ),
            pytest.param(
                complete_timeline(
                    lines=replaced(LINES, 0, line(VERSE, 0, 1_000, 5_000, word("x", 2_000, 2_000)))
                ),
                "lines[0].words[0]",
                id="word-empty",
            ),
        ],
    )
    def test_span_out_of_range(self, document: TimelineDocument, path: str) -> None:
        assert check(document) == [(path, "span_out_of_range")]

    def test_span_out_of_range_is_measured_against_this_audio(self) -> None:
        document = TimelineDocument(sections=(span(VERSE, 0, 20_000),))

        assert check(document, duration_ms=20_000) == []
        assert check(document, duration_ms=19_999) == [("sections[0]", "span_out_of_range")]

    def test_unknown_section(self) -> None:
        document = TimelineDocument(
            sections=replaced(SECTIONS, 2, span(STRANGER, 40_000, 60_000)), lines=LINES
        )

        assert check(document) == [("sections[2]", "unknown_section")]

    def test_duplicate_section(self) -> None:
        document = complete_timeline(sections=(*SECTIONS, span(VERSE, 0, 20_000)))

        assert check(document) == [("sections[3]", "duplicate_section")]

    def test_section_order_when_listed_out_of_plan_order(self) -> None:
        document = complete_timeline(sections=(SECTIONS[1], SECTIONS[0], SECTIONS[2]))

        assert check(document) == [("sections[1]", "section_order")]

    def test_section_order_when_heard_out_of_plan_order(self) -> None:
        document = TimelineDocument(sections=(span(VERSE, 20_000, 40_000), span(CHORUS, 0, 20_000)))

        assert check(document) == [("sections[1]", "section_order")]

    def test_section_overlap(self) -> None:
        document = TimelineDocument(sections=(span(VERSE, 0, 20_000), span(CHORUS, 19_999, 40_000)))

        assert check(document) == [("sections[1]", "section_overlap")]

    def test_section_overlap_reports_every_span_inside_a_long_one(self) -> None:
        """Not only neighbours: the third span overlaps the first, not the second."""
        document = TimelineDocument(
            sections=(
                span(VERSE, 0, 50_000),
                span(CHORUS, 10_000, 20_000),
                span(SOLO, 30_000, 40_000),
            )
        )

        assert check(document) == [
            ("sections[1]", "section_overlap"),
            ("sections[2]", "section_overlap"),
        ]

    def test_unknown_line(self) -> None:
        document = complete_timeline(lines=(*LINES, line(CHORUS, 2, 31_000, 32_000)))

        assert check(document) == [("lines[4]", "unknown_line")]

    def test_unknown_line_for_a_section_the_lyrics_leave_empty(self) -> None:
        document = complete_timeline(lines=(*LINES, line(SOLO, 0, 41_000, 42_000)))

        assert check(document) == [("lines[4]", "unknown_line")]

    def test_every_line_is_unknown_when_the_audio_has_no_lyrics(self) -> None:
        assert check(complete_timeline(), with_lyrics=False) == [
            (f"lines[{index}]", "unknown_line") for index in range(len(LINES))
        ]

    def test_duplicate_line(self) -> None:
        document = complete_timeline(lines=(*LINES, LINES[0]))

        assert check(document) == [("lines[4]", "duplicate_line")]

    def test_line_section_untimed(self) -> None:
        document = TimelineDocument(
            sections=(span(VERSE, 0, 20_000),), lines=(line(CHORUS, 0, 21_000, 25_000),)
        )

        assert check(document) == [("lines[0]", "line_section_untimed")]

    def test_line_outside_section(self) -> None:
        document = complete_timeline(lines=replaced(LINES, 1, line(VERSE, 1, 5_000, 20_001)))

        assert check(document) == [("lines[1]", "line_outside_section")]

    def test_a_line_may_fill_its_whole_section(self) -> None:
        document = TimelineDocument(
            sections=(span(VERSE, 0, 20_000),), lines=(line(VERSE, 0, 0, 20_000),)
        )

        assert check(document) == []

    def test_line_order(self) -> None:
        document = complete_timeline(lines=(LINES[1], LINES[0], LINES[2], LINES[3]))

        assert check(document) == [("lines[1]", "line_order")]

    def test_line_order_follows_the_lyrics_document_s_section_order(self) -> None:
        """ "Lyrics order" is the lyrics document's own order, then ``line_index``."""
        reversed_lyrics = make_lyrics((CHORUS, ["hook"]), (VERSE, ["first"]))
        document = TimelineDocument(
            sections=(span(VERSE, 0, 20_000), span(CHORUS, 20_000, 40_000)),
            lines=(line(VERSE, 0, 1_000, 2_000), line(CHORUS, 0, 21_000, 22_000)),
        )

        issues = validate_timeline(
            document, duration_ms=DURATION_MS, plan=plan(), lyrics=reversed_lyrics
        )

        assert pairs(issues) == [("lines[1]", "line_order")]

    def test_line_overlap(self) -> None:
        document = complete_timeline(lines=replaced(LINES, 1, line(VERSE, 1, 4_999, 9_000)))

        assert check(document) == [("lines[1]", "line_overlap")]

    def test_word_outside_line(self) -> None:
        first = line(
            VERSE, 0, 1_000, 5_000, word("first", 1_000, 2_000), word("line", 2_000, 5_001)
        )

        assert check(complete_timeline(lines=replaced(LINES, 0, first))) == [
            ("lines[0].words[1]", "word_outside_line")
        ]

    def test_word_order(self) -> None:
        first = line(
            VERSE, 0, 1_000, 5_000, word("line", 2_000, 5_000), word("first", 1_000, 2_000)
        )

        assert check(complete_timeline(lines=replaced(LINES, 0, first))) == [
            ("lines[0].words[1]", "word_order")
        ]

    def test_word_overlap(self) -> None:
        first = line(
            VERSE, 0, 1_000, 5_000, word("first", 1_000, 2_001), word("line", 2_000, 5_000)
        )

        assert check(complete_timeline(lines=replaced(LINES, 0, first))) == [
            ("lines[0].words[1]", "word_overlap")
        ]


class TestAllIssuesAtOnce:
    def test_every_issue_is_collected_in_one_pass(self) -> None:
        document = TimelineDocument(
            sections=(
                span(CHORUS, 20_000, 40_000),
                span(VERSE, 0, 20_000),
                span(STRANGER, 50_000, 70_000),
            ),
            lines=(
                line(VERSE, 0, 1_000, 5_000, word("x", 500, 900)),
                line(VERSE, 7, 6_000, 7_000),
                line(CHORUS, 0, 39_000, 41_000),
            ),
        )

        # a multiset: which issues there are matters, not the order they are listed in
        assert sorted(check(document)) == [
            ("lines[0].words[0]", "word_outside_line"),
            ("lines[1]", "unknown_line"),
            ("lines[2]", "line_outside_section"),
            ("sections[1]", "section_order"),
            ("sections[2]", "span_out_of_range"),
            ("sections[2]", "unknown_section"),
        ]

    def test_paths_never_carry_text(self) -> None:
        secret = "SECRET-WORD"
        document = TimelineDocument(
            sections=(), lines=(line(VERSE, 0, 1, 2, word(secret, 0, 3), word(secret, 0, 3)),)
        )

        for issue in validate_timeline(
            document, duration_ms=DURATION_MS, plan=plan(), lyrics=lyrics()
        ):
            assert secret not in issue.path
            assert secret not in issue.code


# -- completeness --------------------------------------------------------------------------------


class TestIsComplete:
    def test_sections_only_when_the_audio_has_no_lyrics(self) -> None:
        full = TimelineDocument(sections=SECTIONS)
        short = TimelineDocument(sections=SECTIONS[:2])

        assert is_complete(full, plan=plan(), lyrics=None) is True
        assert is_complete(short, plan=plan(), lyrics=None) is False

    def test_with_lyrics_every_line_needs_a_span_too(self) -> None:
        assert is_complete(complete_timeline(), plan=plan(), lyrics=lyrics()) is True
        assert (
            is_complete(complete_timeline(lines=LINES[:3]), plan=plan(), lyrics=lyrics()) is False
        )
        assert (
            is_complete(TimelineDocument(sections=SECTIONS), plan=plan(), lyrics=lyrics()) is False
        )

    def test_with_lyrics_every_plan_section_still_needs_a_span(self) -> None:
        without_solo = complete_timeline(sections=SECTIONS[:2])

        assert is_complete(without_solo, plan=plan(), lyrics=lyrics()) is False

    def test_an_instrumental_section_needs_only_its_span(self) -> None:
        """The solo has no lyric lines in valid lyrics, so its span is all it needs."""
        document = complete_timeline()

        assert {entry.section_id for entry in lyrics().sections} == {VERSE, CHORUS}
        assert is_complete(document, plan=plan(), lyrics=lyrics()) is True

    def test_an_instrumental_plan_is_complete_with_its_sections(self) -> None:
        instrumental = make_plan(
            make_section(K.INTRO, section_id=VERSE),
            make_section(K.DROP, section_id=CHORUS),
            instrumental=True,
        )
        document = TimelineDocument(sections=SECTIONS[:2])

        assert is_complete(document, plan=instrumental, lyrics=None) is True


# -- the hash --------------------------------------------------------------------------------


class TestHash:
    def test_it_is_stable_across_runs(self) -> None:
        first = timeline_hash(complete_timeline())
        second = timeline_hash(
            msgspec.json.decode(msgspec.json.encode(complete_timeline()), type=TimelineDocument)
        )

        assert first == second

    def test_it_is_sha256_of_the_tagged_canonical_json(self) -> None:
        document = TimelineDocument(sections=(span(VERSE, 0, 5),))
        canonical = (
            b'timeline\n{"sections":[{"section_id":"00000000-0000-7000-8000-00000000000a",'
            b'"start_ms":0,"end_ms":5}]}'
        )

        assert timeline_hash(document) == hashlib.sha256(canonical).hexdigest()

    def test_different_content_hashes_differently(self) -> None:
        moved = complete_timeline(lines=replaced(LINES, 1, line(VERSE, 1, 5_000, 9_001)))

        assert timeline_hash(moved) != timeline_hash(complete_timeline())

    def test_an_explicit_inferred_false_hashes_like_an_omitted_one(self) -> None:
        """The ``omit_defaults`` guarantee (S3-14): defaults are never encoded, so a field added
        later with a default cannot change the hash of a stored row."""
        implicit = _wire_words({})
        explicit = _wire_words({"inferred": False})

        assert all(w.inferred is False for line_ in explicit.lines for w in line_.words)
        assert timeline_hash(explicit) == timeline_hash(implicit)
        assert b"inferred" not in msgspec.json.encode(explicit)

    def test_explicit_empty_words_and_lines_hash_like_omitted_ones(self) -> None:
        omitted = _decode({"sections": []})
        spelled = _decode({"sections": [], "lines": []})
        bare_line = {"section_id": str(VERSE), "line_index": 0, "start_ms": 0, "end_ms": 1}

        assert timeline_hash(spelled) == timeline_hash(omitted)
        assert timeline_hash(_decode({"sections": [], "lines": [bare_line]})) == timeline_hash(
            _decode({"sections": [], "lines": [{**bare_line, "words": []}]})
        )

    def test_an_inferred_word_changes_the_hash(self) -> None:
        measured = _wire_words({})
        guessed = msgspec.structs.replace(
            measured,
            lines=replaced(
                measured.lines,
                0,
                msgspec.structs.replace(
                    measured.lines[0],
                    words=replaced(
                        measured.lines[0].words,
                        0,
                        msgspec.structs.replace(measured.lines[0].words[0], inferred=True),
                    ),
                ),
            ),
        )

        assert timeline_hash(guessed) != timeline_hash(measured)

    def test_every_struct_omits_its_defaults(self) -> None:
        for struct in (SectionSpan, WordSpan, LineSpan, TimelineDocument):
            assert struct.__struct_config__.omit_defaults is True, struct.__name__
            assert struct.__struct_config__.forbid_unknown_fields is True, struct.__name__
            assert struct.__struct_config__.frozen is True, struct.__name__


def _decode(payload: dict[str, Any]) -> TimelineDocument:
    return msgspec.json.decode(msgspec.json.encode(payload), type=TimelineDocument)


def _wire_words(extra: dict[str, Any]) -> TimelineDocument:
    """:func:`complete_timeline` sent over the wire, with ``extra`` keys on every word."""
    payload = msgspec.json.decode(msgspec.json.encode(complete_timeline()))
    for entry in payload["lines"]:
        for item in entry.get("words", []):
            item.update(extra)
    return _decode(payload)


# -- the stored form -----------------------------------------------------------------------------


class TestStoredForm:
    def test_the_content_round_trips_through_jsonb(self) -> None:
        document = complete_timeline()

        content = timeline_content(document)

        assert decode_stored_timeline(TIMELINE_SCHEMA_VERSION, content) == document
        assert content["sections"][0] == {  # type: ignore[index]
            "section_id": str(VERSE),
            "start_ms": 0,
            "end_ms": 20_000,
        }

    def test_the_schema_version_is_one(self) -> None:
        assert TIMELINE_SCHEMA_VERSION == 1

    def test_an_unknown_schema_version_never_falls_back(self) -> None:
        with pytest.raises(StudioDocumentSchemaError, match="schema_version 2"):
            decode_stored_timeline(2, timeline_content(complete_timeline()))

    def test_word_text_is_stored_in_nfc_and_otherwise_as_sent(self) -> None:
        document = complete_timeline(
            lines=replaced(
                LINES, 0, line(VERSE, 0, 1_000, 5_000, word("  cafe\u0301  ", 1_000, 2_000))
            )
        )

        normalized = normalize_timeline(document)

        assert normalized.lines[0].words[0].text == "  caf\u00e9  "
        assert normalized.sections == document.sections
        assert normalized.lines[0].words[0].start_ms == 1_000


# -- limits (declared on the Structs only) -------------------------------------------------------


def _section_payload(count: int) -> list[dict[str, Any]]:
    return [{"section_id": str(VERSE), "start_ms": 0, "end_ms": 1} for _ in range(count)]


def _line_payload(**overrides: Any) -> dict[str, Any]:
    return {"section_id": str(VERSE), "line_index": 0, "start_ms": 0, "end_ms": 1} | overrides


def _word_payload(**overrides: Any) -> dict[str, Any]:
    return {"text": "a", "start_ms": 0, "end_ms": 1} | overrides


class TestLimits:
    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"sections": _section_payload(64)}, id="64-sections"),
            pytest.param({"sections": [], "lines": [_line_payload()] * 4096}, id="4096-lines"),
            pytest.param({"sections": [], "lines": [_line_payload(line_index=63)]}, id="line-63"),
            pytest.param(
                {"sections": [], "lines": [_line_payload(words=[_word_payload()] * 64)]},
                id="64-words",
            ),
            pytest.param(
                {"sections": [], "lines": [_line_payload(words=[_word_payload(text="a" * 100)])]},
                id="100-chars",
            ),
            pytest.param(
                {"sections": [{"section_id": str(VERSE), "start_ms": 0, "end_ms": AUDIO_MAX_MS}]},
                id="end-at-the-cap",
            ),
        ],
    )
    def test_the_limit_itself_is_accepted(self, payload: dict[str, Any]) -> None:
        _decode(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({"sections": _section_payload(65)}, id="65-sections"),
            pytest.param({"sections": [], "lines": [_line_payload()] * 4097}, id="4097-lines"),
            pytest.param({"sections": [], "lines": [_line_payload(line_index=64)]}, id="line-64"),
            pytest.param({"sections": [], "lines": [_line_payload(line_index=-1)]}, id="line--1"),
            pytest.param(
                {"sections": [], "lines": [_line_payload(words=[_word_payload()] * 65)]},
                id="65-words",
            ),
            pytest.param(
                {"sections": [], "lines": [_line_payload(words=[_word_payload(text="a" * 101)])]},
                id="101-chars",
            ),
            pytest.param(
                {"sections": [], "lines": [_line_payload(words=[_word_payload(text="")])]},
                id="empty-word",
            ),
            pytest.param(
                {"sections": [], "lines": [_line_payload(words=[_word_payload(text="a\x00b")])]},
                id="nul-in-word",
            ),
            pytest.param(
                {"sections": [], "lines": [_line_payload(words=[_word_payload(text="a\n\x00")])]},
                id="nul-after-a-newline",
            ),
            pytest.param(
                {"sections": [{"section_id": str(VERSE), "start_ms": -1, "end_ms": 1}]},
                id="negative-time",
            ),
            pytest.param(
                {
                    "sections": [
                        {"section_id": str(VERSE), "start_ms": 0, "end_ms": AUDIO_MAX_MS + 1}
                    ]
                },
                id="past-the-cap",
            ),
            pytest.param(
                {"sections": [{"section_id": str(VERSE), "start_ms": 0.5, "end_ms": 1}]},
                id="float-time",
            ),
            pytest.param({"sections": [], "complete": True}, id="unknown-field"),
            pytest.param({"lines": []}, id="sections-required"),
        ],
    )
    def test_one_past_the_limit_is_refused_on_the_wire(self, payload: dict[str, Any]) -> None:
        with pytest.raises(msgspec.ValidationError):
            _decode(payload)

    @pytest.mark.parametrize("text", ["a\nb", "a\n", "\t", "\u2028", "\U0001f3b5"])
    def test_any_other_character_is_a_legal_word(self, text: str) -> None:
        """Only NUL is refused: it is the one character PostgreSQL JSONB cannot store."""
        assert _decode(_one_word(text)).lines[0].words[0].text == text

    def test_the_time_cap_is_the_shared_audio_cap(self) -> None:
        assert (AUDIO_MAX_SECONDS, AUDIO_MAX_MS) == (900, 900_000)

    def test_a_word_nfc_expands_past_100_is_refused_after_normalization(self) -> None:
        """RD6: U+FB2C is one code point on the wire and three after NFC."""
        fits = normalize_timeline(_decode(_one_word("\ufb2c" * 33)))
        expands = normalize_timeline(_decode(_one_word("\ufb2c" * 34)))

        assert check_timeline_limits(fits) == []
        assert pairs(check_timeline_limits(expands)) == [
            ("lines[0].words[0].text", "too_long_after_normalization")
        ]

    def test_a_document_past_the_size_cap_is_refused(self) -> None:
        words = [_word_payload(text="a" * 100)] * 64
        document = _decode({"sections": [], "lines": [_line_payload(words=words)] * 20})

        assert len(msgspec.json.encode(document)) > MAX_DOCUMENT_BYTES
        assert pairs(check_timeline_limits(document)) == [("", "document_too_large")]

    def test_a_document_within_the_size_cap_is_accepted(self) -> None:
        assert check_timeline_limits(complete_timeline()) == []


def _one_word(text: str) -> dict[str, Any]:
    return {"sections": [], "lines": [_line_payload(words=[_word_payload(text=text)])]}
