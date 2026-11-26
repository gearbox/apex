"""The LRC and SRT exports (S3a W5): byte-exact output, warnings, and what a renderer may see."""

from __future__ import annotations

import dataclasses
from typing import get_type_hints
from uuid import UUID

import pytest

from src.core.enums import StudioSectionKind as K
from src.core.enums import StudioTimedLyricsFormat
from src.studio.renderers.timed_lyrics import (
    TIMED_LYRICS_RENDERERS,
    LrcRenderer,
    RenderedTimedLyrics,
    SrtRenderer,
    TimedLine,
    TimedLyricsProjection,
    TimedLyricsWarning,
    TimedWord,
    project_timed_lyrics,
)
from src.studio.timeline import LineSpan, SectionSpan, TimelineDocument, WordSpan
from tests.studio_support import make_lyrics, make_plan, make_section

pytestmark = pytest.mark.unit

VERSE = UUID("00000000-0000-7000-8000-00000000000a")
CHORUS = UUID("00000000-0000-7000-8000-00000000000b")


def _line(
    section_id: UUID, line_index: int, start_ms: int, end_ms: int, *words: WordSpan
) -> LineSpan:
    return LineSpan(
        section_id=section_id,
        line_index=line_index,
        start_ms=start_ms,
        end_ms=end_ms,
        words=words,
    )


def _word(text: str, start_ms: int, end_ms: int, *, inferred: bool = False) -> WordSpan:
    return WordSpan(text=text, start_ms=start_ms, end_ms=end_ms, inferred=inferred)


def _timeline(*lines: LineSpan) -> TimelineDocument:
    return TimelineDocument(
        sections=(
            SectionSpan(section_id=VERSE, start_ms=0, end_ms=20_000),
            SectionSpan(section_id=CHORUS, start_ms=20_000, end_ms=40_000),
        ),
        lines=lines,
    )


LYRICS = make_lyrics(
    (VERSE, ["Night falls on the harbour", "Lanterns on the water"]),
    (CHORUS, ["Carry me home"]),
)

PLAIN = _timeline(
    _line(VERSE, 0, 1_000, 5_500),
    _line(VERSE, 1, 5_509, 9_999),
    _line(CHORUS, 0, 21_250, 25_000),
)

ENHANCED = _timeline(
    _line(
        VERSE,
        0,
        1_000,
        5_500,
        _word("Night", 1_000, 1_800),
        _word("falls", 1_800, 2_400),
        _word("on", 2_400, 2_600),
        _word("the", 2_600, 2_900),
        _word("harbour", 2_900, 5_500),
    ),
    _line(VERSE, 1, 5_509, 9_999),
    _line(CHORUS, 0, 21_250, 25_000, _word("Carry", 21_250, 22_000, inferred=True)),
)


def _render(
    export_format: StudioTimedLyricsFormat, timeline: TimelineDocument
) -> RenderedTimedLyrics:
    return TIMED_LYRICS_RENDERERS[export_format].render(project_timed_lyrics(timeline, LYRICS))


class TestByteExact:
    def test_plain_lrc(self) -> None:
        rendered = _render(StudioTimedLyricsFormat.LRC, PLAIN)

        assert rendered.text.encode() == (
            b"[00:01.00]Night falls on the harbour\n"
            b"[00:05.50]Lanterns on the water\n"
            b"[00:21.25]Carry me home\n"
        )
        assert rendered.warnings == ()

    def test_enhanced_lrc_writes_a_stamp_per_word_where_a_line_has_words(self) -> None:
        rendered = _render(StudioTimedLyricsFormat.LRC, ENHANCED)

        assert rendered.text.encode() == (
            b"[00:01.00]<00:01.00>Night <00:01.80>falls <00:02.40>on <00:02.60>the "
            b"<00:02.90>harbour\n"
            b"[00:05.50]Lanterns on the water\n"
            b"[00:21.25]<00:21.25>Carry\n"
        )

    def test_srt(self) -> None:
        rendered = _render(StudioTimedLyricsFormat.SRT, ENHANCED)

        assert rendered.text.encode() == (
            b"1\n00:00:01,000 --> 00:00:05,500\nNight falls on the harbour\n\n"
            b"2\n00:00:05,509 --> 00:00:09,999\nLanterns on the water\n\n"
            b"3\n00:00:21,250 --> 00:00:25,000\nCarry me home\n\n"
        )
        assert rendered.warnings == ()  # SRT carries no words, so nothing inferred is in it

    def test_there_are_no_header_tags(self) -> None:
        text = _render(StudioTimedLyricsFormat.LRC, PLAIN).text

        assert not [ln for ln in text.splitlines() if not ln.startswith("[0")]
        for tag in ("[ti:", "[ar:", "[al:", "[length:", "[by:", "[offset:"):
            assert tag not in text

    @pytest.mark.parametrize("export_format", list(StudioTimedLyricsFormat))
    def test_the_same_input_renders_the_same_bytes(
        self, export_format: StudioTimedLyricsFormat
    ) -> None:
        first = _render(export_format, ENHANCED)
        second = _render(export_format, ENHANCED)

        assert first == second

    def test_every_format_has_exactly_one_renderer(self) -> None:
        assert set(TIMED_LYRICS_RENDERERS) == set(StudioTimedLyricsFormat)
        for export_format, renderer in TIMED_LYRICS_RENDERERS.items():
            assert renderer.format is export_format


@pytest.mark.parametrize(
    ("ms", "lrc", "srt"),
    [
        (0, "00:00.00", "00:00:00,000"),
        (9, "00:00.00", "00:00:00,009"),
        (1_019, "00:01.01", "00:00:01,019"),  # centiseconds truncate (ms // 10), never round
        (59_999, "00:59.99", "00:00:59,999"),
        (60_000, "01:00.00", "00:01:00,000"),
        (754_321, "12:34.32", "00:12:34,321"),  # past ten minutes
        (900_000, "15:00.00", "00:15:00,000"),  # the longest audio
    ],
)
def test_timestamp_formatting_at_the_edges(ms: int, lrc: str, srt: str) -> None:
    projection = TimedLyricsProjection(
        lines=(TimedLine(start_ms=ms, end_ms=ms, text="x", words=()),), untimed=0
    )

    assert LrcRenderer().render(projection).text == f"[{lrc}]x\n"
    assert SrtRenderer().render(projection).text == f"1\n{srt} --> {srt}\nx\n\n"


class TestWarnings:
    @pytest.mark.parametrize("export_format", list(StudioTimedLyricsFormat))
    def test_untimed_lines_are_left_out_and_counted_never_guessed(
        self, export_format: StudioTimedLyricsFormat
    ) -> None:
        rendered = _render(export_format, _timeline(_line(CHORUS, 0, 21_000, 22_000)))

        assert rendered.warnings == (TimedLyricsWarning("untimed_lines", 2),)
        assert "Carry me home" in rendered.text
        assert "Night falls" not in rendered.text
        assert "Lanterns" not in rendered.text

    @pytest.mark.parametrize("export_format", list(StudioTimedLyricsFormat))
    def test_a_timeline_that_times_no_line_exports_the_empty_string(
        self, export_format: StudioTimedLyricsFormat
    ) -> None:
        rendered = _render(export_format, _timeline())

        assert rendered.text == ""
        assert rendered.warnings == (TimedLyricsWarning("untimed_lines", 3),)

    @pytest.mark.parametrize("export_format", list(StudioTimedLyricsFormat))
    def test_no_lyrics_is_an_empty_export_and_only_that_warning(
        self, export_format: StudioTimedLyricsFormat
    ) -> None:
        rendered = TIMED_LYRICS_RENDERERS[export_format].render(None)

        assert rendered.text == ""
        assert rendered.warnings == (TimedLyricsWarning("no_lyrics"),)

    def test_inferred_word_timings_are_counted_and_the_words_still_exported(self) -> None:
        timeline = _timeline(
            _line(
                VERSE,
                0,
                1_000,
                5_000,
                _word("Night", 1_000, 2_000, inferred=True),
                _word("falls", 2_000, 3_000),
                _word("on", 3_000, 4_000, inferred=True),
            ),
            _line(VERSE, 1, 5_000, 6_000),
            _line(CHORUS, 0, 21_000, 25_000, _word("Carry", 21_000, 22_000, inferred=True)),
        )

        rendered = _render(StudioTimedLyricsFormat.LRC, timeline)

        assert rendered.warnings == (TimedLyricsWarning("inferred_word_timings", 3),)
        assert rendered.text == (
            "[00:01.00]<00:01.00>Night <00:02.00>falls <00:03.00>on\n"
            "[00:05.00]Lanterns on the water\n"
            "[00:21.00]<00:21.00>Carry\n"
        )

    def test_both_warnings_can_be_reported_together(self) -> None:
        rendered = _render(
            StudioTimedLyricsFormat.LRC,
            _timeline(_line(VERSE, 0, 1_000, 2_000, _word("x", 1_000, 2_000, inferred=True))),
        )

        assert rendered.warnings == (
            TimedLyricsWarning("untimed_lines", 2),
            TimedLyricsWarning("inferred_word_timings", 1),
        )


class TestText:
    def test_line_text_comes_from_the_lyrics_and_times_from_the_timeline(self) -> None:
        other = make_lyrics((VERSE, ["Other words", "More"]), (CHORUS, ["Else"]))

        rendered = LrcRenderer().render(project_timed_lyrics(PLAIN, other))

        assert rendered.text == "[00:01.00]Other words\n[00:05.50]More\n[00:21.25]Else\n"

    def test_lines_come_out_in_time_order_whatever_order_the_timeline_lists(self) -> None:
        shuffled = _timeline(*reversed(PLAIN.lines))

        assert project_timed_lyrics(shuffled, LYRICS) == project_timed_lyrics(PLAIN, LYRICS)

    def test_a_word_cannot_break_out_of_its_line(self) -> None:
        """Word text is stored as sent, so the line-based format has to hold it on one line."""
        forged = _timeline(
            _line(VERSE, 0, 1_000, 2_000, _word("a\n[00:00.00]forged\r\u2028b", 1_000, 2_000))
        )

        text = _render(StudioTimedLyricsFormat.LRC, forged).text

        assert text == "[00:01.00]<00:01.00>a [00:00.00]forged b\n"
        assert len(text.splitlines()) == 1


class TestLyricsInAnotherSectionOrder:
    """S3-4a: exports are chronological, even when the lyrics document lists the chorus (sung
    second) ahead of the verse. Cue numbers follow time; each cue keeps its own line's text."""

    REORDERED = make_lyrics(
        (CHORUS, ["Carry me home"]),
        (VERSE, ["Night falls on the harbour", "Lanterns on the water"]),
    )

    def test_lrc_lines_are_chronological(self) -> None:
        rendered = LrcRenderer().render(project_timed_lyrics(PLAIN, self.REORDERED))

        assert rendered.text == (
            "[00:01.00]Night falls on the harbour\n"
            "[00:05.50]Lanterns on the water\n"
            "[00:21.25]Carry me home\n"
        )
        assert rendered.warnings == ()

    def test_srt_numbers_follow_time_and_each_cue_keeps_its_own_line_s_text(self) -> None:
        texts = {
            (section.section_id, line_index): lyric_line.text
            for section in self.REORDERED.sections
            for line_index, lyric_line in enumerate(section.lines)
        }
        # PLAIN lists its spans in time order; the text is looked up by identity, not position.
        expected = [
            (str(number), texts[(span.section_id, span.line_index)], span.start_ms)
            for number, span in enumerate(PLAIN.lines, start=1)
        ]

        rendered = SrtRenderer().render(project_timed_lyrics(PLAIN, self.REORDERED))

        cues = [cue.split("\n") for cue in rendered.text.removesuffix("\n\n").split("\n\n")]
        assert [(number, text) for number, _times, text in cues] == [
            (number, text) for number, text, _start in expected
        ]
        assert [times.split(" --> ")[0] for _number, times, _text in cues] == [
            "00:00:01,000",
            "00:00:05,509",
            "00:00:21,250",
        ]
        assert [start for _number, _text, start in expected] == [1_000, 5_509, 21_250]

    @pytest.mark.parametrize("export_format", list(StudioTimedLyricsFormat))
    def test_untimed_lines_are_still_counted(self, export_format: StudioTimedLyricsFormat) -> None:
        timeline = _timeline(_line(VERSE, 1, 5_000, 6_000), _line(CHORUS, 0, 21_000, 22_000))

        rendered = TIMED_LYRICS_RENDERERS[export_format].render(
            project_timed_lyrics(timeline, self.REORDERED)
        )

        assert rendered.warnings == (TimedLyricsWarning("untimed_lines", 1),)
        assert rendered.text.index("Lanterns") < rendered.text.index("Carry me home")


class TestTheProjection:
    """No plan notes, idea, references or labels can reach a renderer: there is no field to
    carry them."""

    def test_it_has_exactly_these_fields_of_these_types(self) -> None:
        def shape(cls: type) -> dict[str, object]:
            hints = get_type_hints(cls)
            return {field.name: hints[field.name] for field in dataclasses.fields(cls)}

        assert shape(TimedLyricsProjection) == {
            "lines": tuple[TimedLine, ...],
            "untimed": int,
        }
        assert shape(TimedLine) == {
            "start_ms": int,
            "end_ms": int,
            "text": str,
            "words": tuple[TimedWord, ...],
        }
        assert shape(TimedWord) == {"start_ms": int, "text": str, "inferred": bool}

    @pytest.mark.parametrize("export_format", list(StudioTimedLyricsFormat))
    def test_nothing_of_the_plan_appears_in_an_export(
        self, export_format: StudioTimedLyricsFormat
    ) -> None:
        plan = make_plan(
            make_section(K.VERSE, section_id=VERSE, label="SECRET-LABEL", notes="SECRET-NOTES"),
            make_section(K.CHORUS, section_id=CHORUS, tags=("SECRET-TAG",)),
            title="SECRET-TITLE",
            idea="SECRET-IDEA",
            references=("SECRET-REFERENCE",),
        )
        assert {section.id for section in plan.sections} == {VERSE, CHORUS}  # what is timed

        text = _render(export_format, ENHANCED).text

        assert "SECRET" not in text

    def test_the_renderers_accept_only_the_projection(self) -> None:
        for renderer in (LrcRenderer, SrtRenderer):
            hints = get_type_hints(
                renderer.render,
                localns={
                    "TimedLyricsProjection": TimedLyricsProjection,
                    "RenderedTimedLyrics": RenderedTimedLyrics,
                },
            )
            assert hints["projection"] == TimedLyricsProjection | None, renderer.__name__
