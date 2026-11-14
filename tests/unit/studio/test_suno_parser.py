"""The Suno-text parser: golden cases for every rule of the lyrics algorithm and I8."""

from __future__ import annotations

import unicodedata

import pytest

from src.core.enums import StudioSectionKind as K
from src.studio.errors import DocumentIssue, ImportInvalid
from src.studio.importers.suno import ImportWarning, ParsedSuno, parse_suno
from tests.studio_support import GOLDEN_EXCLUDE, GOLDEN_LYRICS, GOLDEN_STYLE

pytestmark = pytest.mark.unit


def parse(lyrics: str = "", style: str = "", exclude: str = "") -> ParsedSuno:
    return parse_suno(style=style, exclude=exclude, lyrics=lyrics)


def codes(parsed: ParsedSuno) -> list[tuple[str, int | None]]:
    return [(w.code, w.line) for w in parsed.warnings]


class TestGoldenRendererOutput:
    """What the P1 renderer wrote for its golden song, read back."""

    def test_every_section_is_recovered_in_order(self) -> None:
        parsed = parse(GOLDEN_LYRICS, GOLDEN_STYLE, GOLDEN_EXCLUDE)
        assert [(s.header_name, s.kind) for s in parsed.sections] == [
            ("Intro", K.INTRO),
            ("Verse 1", K.VERSE),
            ("Pre-Chorus", K.PRE_CHORUS),
            ("Chorus", K.CHORUS),
            ("Verse 2", K.VERSE),
            ("Chorus", K.CHORUS),
            ("Drop 1", K.DROP),
            ("Drop 2", K.DROP),
            ("Break It Down", K.INTERLUDE),
            ("Outro", K.OUTRO),
        ]
        assert codes(parsed) == [("unknown_section_name", 25)]

    def test_descriptors_the_instrumental_cue_and_header_lines(self) -> None:
        sections = parse(GOLDEN_LYRICS).sections
        assert sections[0].descriptors == ("sparse pads",)
        assert sections[1].descriptors == ("breathy", "minimal")
        assert sections[1].header_line == "[Verse 1: breathy, minimal]"
        assert (sections[7].descriptors, sections[7].instrumental) == ((), True)
        assert sections[7].header_line == "[Drop 2: instrumental]"
        assert not any(s.instrumental for s in sections[:7])

    def test_lines_and_their_delivery_and_source_lines(self) -> None:
        verse = parse(GOLDEN_LYRICS).sections[1]
        assert [(ln.text, ln.delivery, ln.source_line) for ln in verse.lines] == [
            ("First line", (), 4),
            ("Second line", ("whispered",), 6),
        ]
        assert verse.source_line == 3

    def test_style_and_exclude_tokens_with_the_tempo_taken_out(self) -> None:
        parsed = parse(GOLDEN_LYRICS, GOLDEN_STYLE, GOLDEN_EXCLUDE)
        assert parsed.style_tokens == ("synthwave", "female vocals", "Dreamy", "Warm Reverb")
        assert parsed.tempo_bpm == 118
        assert parsed.exclude_tokens == ("heavy metal", "autotune", "screaming")


class TestHeaders:
    def test_hand_written_text_without_blank_lines(self) -> None:
        parsed = parse("[Verse 1]\nline one\nline two\n[Chorus]\nsing it\n[Verse 2]\nagain")
        assert [(s.header_name, [ln.text for ln in s.lines]) for s in parsed.sections] == [
            ("Verse 1", ["line one", "line two"]),
            ("Chorus", ["sing it"]),
            ("Verse 2", ["again"]),
        ]
        assert parsed.warnings == ()

    @pytest.mark.parametrize(
        ("name", "kind"),
        [
            ("Refrain", K.CHORUS),
            ("Fade Out", K.OUTRO),
            ("Break", K.BREAKDOWN),
            ("solo", K.INSTRUMENTAL),
        ],
    )
    def test_aliases(self, name: str, kind: K) -> None:
        section = parse(f"[{name}]\nx").sections[0]
        assert (section.kind, section.header_name) == (kind, name)

    def test_an_unknown_name_is_an_interlude_with_a_warning(self) -> None:
        parsed = parse("[Verse]\na\n\n[Guitar Duel]\nb")
        assert parsed.sections[1].kind is K.INTERLUDE
        assert parsed.sections[1].header_name == "Guitar Duel"
        assert codes(parsed) == [("unknown_section_name", 4)]

    def test_a_bracket_line_opening_a_block_is_a_header_even_if_unknown(self) -> None:
        parsed = parse("[whispered]\nhello")
        assert parsed.sections[0].header_name == "whispered"
        assert codes(parsed) == [("unknown_section_name", 1)]

    def test_a_known_header_in_the_middle_of_a_block_is_a_header(self) -> None:
        parsed = parse("[Verse]\na\n[Chorus: big]\nb")
        assert [s.header_name for s in parsed.sections] == ["Verse", "Chorus"]
        assert parsed.sections[1].descriptors == ("big",)

    def test_an_unknown_bracket_in_the_middle_of_a_block_is_a_delivery(self) -> None:
        parsed = parse("[Verse]\n[softly, close]\nhello")
        assert parsed.sections[0].lines[0].delivery == ("softly", "close")
        assert parsed.warnings == ()

    def test_split_at_the_first_colon_only(self) -> None:
        section = parse("[Bridge: key: C minor, slow]\nx").sections[0]
        assert (section.header_name, section.descriptors) == ("Bridge", ("key: C minor", "slow"))

    def test_the_instrumental_cue_is_removed_wherever_it_stands(self) -> None:
        for header in (
            "[Drop: instrumental, heavy]",
            "[Drop: heavy, Instrumental]",
            "[Drop: INSTRUMENTAL]",
        ):
            section = parse(header).sections[0]
            assert section.instrumental
            assert "instrumental" not in {d.casefold() for d in section.descriptors}

    def test_an_instrumental_kind_does_not_set_the_flag(self) -> None:
        section = parse("[Instrumental]").sections[0]
        assert section.kind is K.INSTRUMENTAL
        assert not section.instrumental

    def test_duplicate_descriptors_are_deduped_with_a_warning(self) -> None:
        parsed = parse("[Verse: soft, Soft, warm]")
        assert parsed.sections[0].descriptors == ("soft", "warm")
        assert codes(parsed) == [("duplicate_descriptor", 1)]

    def test_an_empty_name_makes_the_whole_text_the_name(self) -> None:
        section = parse("[: odd]").sections[0]
        assert (section.header_name, section.descriptors) == (": odd", ())

    def test_header_padding_is_canonicalised(self) -> None:
        section = parse("[  Verse   1 :  soft  ]").sections[0]
        assert section.header_line == "[Verse 1 : soft]"
        assert (section.header_name, section.descriptors) == ("Verse 1", ("soft",))


class TestBlocksAndImplicitSections:
    def test_a_blank_line_then_text_continues_the_section(self) -> None:
        parsed = parse("[Verse]\na\n\nb\n\n[Chorus]\nc")
        assert [ln.text for ln in parsed.sections[0].lines] == ["a", "b"]
        assert parsed.warnings == ()

    def test_no_headers_at_all_is_one_implicit_verse(self) -> None:
        parsed = parse("first\nsecond")
        (section,) = parsed.sections
        assert (section.kind, section.header_line, section.header_name) == (K.VERSE, None, "")
        assert [ln.text for ln in section.lines] == ["first", "second"]
        assert codes(parsed) == [("no_section_headers", 1)]

    def test_text_before_the_first_header_is_an_implicit_verse(self) -> None:
        parsed = parse("stray\n\n[Chorus]\nsing")
        assert [(s.kind, s.header_line) for s in parsed.sections] == [
            (K.VERSE, None),
            (K.CHORUS, "[Chorus]"),
        ]
        assert codes(parsed) == [("text_before_first_header", 1)]

    def test_exactly_one_of_the_two_implicit_warnings(self) -> None:
        assert [c for c, _ in codes(parse("a\nb\nc"))] == ["no_section_headers"]
        assert [c for c, _ in codes(parse("a\n\n[Verse]\nb"))] == ["text_before_first_header"]

    def test_headers_without_lines_are_sections_without_lines(self) -> None:
        parsed = parse("[Intro]\n\n[Verse]\n\n[Outro]")
        assert [(s.header_name, s.lines) for s in parsed.sections] == [
            ("Intro", ()),
            ("Verse", ()),
            ("Outro", ()),
        ]
        assert parsed.warnings == ()

    def test_an_empty_lyrics_box_is_zero_sections(self) -> None:
        for lyrics in ("", "\n\n", "   \n \r\n"):
            parsed = parse(lyrics)
            assert parsed.sections == ()
            assert parsed.warnings == ()

    def test_line_numbers_count_blank_lines(self) -> None:
        parsed = parse("\n\n[Verse]\n\nhello")
        assert parsed.sections[0].source_line == 3
        assert parsed.sections[0].lines[0].source_line == 5


class TestDelivery:
    def test_a_delivery_line_applies_to_the_next_text_line_only(self) -> None:
        lines = parse("[Verse]\n[whispered]\na\nb").sections[0].lines
        assert [ln.delivery for ln in lines] == [("whispered",), ()]

    def test_a_dangling_delivery_at_the_end_of_a_block_is_dropped_with_a_warning(self) -> None:
        parsed = parse("[Verse]\na\n[whispered]\n\n[Chorus]\nb")
        assert [ln.text for ln in parsed.sections[0].lines] == ["a"]
        assert codes(parsed) == [("dangling_delivery", 3)]

    def test_a_dangling_delivery_at_the_end_of_the_input(self) -> None:
        parsed = parse("[Verse]\na\n[whispered]")
        assert codes(parsed) == [("dangling_delivery", 3)]

    def test_a_dangling_delivery_before_a_header_in_the_same_block(self) -> None:
        parsed = parse("[Verse]\na\n[whispered]\n[Chorus]\nb")
        assert codes(parsed) == [("dangling_delivery", 3)]
        assert [s.header_name for s in parsed.sections] == ["Verse", "Chorus"]

    def test_two_delivery_lines_in_a_row_drop_the_first(self) -> None:
        parsed = parse("[Verse]\n[soft]\n[loud]\nhello")
        assert parsed.sections[0].lines[0].delivery == ("loud",)
        assert codes(parsed) == [("dangling_delivery", 2)]

    def test_delivery_duplicates_are_collapsed_with_a_warning(self) -> None:
        parsed = parse("[Verse]\n[soft, Soft]\nhello")
        assert parsed.sections[0].lines[0].delivery == ("soft",)
        assert codes(parsed) == [("duplicate_descriptor", 2)]

    def test_a_delivery_with_nothing_in_it_is_dangling(self) -> None:
        parsed = parse("[Verse]\n[ , ]\nhello")
        assert parsed.sections[0].lines[0].delivery == ()
        assert codes(parsed) == [("dangling_delivery", 2)]


class TestTextHandling:
    def test_crlf_and_cr_line_endings(self) -> None:
        crlf = parse("[Verse]\r\nhello\r\n\r\n[Chorus]\r\nsing")
        cr = parse("[Verse]\rhello\r\r[Chorus]\rsing")
        lf = parse("[Verse]\nhello\n\n[Chorus]\nsing")
        assert crlf == cr == lf

    def test_nfd_input_is_read_as_nfc(self) -> None:
        nfd = unicodedata.normalize("NFD", "Café au lait")
        assert nfd != "Café au lait"
        parsed = parse(f"[Verse]\n{nfd}", style=nfd)
        assert parsed.sections[0].lines[0].text == "Café au lait"
        assert parsed.style_tokens == ("Café au lait",)

    def test_parentheses_are_kept_verbatim(self) -> None:
        parsed = parse("[Chorus]\nla la (la la)\n(ooh)")
        assert [ln.text for ln in parsed.sections[0].lines] == ["la la (la la)", "(ooh)"]

    def test_cyrillic(self) -> None:
        parsed = parse("[Куплет]\nПривет, мир", style="русский рок")  # noqa: RUF001
        assert parsed.sections[0].lines[0].text == "Привет, мир"
        assert parsed.style_tokens == ("русский рок",)

    def test_whitespace_inside_a_line_is_collapsed(self) -> None:
        assert parse("[Verse]\n  a    b  ").sections[0].lines[0].text == "a b"

    def test_deterministic(self) -> None:
        text = "[Verse]\n[soft]\na\n\n[Chorus: x]\nb"
        assert parse(text, "a, b, 120 bpm", "c") == parse(text, "a, b, 120 bpm", "c")


class TestInlineTags:
    def test_a_bracket_inside_a_lyric_line_is_refused(self) -> None:
        with pytest.raises(ImportInvalid) as exc:
            parse("[Verse]\n[x] text")
        assert exc.value.issues == (DocumentIssue("lyrics[2]", "unsupported_inline_tag"),)
        assert exc.value.source_lines == {"lyrics[2]": 2}

    @pytest.mark.parametrize("line", ["text [x]", "a [x] b", "oops ]", "[", "]", "[ ]", "[]"])
    def test_every_stray_bracket_is_refused(self, line: str) -> None:
        with pytest.raises(ImportInvalid):
            parse(f"[Verse]\n{line}")

    def test_every_offending_line_is_reported(self) -> None:
        with pytest.raises(ImportInvalid) as exc:
            parse("a [x]\nok\nb [y]")
        assert [i.path for i in exc.value.issues] == ["lyrics[1]", "lyrics[3]"]
        assert exc.value.detail == {
            "issues": [
                {"path": "lyrics[1]", "code": "unsupported_inline_tag"},
                {"path": "lyrics[3]", "code": "unsupported_inline_tag"},
            ],
            "source_lines": {"lyrics[1]": 1, "lyrics[3]": 3},
        }


class TestStyleAndExclude:
    @pytest.mark.parametrize(
        "text", ["lofi, 120 BPM", "lofi, 120bpm", "lofi, 120 bpm", "lofi,120  Bpm"]
    )
    def test_bpm_forms(self, text: str) -> None:
        parsed = parse(style=text)
        assert (parsed.tempo_bpm, parsed.style_tokens) == (120, ("lofi",))

    @pytest.mark.parametrize("token", ["300 BPM", "39 bpm", "1000 bpm", "fast bpm", "5 BPM"])
    def test_a_tempo_out_of_range_or_malformed_stays_a_tag(self, token: str) -> None:
        parsed = parse(style=f"lofi, {token}")
        assert parsed.tempo_bpm is None
        assert parsed.style_tokens == ("lofi", token)

    def test_the_bounds_are_inclusive(self) -> None:
        assert parse(style="40 bpm").tempo_bpm == 40
        assert parse(style="240 bpm").tempo_bpm == 240

    def test_only_the_first_tempo_counts_and_later_ones_are_dropped(self) -> None:
        """I8 amended: a later tempo is dropped (warned), never kept as a tag the plan cannot hold."""
        parsed = parse(style="100 bpm, lofi, 120 BPM")
        assert parsed.tempo_bpm == 100
        assert parsed.style_tokens == ("lofi",)
        assert codes(parsed) == [("duplicate_tempo", None)]

    def test_every_later_tempo_is_dropped_with_its_own_warning(self) -> None:
        parsed = parse(style="90 BPM, 90 bpm, a, 120bpm")
        assert (parsed.tempo_bpm, parsed.style_tokens) == (90, ("a",))
        assert codes(parsed) == [("duplicate_tempo", None)] * 2

    def test_the_tempo_may_stand_anywhere_in_the_text(self) -> None:
        """``120 BPM, rock`` is a typical hand-written Style: the tempo is read, rock stays."""
        parsed = parse(style="120 BPM, rock")
        assert (parsed.tempo_bpm, parsed.style_tokens) == (120, ("rock",))

    def test_a_tempo_out_of_range_does_not_block_a_later_valid_one(self) -> None:
        parsed = parse(style="300 bpm, 90 bpm")
        assert (parsed.tempo_bpm, parsed.style_tokens) == (90, ("300 bpm",))

    def test_empty_items_are_skipped_and_items_trimmed(self) -> None:
        parsed = parse(style=" a ,, b ,  ,", exclude=",x,")
        assert parsed.style_tokens == ("a", "b")
        assert parsed.exclude_tokens == ("x",)
        assert parsed.warnings == ()

    def test_duplicates_are_deduped_case_insensitively_first_wins(self) -> None:
        parsed = parse(style="Dreamy, synth, dreamy, DREAMY", exclude="a, A")
        assert parsed.style_tokens == ("Dreamy", "synth")
        assert parsed.exclude_tokens == ("a",)
        assert codes(parsed) == [("duplicate_style_item", None)] * 3

    def test_a_bpm_token_in_exclude_is_just_a_token(self) -> None:
        parsed = parse(exclude="120 bpm")
        assert (parsed.tempo_bpm, parsed.exclude_tokens) == (None, ("120 bpm",))

    def test_empty_boxes(self) -> None:
        parsed = parse()
        assert parsed == ParsedSuno((), None, (), (), ())

    def test_warnings_are_comparable_values(self) -> None:
        assert ImportWarning("x", 3) == ImportWarning("x", 3)
        assert ImportWarning("x") != ImportWarning("x", 3)
