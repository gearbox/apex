"""One case per issue code, multi-issue collection, normalization output."""

from __future__ import annotations

import unicodedata
from uuid import UUID

import msgspec
import pytest

from src.core.enums import StudioSectionKind as K
from src.core.uid import new_id
from src.studio.documents import (
    LyricLine,
    LyricsDocument,
    LyricsSection,
    PlanDocument,
    PlanSection,
    SyllableRange,
)
from src.studio.errors import DocumentIssue, StudioDocumentInvalid
from src.studio.validation import (
    MAX_DOCUMENT_BYTES,
    MAX_LYRICS_TEXT_CHARS,
    check_pair_compatible,
    collect_artist_style,
    is_instrumental,
    normalize_artist_style,
    normalize_block,
    normalize_lyrics,
    normalize_plan,
    normalize_text,
)
from tests.studio_support import make_lyrics, make_plan, make_section

pytestmark = pytest.mark.unit


def _issues(exc: pytest.ExceptionInfo[StudioDocumentInvalid]) -> set[tuple[str, str]]:
    return {(i.path, i.code) for i in exc.value.issues}


class TestNormalizeText:
    def test_trims_and_collapses_whitespace_including_newlines_and_nbsp(self) -> None:
        assert normalize_text("  a \t b\n\nc\u00a0d  ") == "a b c d"

    def test_composes_to_nfc(self) -> None:
        decomposed = unicodedata.normalize("NFD", "café")
        assert decomposed != "café"
        assert normalize_text(decomposed) == "café"

    def test_block_keeps_inner_line_breaks(self) -> None:
        assert normalize_block("  one\r\ntwo\rthree \n") == "one\ntwo\nthree"

    @pytest.mark.parametrize("blank", [None, "", "  \n\t "])
    def test_block_blank_is_none(self, blank: str | None) -> None:
        assert normalize_block(blank) is None


class TestNormalizePlan:
    def test_returns_normalized_copy(self) -> None:
        plan = make_plan(
            make_section(label="  Big   Hook ", tags=(" Synth  Pop",), delivery=("  soft ",)),
            title="  My   Song ",
            tags=("a  b", "C"),
            avoid=("x",),
            idea="  line1\r\nline2 ",
            references=(" Some   Artist ",),
        )
        out = normalize_plan(plan)
        assert out.title == "My Song"
        assert out.idea == "line1\nline2"
        assert out.references == ("Some Artist",)
        assert out.global_style.tags == ("a b", "C")
        assert out.sections[0].label == "Big Hook"
        assert out.sections[0].style.tags == ("Synth Pop",)
        assert out.sections[0].delivery == ("soft",)

    def test_normalization_is_idempotent(self) -> None:
        once = normalize_plan(make_plan(make_section(tags=("A  b",)), title=" T "))
        assert normalize_plan(once) == once

    def test_empty_after_normalization_everywhere(self) -> None:
        plan = make_plan(
            make_section(label="  ", tags=(" ",), avoid=("\t",), delivery=("  ",)),
            title="   ",
            tags=(" ",),
            references=("  ",),
        )
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(plan)
        assert _issues(exc) >= {
            ("title", "empty_after_normalization"),
            ("references[0]", "empty_after_normalization"),
            ("global_style.tags[0]", "empty_after_normalization"),
            ("sections[0].label", "empty_after_normalization"),
            ("sections[0].style.tags[0]", "empty_after_normalization"),
            ("sections[0].style.avoid[0]", "empty_after_normalization"),
            ("sections[0].delivery[0]", "empty_after_normalization"),
        }

    def test_duplicate_section_id(self) -> None:
        shared = new_id()
        plan = make_plan(make_section(section_id=shared), make_section(section_id=shared))
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(plan)
        assert _issues(exc) == {("sections[1].id", "duplicate_section_id")}

    @pytest.mark.parametrize(
        ("first", "second"), [("Pop", "pop"), ("pop", "  POP "), ("Рок", "рок"), ("ß", "SS")]
    )
    def test_duplicate_tag_is_casefold_based(self, first: str, second: str) -> None:
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(make_plan(tags=(first, second)))
        assert _issues(exc) == {("global_style.tags[1]", "duplicate_tag")}

    def test_duplicate_tag_covers_avoid_and_delivery(self) -> None:
        plan = make_plan(make_section(avoid=("a", "A"), delivery=("x", "X")))
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(plan)
        assert _issues(exc) == {
            ("sections[0].style.avoid[1]", "duplicate_tag"),
            ("sections[0].delivery[1]", "duplicate_tag"),
        }

    def test_tag_conflict_between_tags_and_avoid(self) -> None:
        plan = make_plan(
            make_section(tags=("Дарк",), avoid=("дарк",)), tags=("Rock",), avoid=("rock",)
        )
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(plan)
        assert _issues(exc) == {
            ("global_style.avoid[0]", "tag_conflict"),
            ("sections[0].style.avoid[0]", "tag_conflict"),
        }

    def test_same_tag_in_different_scopes_is_not_a_conflict(self) -> None:
        normalize_plan(make_plan(make_section(avoid=("loud",)), tags=("loud",)))

    @pytest.mark.parametrize("bad", ["a[b", "a]b", "[x]"])
    def test_reserved_characters_in_tags_delivery_and_labels(self, bad: str) -> None:
        plan = make_plan(
            make_section(label=bad, tags=(bad,), avoid=(f"{bad}z",), delivery=(bad,)), tags=(bad,)
        )
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(plan)
        assert _issues(exc) == {
            ("global_style.tags[0]", "reserved_characters"),
            ("sections[0].label", "reserved_characters"),
            ("sections[0].style.tags[0]", "reserved_characters"),
            ("sections[0].style.avoid[0]", "reserved_characters"),
            ("sections[0].delivery[0]", "reserved_characters"),
        }

    def test_reserved_characters_allowed_in_free_text(self) -> None:
        out = normalize_plan(
            make_plan(make_section(notes="see [1]"), idea="a [b]", references=("[x]",))
        )
        assert out.idea == "a [b]"

    def test_total_duration_cap(self) -> None:
        ok = make_plan(*[make_section(duration_seconds=600) for _ in range(3)])
        normalize_plan(ok)  # exactly 1800 is allowed
        too_long = make_plan(
            *[make_section(duration_seconds=600) for _ in range(3)],
            make_section(duration_seconds=1),
        )
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(too_long)
        assert _issues(exc) == {("sections", "total_duration_exceeded")}

    def test_sections_without_duration_do_not_count(self) -> None:
        normalize_plan(make_plan(*[make_section() for _ in range(64)]))

    def test_collects_all_issues_in_one_error(self) -> None:
        shared = new_id()
        plan = make_plan(
            make_section(section_id=shared, tags=("a", "A")),
            make_section(section_id=shared, label="[x]"),
            title=" ",
            tags=("p",),
            avoid=("P",),
        )
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(plan)
        assert _issues(exc) == {
            ("title", "empty_after_normalization"),
            ("global_style.avoid[0]", "tag_conflict"),
            ("sections[0].style.tags[1]", "duplicate_tag"),
            ("sections[1].id", "duplicate_section_id"),
            ("sections[1].label", "reserved_characters"),
        }

    def test_issue_paths_never_contain_document_text(self) -> None:
        secret = "SECRET-TAG-TEXT"
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(make_plan(tags=(secret, secret.lower())))
        assert all(secret.lower() not in repr(i).lower() for i in exc.value.issues)


class TestNormalizeLyrics:
    def _plan(self):
        verse, chorus, inst = (
            make_section(K.VERSE),
            make_section(K.CHORUS),
            make_section(K.INSTRUMENTAL),
        )
        return make_plan(verse, chorus, inst), verse, chorus, inst

    def test_normalizes_lines_and_delivery(self) -> None:
        plan, verse, *_ = self._plan()
        lyrics = LyricsDocument(
            language="en",
            sections=(
                LyricsSection(
                    section_id=verse.id,
                    lines=(LyricLine(text="  hello   world ", delivery=(" soft  ",)),),
                ),
            ),
        )
        out = normalize_lyrics(lyrics, plan)
        assert out.sections[0].lines[0].text == "hello world"
        assert out.sections[0].lines[0].delivery == ("soft",)

    def test_empty_after_normalization(self) -> None:
        plan, verse, *_ = self._plan()
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(
                make_lyrics((verse.id, ["  ", LyricLine(text="x", delivery=(" ",))])), plan
            )
        assert _issues(exc) == {
            ("sections[0].lines[0].text", "empty_after_normalization"),
            ("sections[0].lines[1].delivery[0]", "empty_after_normalization"),
        }

    def test_unknown_section(self) -> None:
        plan, *_ = self._plan()
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(make_lyrics((new_id(), ["x"])), plan)
        assert _issues(exc) == {("sections[0].section_id", "unknown_section")}

    def test_duplicate_section(self) -> None:
        plan, verse, *_ = self._plan()
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(make_lyrics((verse.id, ["a"]), (verse.id, ["b"])), plan)
        assert _issues(exc) == {("sections[1].section_id", "duplicate_section")}

    def test_instrumental_section_has_lines(self) -> None:
        plan, _, _, inst = self._plan()
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(make_lyrics((inst.id, ["la"])), plan)
        assert _issues(exc) == {("sections[0]", "instrumental_section_has_lines")}

    def test_instrumental_flag_on_a_vocal_kind_counts(self) -> None:
        flagged = make_section(K.VERSE, instrumental=True)
        assert is_instrumental(flagged)
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(make_lyrics((flagged.id, ["la"])), make_plan(flagged))
        assert _issues(exc) == {("sections[0]", "instrumental_section_has_lines")}

    @pytest.mark.parametrize("bad", ["a[b", "x]"])
    def test_reserved_characters_in_text_and_delivery(self, bad: str) -> None:
        plan, verse, *_ = self._plan()
        lyrics = make_lyrics((verse.id, [LyricLine(text=bad, delivery=(bad,))]))
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(lyrics, plan)
        assert _issues(exc) == {
            ("sections[0].lines[0].text", "reserved_characters"),
            ("sections[0].lines[0].delivery[0]", "reserved_characters"),
        }

    def test_parentheses_are_allowed(self) -> None:
        plan, verse, *_ = self._plan()
        out = normalize_lyrics(make_lyrics((verse.id, ["oh (oh oh)"])), plan)
        assert out.sections[0].lines[0].text == "oh (oh oh)"

    def test_syllable_range_inverted(self) -> None:
        plan, *_ = self._plan()
        lyrics = LyricsDocument(language="en", line_syllables=SyllableRange(min=9, max=3))
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(lyrics, plan)
        assert _issues(exc) == {("line_syllables", "syllable_range_inverted")}

    def test_equal_syllable_bounds_are_fine(self) -> None:
        plan, *_ = self._plan()
        normalize_lyrics(
            LyricsDocument(language="en", line_syllables=SyllableRange(min=4, max=4)), plan
        )

    def test_multiple_issues_are_collected(self) -> None:
        plan, verse, *_ = self._plan()
        lyrics = make_lyrics((verse.id, [" "]), (new_id(), ["x"]))
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(lyrics, plan)
        assert len(exc.value.issues) == 2

    def test_empty_lyrics_are_valid(self) -> None:
        plan, *_ = self._plan()
        assert normalize_lyrics(LyricsDocument(language="en"), plan).sections == ()


def _encoded_size(document: PlanDocument | LyricsDocument) -> int:
    return len(msgspec.json.encode(document))


def _plan_of_size(target_bytes: int) -> PlanDocument:
    """A wire-valid plan whose normalized compact JSON is exactly ``target_bytes``.

    Bulk comes from sections that each carry a full list of 40 distinct 60-char tags; the padded
    ``idea`` (<= 4000 chars, the wire cap) absorbs the remainder to the byte. The result is
    already in normalized form, so its own encoding is the measured size — no ``normalize_plan``
    call, which would itself reject targets above the cap. Every field respects its wire limit,
    because the limits are re-checked after normalization (RD6).
    """

    def loaded_section(index: int) -> PlanSection:
        tags = [f"{index:02d}-{n:02d}".ljust(60, "t") for n in range(40)]
        return make_section(K.VERSE, section_id=UUID(int=index + 1), tags=tags)

    sections: list[PlanSection] = []
    while True:
        grown = [*sections, loaded_section(len(sections))]
        if len(grown) > 64 or _encoded_size(make_plan(*grown, idea="x")) > target_bytes:
            break
        sections = grown
    probe = _encoded_size(make_plan(*sections, idea="x"))
    idea_chars = 1 + target_bytes - probe
    assert 1 <= idea_chars <= 4000, "the remainder must fit in `idea`"
    plan = make_plan(*sections, idea="x" * idea_chars)
    assert _encoded_size(plan) == target_bytes
    return plan


def _long_lyrics(
    total_chars: int, *, line_chars: int = 300, char: str = "x"
) -> tuple[PlanDocument, LyricsDocument]:
    """A plan and lyrics whose line text totals exactly ``total_chars`` code points.

    Lines are ``line_chars`` long and spread over sections of at most 60 lines, so the wire
    caps (64 lines per section, 300 chars per line) hold.
    """
    full, rest = divmod(total_chars, line_chars)
    lines = [char * line_chars] * full + ([char * rest] if rest else [])
    chunks = [lines[i : i + 60] for i in range(0, len(lines), 60)]
    sections = [make_section(K.VERSE) for _ in chunks]
    plan = make_plan(*sections)
    lyrics = make_lyrics(*[(s.id, chunk) for s, chunk in zip(sections, chunks, strict=True)])
    return plan, lyrics


class TestDocumentSizeCaps:
    """Invariant 25: size caps are validation issues, measured after normalization (RD1)."""

    def test_limits_are_the_locked_values(self) -> None:
        assert (MAX_LYRICS_TEXT_CHARS, MAX_DOCUMENT_BYTES) == (20_000, 131_072)

    def test_lyrics_text_of_exactly_the_cap_passes(self) -> None:
        plan, lyrics = _long_lyrics(MAX_LYRICS_TEXT_CHARS)
        out = normalize_lyrics(lyrics, plan)
        assert sum(len(line.text) for s in out.sections for line in s.lines) == 20_000

    def test_one_character_over_the_cap_is_lyrics_too_long(self) -> None:
        plan, lyrics = _long_lyrics(MAX_LYRICS_TEXT_CHARS + 1)
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(lyrics, plan)
        assert _issues(exc) == {("sections", "lyrics_too_long")}

    def test_the_cap_counts_normalized_text_not_the_raw_request(self) -> None:
        # Raw lines carry 20 padding spaces each; after trim they total exactly the cap.
        sections = [make_section(K.VERSE) for _ in range(4)]
        padded = [f"  {'x' * 100}  " + " " * 16 for _ in range(50)]
        lyrics = make_lyrics(*[(s.id, padded) for s in sections])
        assert sum(len(line.text) for s in lyrics.sections for line in s.lines) > 20_000
        out = normalize_lyrics(lyrics, make_plan(*sections))
        assert sum(len(line.text) for s in out.sections for line in s.lines) == 20_000

    def test_the_cap_counts_code_points_and_cyrillic_stays_under_the_byte_cap(self) -> None:
        # 20,000 Cyrillic characters in 400 lines of 50: ~2 bytes each, still far below 128 KiB.
        plan, lyrics = _long_lyrics(20_000, line_chars=50, char="я")
        out = normalize_lyrics(lyrics, plan)
        size = _encoded_size(out)
        assert 2 * 20_000 < size < MAX_DOCUMENT_BYTES

    def test_a_plan_of_exactly_the_byte_cap_passes(self) -> None:
        out = normalize_plan(_plan_of_size(MAX_DOCUMENT_BYTES))
        assert _encoded_size(out) == MAX_DOCUMENT_BYTES

    def test_one_byte_over_the_cap_is_document_too_large(self) -> None:
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(_plan_of_size(MAX_DOCUMENT_BYTES + 1))
        assert _issues(exc) == {("", "document_too_large")}

    def test_the_byte_cap_is_measured_after_normalization(self) -> None:
        # A raw idea of 200k blank-padded characters normalizes down to a tiny document.
        raw = make_plan(idea="x" + " " * 200_000)
        assert normalize_plan(raw).idea == "x"

    def test_lyrics_over_the_byte_cap_with_few_characters_is_document_too_large(self) -> None:
        # Thousands of 1-char lines: little text, but 26+ bytes of JSON overhead per line.
        verse = make_section(K.VERSE)
        lines = tuple(LyricLine(text="a") for _ in range(6_000))
        lyrics = LyricsDocument(
            language="en", sections=(LyricsSection(section_id=verse.id, lines=lines),)
        )
        assert _encoded_size(lyrics) > MAX_DOCUMENT_BYTES
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(lyrics, make_plan(verse))
        assert _issues(exc) == {("", "document_too_large")}

    def test_both_codes_are_collected_with_every_other_issue(self) -> None:
        verse = make_section(K.VERSE)
        big = LyricsSection(
            section_id=verse.id, lines=tuple(LyricLine(text="abcd") for _ in range(6_000))
        )
        stray = LyricsSection(section_id=new_id(), lines=(LyricLine(text=" "),))
        lyrics = LyricsDocument(language="en", sections=(big, stray))
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(lyrics, make_plan(verse))
        assert _issues(exc) == {
            ("sections[1].lines[0].text", "empty_after_normalization"),
            ("sections[1].section_id", "unknown_section"),
            ("sections", "lyrics_too_long"),
            ("", "document_too_large"),
        }

    def test_plan_size_is_collected_with_other_plan_issues(self) -> None:
        # The blank title below normalizes to "", so pad by the default title's length to keep
        # the *normalized* document exactly one byte over the cap.
        plan = _plan_of_size(MAX_DOCUMENT_BYTES + 1 + len("Song"))
        broken = PlanDocument(
            title=" ",
            idea=plan.idea,
            global_style=plan.global_style,
            sections=plan.sections,
        )
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(broken)
        assert _issues(exc) == {
            ("title", "empty_after_normalization"),
            ("", "document_too_large"),
        }

    def test_issue_paths_stay_text_free(self) -> None:
        plan, lyrics = _long_lyrics(MAX_LYRICS_TEXT_CHARS + 1, char="S")
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(lyrics, plan)
        assert all("SSS" not in repr(i) for i in exc.value.issues)


class TestCheckPairCompatible:
    def test_compatible_pair_has_no_issues(self) -> None:
        verse = make_section(K.VERSE)
        assert check_pair_compatible(make_plan(verse), make_lyrics((verse.id, ["x"]))) == ()

    def test_reports_only_the_plan_dependent_issues(self) -> None:
        verse, inst = make_section(K.VERSE), make_section(K.INSTRUMENTAL)
        lyrics = make_lyrics((new_id(), ["x"]), (inst.id, ["y"]), (verse.id, [" "]))
        assert check_pair_compatible(make_plan(verse, inst), lyrics) == (
            DocumentIssue("sections[0].section_id", "unknown_section"),
            DocumentIssue("sections[1]", "instrumental_section_has_lines"),
        )


class TestArtistStyle:
    def test_normalizes_all_three_lists(self) -> None:
        out = normalize_artist_style(
            style_tags=(" Husky  vocals",),
            global_avoid=(" auto tune ",),
            aesthetic_references=("  1980s  Berlin ",),
        )
        assert out.style_tags == ("Husky vocals",)
        assert out.global_avoid == ("auto tune",)
        assert out.aesthetic_references == ("1980s Berlin",)

    def test_validated_like_a_style_spec(self) -> None:
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_artist_style(
                style_tags=("a", "A", "[x]", " "),
                global_avoid=("A", "b"),
                aesthetic_references=(" ",),
            )
        assert _issues(exc) == {
            ("style_tags[1]", "duplicate_tag"),
            ("style_tags[2]", "reserved_characters"),
            ("style_tags[3]", "empty_after_normalization"),
            ("global_avoid[0]", "tag_conflict"),
            ("aesthetic_references[0]", "empty_after_normalization"),
        }

    def test_aesthetic_references_may_repeat_and_contain_brackets(self) -> None:
        out = normalize_artist_style(
            style_tags=(), global_avoid=(), aesthetic_references=("[a]", "[a]")
        )
        assert out.aesthetic_references == ("[a]", "[a]")


# U+0958 (DEVANAGARI LETTER QA) is a composition exclusion: NFC turns it into 2 code points
# (U+0915 U+093C) and never recomposes. U+FB2C (HEBREW LETTER SHIN WITH DAGESH AND SHIN DOT)
# turns into 3. A value within its wire limit as sent can leave normalization past it.
QA = "\u0958"
SHIN = "\ufb2c"
TOO_LONG = "too_long_after_normalization"


class TestLimitsAreRecheckedAfterNormalization:
    """Invariant 31: no stored value exceeds the limit its wire Struct declares."""

    def test_the_premise_nfc_expands_these_characters(self) -> None:
        assert len(normalize_text(QA)) == 2
        assert len(normalize_text(SHIN)) == 3
        assert normalize_text(normalize_text(QA)) == normalize_text(QA)  # idempotent: stays long

    def test_plan_title_that_expands_past_200_is_refused(self) -> None:
        raw = make_plan(title=QA * 101)
        assert len(raw.title) == 101  # within the wire limit as sent
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(raw)
        assert _issues(exc) == {("title", TOO_LONG)}

    def test_plan_title_at_exactly_the_limit_after_expansion_passes(self) -> None:
        out = normalize_plan(make_plan(title=QA * 100))
        assert len(out.title) == 200

    def test_lyric_line_that_expands_past_300_is_refused(self) -> None:
        verse = make_section(K.VERSE)
        lyrics = make_lyrics((verse.id, [SHIN * 101]))
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(lyrics, make_plan(verse))
        assert _issues(exc) == {("sections[0].lines[0].text", TOO_LONG)}

    def test_lyric_line_at_exactly_the_limit_after_expansion_passes(self) -> None:
        verse = make_section(K.VERSE)
        out = normalize_lyrics(make_lyrics((verse.id, [SHIN * 100])), make_plan(verse))
        assert len(out.sections[0].lines[0].text) == 300

    def test_every_plan_field_that_normalization_touches_is_rechecked(self) -> None:
        # Each value is within its wire limit as sent and over it once normalized.
        verse = make_section(
            K.VERSE,
            label=QA * 31,  # 60
            tags=("fine", QA * 31),  # 60
            avoid=(SHIN * 21,),  # 63
            delivery=(QA * 21,),  # 40
            notes=QA * 501,  # 1000
        )
        plan = make_plan(
            make_section(K.CHORUS),
            verse,
            title="ok",
            idea=QA * 2001,  # 4000
            references=("ok", QA * 101),  # 200
            tags=(QA * 31,),
            avoid=(SHIN * 21,),
        )
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(plan)
        assert _issues(exc) == {
            ("idea", TOO_LONG),
            ("references[1]", TOO_LONG),
            ("global_style.tags[0]", TOO_LONG),
            ("global_style.avoid[0]", TOO_LONG),
            ("sections[1].label", TOO_LONG),
            ("sections[1].style.tags[1]", TOO_LONG),
            ("sections[1].style.avoid[0]", TOO_LONG),
            ("sections[1].delivery[0]", TOO_LONG),
            ("sections[1].notes", TOO_LONG),
        }

    def test_lyric_delivery_tags_are_rechecked(self) -> None:
        verse = make_section(K.VERSE)
        lyrics = make_lyrics((verse.id, [LyricLine(text="ok", delivery=(QA * 21,))]))
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(lyrics, make_plan(verse))
        assert _issues(exc) == {("sections[0].lines[0].delivery[0]", TOO_LONG)}

    def test_it_is_collected_with_every_other_issue(self) -> None:
        plan = make_plan(
            make_section(K.VERSE, tags=("a", "A")),
            title=QA * 101,
            tags=(" ",),
        )
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(plan)
        assert _issues(exc) == {
            ("title", TOO_LONG),
            ("global_style.tags[0]", "empty_after_normalization"),
            ("sections[0].style.tags[1]", "duplicate_tag"),
        }

    def test_a_value_that_shrinks_below_its_limit_is_accepted(self) -> None:
        # The check is on the normalized value: 250 raw characters collapse to 100.
        # (Raw 250 would be refused by the wire; the walker only sees the normalized form.)
        out = normalize_plan(make_plan(title=" ".join(["ab"] * 40)))
        assert len(out.title) == 119

    def test_issue_paths_stay_text_free(self) -> None:
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(make_plan(title=QA * 101))
        assert all(QA not in repr(i) for i in exc.value.issues)

    def test_collect_artist_style_collects_instead_of_raising(self) -> None:
        issues: list[DocumentIssue] = []
        style = collect_artist_style(
            style_tags=("a", "A"), global_avoid=("a",), aesthetic_references=(" ",), issues=issues
        )
        assert {(i.path, i.code) for i in issues} == {
            ("style_tags[1]", "duplicate_tag"),
            ("global_avoid[0]", "tag_conflict"),
            ("aesthetic_references[0]", "empty_after_normalization"),
        }
        assert style.style_tags == ("a", "A")


class TestReservedNames:
    """P1b (I4): names the Suno importer would read back as something else are refused."""

    @pytest.mark.parametrize("word", ["instrumental", "Instrumental", "INSTRUMENTAL"])
    def test_instrumental_in_a_section_delivery_is_reserved(self, word: str) -> None:
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(make_plan(make_section(K.DROP, delivery=("loud", word))))
        assert _issues(exc) == {("sections[0].delivery[1]", "reserved_descriptor")}

    @pytest.mark.parametrize("word", ["instrumental", "Instrumental", "INSTRUMENTAL"])
    def test_instrumental_in_a_section_style_tag_is_reserved(self, word: str) -> None:
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(make_plan(make_section(K.DROP, tags=(word,))))
        assert _issues(exc) == {("sections[0].style.tags[0]", "reserved_descriptor")}

    def test_it_is_the_whole_word_that_is_reserved(self) -> None:
        plan = normalize_plan(
            make_plan(make_section(K.DROP, tags=("instrumentals", "non-instrumental")))
        )
        assert plan.sections[0].style.tags == ("instrumentals", "non-instrumental")

    def test_the_global_style_and_the_avoid_list_are_not_affected(self) -> None:
        plan = normalize_plan(
            make_plan(make_section(K.VERSE, avoid=("instrumental",)), tags=("instrumental",))
        )
        assert plan.global_style.tags == ("instrumental",)

    def test_the_word_is_checked_after_normalization(self) -> None:
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(make_plan(make_section(K.DROP, tags=("  Instrumental ",))))
        assert _issues(exc) == {("sections[0].style.tags[0]", "reserved_descriptor")}

    def test_it_is_collected_with_every_other_issue(self) -> None:
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(
                make_plan(make_section(K.DROP, tags=("instrumental", "a[b]")), title="  ")
            )
        assert _issues(exc) == {
            ("title", "empty_after_normalization"),
            ("sections[0].style.tags[0]", "reserved_descriptor"),
            ("sections[0].style.tags[1]", "reserved_characters"),
        }

    @pytest.mark.parametrize(
        "delivery",
        [
            ("Verse",),
            ("chorus",),
            ("Pre-Chorus 2",),
            ("Refrain",),
            ("fade out",),
            ("BREAK",),
            ("Verse: whispered",),
        ],
    )
    def test_a_line_delivery_that_reads_as_a_header_is_reserved(
        self, delivery: tuple[str, ...]
    ) -> None:
        verse = make_section(K.VERSE)
        plan = normalize_plan(make_plan(verse))
        lyrics = make_lyrics((verse.id, [LyricLine(text="la", delivery=delivery)]))
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(lyrics, plan)
        assert _issues(exc) == {("sections[0].lines[0].delivery", "reserved_delivery_tag")}

    @pytest.mark.parametrize(
        "delivery",
        [
            ("whispered",),
            ("soft", "close"),
            ("Verse x",),
            ("end of line",),
            ("Verse", "soft"),  # written "[Verse, soft]": the name part is "Verse, soft"
        ],
    )
    def test_ordinary_delivery_is_fine(self, delivery: tuple[str, ...]) -> None:
        verse = make_section(K.VERSE)
        plan = normalize_plan(make_plan(verse))
        lyrics = make_lyrics((verse.id, [LyricLine(text="la", delivery=delivery)]))
        assert normalize_lyrics(lyrics, plan).sections[0].lines[0].delivery == delivery

    def test_the_issue_is_at_the_line_and_is_collected_with_others(self) -> None:
        verse = make_section(K.VERSE)
        plan = normalize_plan(make_plan(verse))
        lyrics = make_lyrics(
            (
                verse.id,
                [
                    LyricLine(text="ok", delivery=("soft",)),
                    LyricLine(text="la", delivery=("Chorus",)),
                    LyricLine(text="a[b]"),
                ],
            )
        )
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(lyrics, plan)
        assert _issues(exc) == {
            ("sections[0].lines[1].delivery", "reserved_delivery_tag"),
            ("sections[0].lines[2].text", "reserved_characters"),
        }

    def test_stored_rows_with_these_values_still_decode(self) -> None:
        """Only new writes are refused: the decode path never applies the semantic rules."""
        from src.core.enums import StudioDocumentKind
        from src.studio.documents import decode_stored

        content = {
            "title": "Old",
            "sections": [
                {
                    "id": str(new_id()),
                    "kind": "drop",
                    "delivery": ["Instrumental"],
                    "style": {"tags": ["instrumental"], "avoid": []},
                }
            ],
        }
        plan = decode_stored(StudioDocumentKind.PLAN, 2, content)
        assert isinstance(plan, PlanDocument)
        assert plan.sections[0].delivery == ("Instrumental",)
        lyrics = {
            "language": "en",
            "sections": [
                {"section_id": str(new_id()), "lines": [{"text": "la", "delivery": ["Chorus"]}]}
            ],
        }
        decode_stored(StudioDocumentKind.LYRICS, 1, lyrics)


class TestReservedTempoTag:
    """R1: a tag the importer would read as the tempo is reserved — the tempo is ``tempo_bpm``."""

    @pytest.mark.parametrize(
        "tag", ["80 BPM", "120bpm", "120 bpm", "  90   Bpm ", "40 BPM", "240 BPM"]
    )
    def test_a_plan_global_tag_that_reads_as_a_tempo_is_refused(self, tag: str) -> None:
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(make_plan(tags=("rock", tag)))
        assert _issues(exc) == {("global_style.tags[1]", "reserved_tempo_tag")}

    @pytest.mark.parametrize(
        "tag", ["300 BPM", "39 BPM", "1200 BPM", "BPM 120", "120 BPMs", "5 BPM", "fast bpm"]
    )
    def test_anything_that_is_not_a_tempo_is_an_ordinary_tag(self, tag: str) -> None:
        assert normalize_plan(make_plan(tags=(tag,))).global_style.tags == (tag,)

    def test_the_other_lists_are_not_checked(self) -> None:
        """Nothing reads a tempo from section tags, delivery or an avoid list."""
        verse = make_section(K.VERSE, tags=("80 BPM",), delivery=("90 bpm",), avoid=("100 BPM",))
        plan = normalize_plan(make_plan(verse, avoid=("120 BPM",)))
        assert plan.global_style.avoid == ("120 BPM",)
        assert plan.sections[0].style.tags == ("80 BPM",)
        assert plan.sections[0].delivery == ("90 bpm",)
        assert plan.sections[0].style.avoid == ("100 BPM",)

    def test_it_is_collected_with_every_other_issue(self) -> None:
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(make_plan(tags=("80 BPM", "a[b]"), title="  "))
        assert _issues(exc) == {
            ("title", "empty_after_normalization"),
            ("global_style.tags[0]", "reserved_tempo_tag"),
            ("global_style.tags[1]", "reserved_characters"),
        }

    def test_the_tempo_itself_is_still_fine(self) -> None:
        assert normalize_plan(make_plan(tags=("rock",), tempo_bpm=120)).tempo_bpm == 120

    @pytest.mark.parametrize(
        ("tags", "tempo", "path"),
        [
            # the reproduction table of the remediation prompt, rows 1 and 4 (plan side)
            (("80 BPM",), 120, "global_style.tags[0]"),
            (("120 bpm",), 120, "global_style.tags[0]"),
        ],
    )
    def test_the_reproduction_rows_with_a_plan_tag_are_rejected_at_write(
        self, tags: tuple[str, ...], tempo: int, path: str
    ) -> None:
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_plan(make_plan(tags=tags, tempo_bpm=tempo))
        assert _issues(exc) == {(path, "reserved_tempo_tag")}

    @pytest.mark.parametrize("tag", ["80 BPM", "90bpm"])
    def test_the_reproduction_rows_with_an_artist_tag_are_rejected_at_write(self, tag: str) -> None:
        """Rows 2 and 3: the artist's tag is rendered into every Style box of the project."""
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_artist_style(style_tags=("rock", tag), global_avoid=())
        assert _issues(exc) == {("style_tags[1]", "reserved_tempo_tag")}

    def test_an_artists_avoid_list_and_references_are_not_checked(self) -> None:
        style = normalize_artist_style(
            style_tags=("300 BPM",), global_avoid=("120 BPM",), aesthetic_references=("90 BPM",)
        )
        assert (style.style_tags, style.global_avoid) == (("300 BPM",), ("120 BPM",))

    def test_collect_artist_style_collects_it_instead_of_raising(self) -> None:
        issues: list[DocumentIssue] = []
        collect_artist_style(style_tags=("80 BPM",), global_avoid=(), issues=issues)
        assert [(i.path, i.code) for i in issues] == [("style_tags[0]", "reserved_tempo_tag")]

    def test_stored_rows_with_a_tempo_tag_still_decode(self) -> None:
        """Only new writes are refused (the P1b precedent): the decode path applies no rule."""
        from src.core.enums import StudioDocumentKind
        from src.studio.documents import decode_stored

        content = {
            "title": "Old",
            "tempo_bpm": 120,
            "global_style": {"tags": ["80 BPM"], "avoid": []},
            "sections": [],
        }
        plan = decode_stored(StudioDocumentKind.PLAN, 2, content)
        assert isinstance(plan, PlanDocument)
        assert plan.global_style.tags == ("80 BPM",)


class TestInstrumentalPlanAndEmptyPlan:
    def test_the_flag_is_normalized_through(self) -> None:
        assert normalize_plan(make_plan(instrumental=True)).instrumental is True
        assert normalize_plan(make_plan()).instrumental is False

    def test_a_plan_may_have_no_sections(self) -> None:
        plan = normalize_plan(make_plan(empty=True, tags=("lofi",)))
        assert plan.sections == ()

    def test_an_instrumental_plan_refuses_lyrics_with_lines(self) -> None:
        verse = make_section(K.VERSE)
        plan = normalize_plan(make_plan(verse, instrumental=True))
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(make_lyrics((verse.id, ["la"])), plan)
        assert _issues(exc) == {("sections", "instrumental_plan_has_lines")}

    def test_it_is_one_issue_besides_the_per_section_ones(self) -> None:
        drop = make_section(K.DROP, instrumental=True)
        verse = make_section(K.VERSE)
        plan = normalize_plan(make_plan(drop, verse, instrumental=True))
        lyrics = make_lyrics((drop.id, ["a"]), (verse.id, ["b", "c"]))
        issues = check_pair_compatible(plan, lyrics)
        assert [(i.path, i.code) for i in issues] == [
            ("sections", "instrumental_plan_has_lines"),
            ("sections[0]", "instrumental_section_has_lines"),
        ]

    def test_an_instrumental_plan_accepts_lyrics_without_lines(self) -> None:
        verse = make_section(K.VERSE)
        plan = normalize_plan(make_plan(verse, instrumental=True))
        assert normalize_lyrics(LyricsDocument(language="en"), plan).sections == ()
        assert check_pair_compatible(plan, LyricsDocument(language="en")) == ()

    def test_a_plan_without_sections_accepts_lyrics_only_without_sections(self) -> None:
        plan = normalize_plan(make_plan(empty=True))
        assert normalize_lyrics(LyricsDocument(language="en"), plan).sections == ()
        with pytest.raises(StudioDocumentInvalid) as exc:
            normalize_lyrics(make_lyrics((new_id(), ["la"])), plan)
        assert _issues(exc) == {("sections[0].section_id", "unknown_section")}
