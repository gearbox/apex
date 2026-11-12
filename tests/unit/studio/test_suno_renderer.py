"""Suno renderer: golden output, numbering rules, merge order, warnings, limits, determinism."""

from __future__ import annotations

import dataclasses

import pytest

from src.core.enums import StudioExportTarget
from src.core.enums import StudioSectionKind as K
from src.studio.documents import LyricLine, PlanDocument
from src.studio.renderers import RENDERERS
from src.studio.renderers.base import ArtistRenderInput, RenderContext
from src.studio.renderers.projection import to_renderable_plan
from src.studio.renderers.suno import (
    EXCLUDE_LIMIT,
    LYRICS_LIMIT,
    STYLE_LIMIT,
    VERSION,
)
from src.studio.validation import normalize_lyrics, normalize_plan
from tests.studio_support import make_lyrics, make_plan, make_section

pytestmark = pytest.mark.unit

SUNO = RENDERERS[StudioExportTarget.SUNO]


def _render(plan: PlanDocument, lyrics=None, artist: ArtistRenderInput | None = None):
    return SUNO.render(RenderContext(plan=to_renderable_plan(plan), lyrics=lyrics, artist=artist))


def _text(export, name: str) -> str:
    return export.fields[name].text


def _warning_codes(export) -> list[str]:
    return [w.code for w in export.warnings]


class TestGolden:
    """A multi-section plan with repeated verses / choruses / drops, local tags and delivery."""

    def build(self):
        intro = make_section(K.INTRO, tags=("sparse pads",))
        verse_a = make_section(K.VERSE, delivery=("breathy",), tags=("minimal",))
        pre = make_section(K.PRE_CHORUS)
        chorus_1 = make_section(K.CHORUS)
        verse_b = make_section(K.VERSE)
        chorus_2 = make_section(K.CHORUS)
        drop = make_section(
            K.DROP, tags=("heavy bass",), avoid=("distortion",), duration_seconds=20
        )
        instrumental_drop = make_section(K.DROP, instrumental=True)
        bridge = make_section(K.BRIDGE, label="Break It Down")
        outro = make_section(K.OUTRO)
        plan = normalize_plan(
            make_plan(
                intro,
                verse_a,
                pre,
                chorus_1,
                verse_b,
                chorus_2,
                drop,
                instrumental_drop,
                bridge,
                outro,
                title="Neon Rain",
                tempo_bpm=118,
                tags=("synthwave", "female vocals", "Dreamy"),
                avoid=("heavy metal", "autotune"),
                idea="SECRET IDEA",
                references=("SECRET REFERENCE",),
            )
        )
        lyrics = normalize_lyrics(
            make_lyrics(
                (
                    verse_a.id,
                    ["First line", LyricLine(text="Second line", delivery=("whispered",))],
                ),
                (pre.id, ["Pre line"]),
                (chorus_1.id, ["Chorus one", "Chorus two"]),
                (verse_b.id, ["Verse b line"]),
                (chorus_2.id, ["Chorus again"]),
            ),
            plan,
        )
        return plan, lyrics, drop

    def test_output(self) -> None:
        plan, lyrics, drop = self.build()
        artist = ArtistRenderInput(
            style_tags=("dreamy", "Warm Reverb"), global_avoid=("Autotune", "screaming")
        )
        export = _render(plan, lyrics, artist)

        assert export.renderer_version == VERSION == "suno-v6.2026-11"
        assert export.target is StudioExportTarget.SUNO
        assert _text(export, "title") == "Neon Rain"
        assert _text(export, "style") == "synthwave, female vocals, Dreamy, Warm Reverb, 118 BPM"
        assert _text(export, "exclude") == "heavy metal, autotune, screaming"
        assert _text(export, "lyrics") == (
            "[Intro: sparse pads]\n"
            "\n"
            "[Verse 1: breathy, minimal]\n"
            "First line\n"
            "[whispered]\n"
            "Second line\n"
            "\n"
            "[Pre-Chorus]\n"
            "Pre line\n"
            "\n"
            "[Chorus]\n"
            "Chorus one\n"
            "Chorus two\n"
            "\n"
            "[Verse 2]\n"
            "Verse b line\n"
            "\n"
            "[Chorus]\n"
            "Chorus again\n"
            "\n"
            "[Drop 1: heavy bass]\n"
            "\n"
            "[Drop 2: instrumental]\n"
            "\n"
            "[Break It Down]\n"
            "\n"
            "[Outro]"
        )
        assert [(w.code, w.section_id) for w in export.warnings] == [
            ("local_avoid_not_exportable", drop.id),
            ("section_duration_not_exportable", drop.id),
        ]

    def test_never_contains_non_exportable_text_or_parentheses(self) -> None:
        plan, lyrics, _ = self.build()
        export = _render(plan, lyrics)
        everything = "\n".join(f.text for f in export.fields.values())
        for forbidden in ("SECRET IDEA", "SECRET REFERENCE", "(", ")"):
            assert forbidden not in everything

    def test_parentheses_only_appear_when_present_in_line_text(self) -> None:
        verse = make_section(K.VERSE)
        plan = normalize_plan(make_plan(verse))
        lyrics = normalize_lyrics(make_lyrics((verse.id, ["la (la la)"])), plan)
        assert "la (la la)" in _text(_render(plan, lyrics), "lyrics")

    def test_double_render_is_byte_identical(self) -> None:
        plan, lyrics, _ = self.build()
        artist = ArtistRenderInput(style_tags=("a",), global_avoid=("b",))
        first, second = _render(plan, lyrics, artist), _render(plan, lyrics, artist)
        assert first == second
        assert {k: v.text.encode() for k, v in first.fields.items()} == {
            k: v.text.encode() for k, v in second.fields.items()
        }


class TestSectionNumbering:
    @pytest.mark.parametrize(
        ("kinds", "expected"),
        [
            ([K.VERSE], ["[Verse]"]),
            ([K.VERSE, K.VERSE], ["[Verse 1]", "[Verse 2]"]),
            (
                [K.VERSE, K.CHORUS, K.VERSE, K.VERSE],
                ["[Verse 1]", "[Chorus]", "[Verse 2]", "[Verse 3]"],
            ),
            ([K.CHORUS, K.CHORUS], ["[Chorus]", "[Chorus]"]),
            ([K.PRE_CHORUS, K.PRE_CHORUS], ["[Pre-Chorus]", "[Pre-Chorus]"]),
            ([K.POST_CHORUS, K.POST_CHORUS], ["[Post-Chorus]", "[Post-Chorus]"]),
            ([K.HOOK, K.HOOK], ["[Hook]", "[Hook]"]),
            ([K.DROP, K.DROP], ["[Drop 1]", "[Drop 2]"]),
            ([K.BREAKDOWN, K.BREAKDOWN], ["[Breakdown 1]", "[Breakdown 2]"]),
            ([K.BUILD, K.BUILD], ["[Build 1]", "[Build 2]"]),
            ([K.INTERLUDE, K.INTERLUDE], ["[Interlude 1]", "[Interlude 2]"]),
            ([K.INSTRUMENTAL, K.INSTRUMENTAL], ["[Instrumental 1]", "[Instrumental 2]"]),
            ([K.INTRO, K.OUTRO, K.BRIDGE], ["[Intro]", "[Outro]", "[Bridge]"]),
        ],
    )
    def test_default_labels(self, kinds: list[K], expected: list[str]) -> None:
        plan = normalize_plan(make_plan(*[make_section(k) for k in kinds]))
        assert _text(_render(plan), "lyrics").split("\n\n") == expected

    def test_explicit_label_wins_and_occurrence_counts_every_section_of_the_kind(self) -> None:
        plan = normalize_plan(
            make_plan(make_section(K.VERSE, label="Verse A"), make_section(K.VERSE))
        )
        assert _text(_render(plan), "lyrics").split("\n\n") == ["[Verse A]", "[Verse 2]"]


class TestStyleAndExclude:
    def test_plan_tags_first_then_artist_then_bpm(self) -> None:
        plan = normalize_plan(make_plan(tags=("b", "a"), tempo_bpm=90))
        export = _render(plan, artist=ArtistRenderInput(style_tags=("c", "A"), global_avoid=()))
        assert _text(export, "style") == "b, a, c, 90 BPM"

    def test_dedupe_is_case_insensitive_and_first_wins(self) -> None:
        plan = normalize_plan(make_plan(tags=("Рок",), avoid=("Шум",)))
        artist = ArtistRenderInput(style_tags=("рок", "pop"), global_avoid=("шум", "x"))
        export = _render(plan, artist=artist)
        assert _text(export, "style") == "Рок, pop"
        assert _text(export, "exclude") == "Шум, x"

    def test_no_artist_and_no_tempo(self) -> None:
        export = _render(normalize_plan(make_plan(tags=("a",))))
        assert _text(export, "style") == "a"
        assert _text(export, "exclude") == ""

    def test_empty_everything_renders_empty_fields(self) -> None:
        export = _render(normalize_plan(make_plan()))
        assert _text(export, "style") == ""
        assert export.fields["style"].report.length == 0

    def test_overlap_after_merge_is_kept_in_both_and_warned(self) -> None:
        plan = normalize_plan(make_plan(tags=("lofi",), avoid=("jazz",)))
        export = _render(plan, artist=ArtistRenderInput(style_tags=("Jazz",), global_avoid=()))
        assert _text(export, "style") == "lofi, Jazz"
        assert _text(export, "exclude") == "jazz"
        assert _warning_codes(export) == ["style_exclude_overlap"]


class TestSectionDescriptors:
    def test_delivery_then_style_tags_deduped_ci(self) -> None:
        section = make_section(K.VERSE, delivery=("soft", "Raspy"), tags=("raspy", "dark"))
        plan = normalize_plan(make_plan(section))
        assert _text(_render(plan), "lyrics") == "[Verse: soft, Raspy, dark]"

    def test_an_instrumental_flag_on_another_kind_is_cued_not_dropped(self) -> None:
        # Review probe: this used to render a bare "[Drop]" with no warning and no cue.
        plan = to_renderable_plan(make_plan(make_section(K.DROP, instrumental=True)))
        export = SUNO.render(RenderContext(plan=plan, lyrics=None, artist=None))
        assert _text(export, "lyrics") == "[Drop: instrumental]"
        assert export.warnings == ()

    def test_the_cue_goes_first_and_counts_toward_the_length(self) -> None:
        section = make_section(K.DROP, instrumental=True, delivery=("massive",), tags=("heavy",))
        export = _render(normalize_plan(make_plan(section)))
        assert _text(export, "lyrics") == "[Drop: instrumental, massive, heavy]"
        assert export.fields["lyrics"].report.length == len("[Drop: instrumental, massive, heavy]")

    @pytest.mark.parametrize("typed", ["Instrumental", "INSTRUMENTAL", "instrumental"])
    def test_a_typed_instrumental_descriptor_is_deduped_case_insensitively(
        self, typed: str
    ) -> None:
        by_delivery = make_section(K.DROP, instrumental=True, delivery=(typed,))
        by_tag = make_section(K.DROP, instrumental=True, tags=(typed,))
        for section in (by_delivery, by_tag):
            export = _render(normalize_plan(make_plan(section)))
            assert _text(export, "lyrics") == "[Drop: instrumental]"  # the cue was first

    def test_the_cue_applies_to_vocal_kinds_and_custom_labels(self) -> None:
        plan = normalize_plan(
            make_plan(
                make_section(K.VERSE, instrumental=True),
                make_section(K.BRIDGE, label="Break", instrumental=True),
            )
        )
        assert _text(_render(plan), "lyrics").split("\n\n") == [
            "[Verse: instrumental]",
            "[Break: instrumental]",
        ]

    @pytest.mark.parametrize("flag", [True, False])
    def test_an_instrumental_kind_section_gets_no_extra_cue(self, flag: bool) -> None:
        plan = normalize_plan(
            make_plan(
                make_section(K.INSTRUMENTAL, instrumental=flag),
                make_section(K.INSTRUMENTAL, instrumental=flag, tags=("pads",)),
            )
        )
        assert _text(_render(plan), "lyrics").split("\n\n") == [
            "[Instrumental 1]",
            "[Instrumental 2: pads]",
        ]

    def test_a_section_that_is_not_instrumental_gets_no_cue(self) -> None:
        plan = normalize_plan(make_plan(make_section(K.DROP, instrumental=False)))
        assert _text(_render(plan), "lyrics") == "[Drop]"

    def test_vocal_section_without_lyrics_renders_header_only(self) -> None:
        plan = normalize_plan(make_plan(make_section(K.VERSE), make_section(K.CHORUS)))
        assert _text(_render(plan, make_lyrics()), "lyrics") == "[Verse]\n\n[Chorus]"

    def test_title_is_reported_without_limit(self) -> None:
        field = _render(normalize_plan(make_plan(title="Song"))).fields["title"]
        assert (field.report.length, field.report.limit, field.report.within_limit) == (
            4,
            None,
            True,
        )


class TestWarnings:
    def test_every_code(self) -> None:
        a = make_section(K.VERSE, avoid=("x",), duration_seconds=10)
        b = make_section(K.CHORUS, duration_seconds=5)
        plan = normalize_plan(make_plan(a, b, tags=("t",), avoid=("T2",)))
        lines = ["l" * 100] * 31  # > 3000 chars once joined
        lyrics = normalize_lyrics(make_lyrics((a.id, lines)), plan)
        export = _render(plan, lyrics, ArtistRenderInput(style_tags=("t2",), global_avoid=()))
        assert [(w.code, w.section_id) for w in export.warnings] == [
            ("style_exclude_overlap", None),
            ("lyrics_above_recommended_length", None),
            ("local_avoid_not_exportable", a.id),
            ("section_duration_not_exportable", a.id),
            ("section_duration_not_exportable", b.id),
        ]

    @pytest.mark.parametrize(("last_line", "expect_warning"), [(283, False), (284, True)])
    def test_recommended_length_boundary_is_exclusive(
        self, last_line: int, expect_warning: bool
    ) -> None:
        verse = make_section(K.VERSE)
        plan = normalize_plan(make_plan(verse))
        # "[Verse]\n" (8) + 9 * 300 + 283 + 9 newlines == 3000 exactly
        lines = ["x" * 300] * 9 + ["y" * last_line]
        lyrics = normalize_lyrics(make_lyrics((verse.id, lines)), plan)
        export = _render(plan, lyrics)
        assert export.fields["lyrics"].report.length == 3000 + (last_line - 283)
        assert ("lyrics_above_recommended_length" in _warning_codes(export)) is expect_warning


class TestLimitsNeverFail:
    def test_over_limit_text_is_returned_in_full(self) -> None:
        tags = tuple(f"tag number {i:04d} padding padding" for i in range(40))
        verse = make_section(K.VERSE)
        plan = normalize_plan(make_plan(verse, tags=tags, avoid=tags[:0] or ("a",)))
        artist = ArtistRenderInput(
            style_tags=tuple(f"artist tag {i:04d} filler filler filler" for i in range(40)),
            global_avoid=tuple(f"avoid tag {i:04d} filler filler filler" for i in range(40)),
        )
        lyrics = normalize_lyrics(make_lyrics((verse.id, ["z" * 300] * 64)), plan)
        export = _render(plan, lyrics, artist)

        for name, limit in (
            ("style", STYLE_LIMIT),
            ("exclude", EXCLUDE_LIMIT),
            ("lyrics", LYRICS_LIMIT),
        ):
            field = export.fields[name]
            assert field.report.limit == limit
            assert field.report.length == len(field.text)
        assert export.fields["style"].report.within_limit is (
            len(_text(export, "style")) <= STYLE_LIMIT
        )
        assert export.fields["lyrics"].report.length > LYRICS_LIMIT
        assert export.fields["lyrics"].report.within_limit is False
        assert _text(export, "lyrics").count("z" * 300) == 64  # nothing truncated

    def test_within_limit_boundaries(self) -> None:
        plan = normalize_plan(make_plan(tags=("x" * 60,) * 1))
        report = _render(plan).fields["style"].report
        assert report.within_limit is True
        assert report.limit == STYLE_LIMIT


class TestProtocol:
    def test_registry_and_renderer_identity(self) -> None:
        assert set(RENDERERS) == {StudioExportTarget.SUNO}
        assert SUNO.target is StudioExportTarget.SUNO
        assert SUNO.version == VERSION

    def test_results_are_immutable(self) -> None:
        export = _render(normalize_plan(make_plan()))
        with pytest.raises(dataclasses.FrozenInstanceError):
            export.renderer_version = "x"  # type: ignore[misc]
