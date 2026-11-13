"""The pure half of the import: section ids, carry-over (I3/I3a), artist tags, error mapping."""

from __future__ import annotations

from uuid import uuid5

import msgspec
import pytest

from src.core.enums import StudioSectionKind as K
from src.studio.documents import LyricLine, SyllableRange
from src.studio.errors import ImportInvalid
from src.studio.renderers.base import ArtistRenderInput
from src.studio.section_names import rendered_labels
from tests.studio_import import TRACK_ID, import_text, reimport, render_song, texts
from tests.studio_support import (
    GOLDEN_EXCLUDE,
    GOLDEN_LYRICS,
    GOLDEN_STYLE,
    golden_song,
    make_lyrics,
    make_plan,
    make_section,
)

pytestmark = pytest.mark.unit


def codes(built) -> list[tuple[str, int | None]]:
    return [(w.code, w.line) for w in built.warnings]


class TestGoldenSongRoundTrip:
    def test_the_golden_text_is_what_the_fixture_renders(self) -> None:
        song = golden_song()
        shown = texts(render_song(song.plan, song.lyrics, song.artist))
        assert (shown["style"], shown["exclude"], shown["lyrics"]) == (
            GOLDEN_STYLE,
            GOLDEN_EXCLUDE,
            GOLDEN_LYRICS,
        )

    def test_over_the_song_itself_the_import_is_a_no_op(self) -> None:
        """Non-exportable fields, section kinds, delivery/tag splits: all carried over."""
        song = golden_song()
        plan, lyrics, built = reimport(song.plan, song.lyrics, song.artist, parent=True)
        assert plan == song.plan
        assert lyrics == song.lyrics
        # the only warning is the parser's own: "Break It Down" is not a known section name
        assert [w.code for w in built.warnings] == ["unknown_section_name"]

    def test_with_no_parent_the_text_still_renders_back_identically(self) -> None:
        song = golden_song()
        plan, lyrics, _ = reimport(song.plan, song.lyrics, song.artist, parent=False)
        assert texts(render_song(plan, lyrics, song.artist)) == texts(
            render_song(song.plan, song.lyrics, song.artist)
        )

    def test_without_a_parent_what_a_header_cannot_say_is_lost(self) -> None:
        song = golden_song()
        plan, _, _ = reimport(song.plan, song.lyrics, song.artist, parent=False)
        assert plan.idea is None
        assert plan.references == ()
        assert all(s.notes is None and s.duration_seconds is None for s in plan.sections)
        assert all(s.style.avoid == () for s in plan.sections)
        # the delivery/tag split is gone too: everything in a header is a style tag
        assert plan.sections[1].delivery == ()
        assert plan.sections[1].style.tags == ("breathy", "minimal")


class TestSectionIds:
    def test_ids_survive_an_edited_line(self) -> None:
        song = golden_song()
        edited = GOLDEN_LYRICS.replace("Chorus one", "Chorus uno")
        plan, lyrics, _ = import_text(
            style=GOLDEN_STYLE,
            exclude=GOLDEN_EXCLUDE,
            lyrics=edited,
            title=song.plan.title,
            parent_plan=song.plan,
            parent_lyrics=song.lyrics,
            artist=song.artist,
        )
        assert plan == song.plan
        assert lyrics is not None
        assert lyrics != song.lyrics
        assert [s.section_id for s in lyrics.sections] == [
            s.section_id for s in song.lyrics.sections
        ]

    def test_a_section_takes_the_first_unused_parent_section_with_that_label(self) -> None:
        song = golden_song()
        plan, _, _ = import_text(
            style=GOLDEN_STYLE,
            exclude=GOLDEN_EXCLUDE,
            lyrics=GOLDEN_LYRICS,
            parent_plan=song.plan,
            artist=song.artist,
        )
        # two "[Chorus]" blocks map to the two chorus sections in order, not to the first twice
        chorus_ids = [s.id for s in song.plan.sections if s.kind is K.CHORUS]
        assert [s.id for s in plan.sections if s.kind is K.CHORUS] == chorus_ids
        assert len({s.id for s in plan.sections}) == len(plan.sections)

    def test_an_unmatched_section_gets_a_deterministic_uuid5(self) -> None:
        plan, _, _ = import_text(lyrics="[Verse]\na\n\n[Chorus]\nb\n\n[Chorus]\nc")
        assert [s.id for s in plan.sections] == [
            uuid5(TRACK_ID, "suno-import:0:Verse"),
            uuid5(TRACK_ID, "suno-import:1:Chorus"),
            uuid5(TRACK_ID, "suno-import:2:Chorus"),
        ]

    def test_the_same_text_gives_the_same_ids_even_for_another_run(self) -> None:
        text = "[Verse]\na\n\n[Bridge]\nb"
        assert import_text(lyrics=text)[0] == import_text(lyrics=text)[0]

    def test_the_id_depends_on_the_track(self) -> None:
        other = uuid5(TRACK_ID, "another track")
        a = import_text(lyrics="[Verse]\na")[0].sections[0].id
        b = import_text(lyrics="[Verse]\na", track_id=other)[0].sections[0].id
        assert a != b

    def test_an_implicit_section_gets_an_id_too(self) -> None:
        plan, _, _ = import_text(lyrics="just words")
        assert plan.sections[0].id == uuid5(TRACK_ID, "suno-import:0:")

    def test_a_renamed_header_is_a_new_section(self) -> None:
        song = golden_song()
        renamed = GOLDEN_LYRICS.replace("[Pre-Chorus]", "[Lift]")
        plan, _, _ = import_text(
            style=GOLDEN_STYLE,
            exclude=GOLDEN_EXCLUDE,
            lyrics=renamed,
            parent_plan=song.plan,
            artist=song.artist,
        )
        old_pre = next(s for s in song.plan.sections if s.kind is K.PRE_CHORUS)
        assert old_pre.id not in {s.id for s in plan.sections}
        assert len(plan.sections) == len(song.plan.sections)

    def test_a_generated_id_never_collides_with_a_kept_one(self) -> None:
        # Parent section X owns exactly the uuid5 the *new* "Verse" at position 0 would get; the
        # label "Opening" matches X, so "Verse" must not reuse X's id.
        x = make_section(
            K.BRIDGE, label="Opening", section_id=uuid5(TRACK_ID, "suno-import:0:Verse")
        )
        parent = make_plan(x)
        plan, _, _ = import_text(lyrics="[Verse]\n\n[Opening]", parent_plan=parent)
        ids = [s.id for s in plan.sections]
        assert len(set(ids)) == 2
        assert ids[1] == x.id


class TestCarryOver:
    def test_a_matched_section_with_a_changed_header_keeps_the_non_exportable_fields(self) -> None:
        verse = make_section(
            K.VERSE,
            delivery=("breathy",),
            tags=("minimal",),
            avoid=("noise",),
            duration_seconds=33,
            notes="keep me",
        )
        parent = make_plan(verse, make_section(K.CHORUS))
        plan, _, _ = import_text(lyrics="[Verse: loud, big]\n\n[Chorus]", parent_plan=parent)
        section = plan.sections[0]
        assert section.id == verse.id
        assert (section.duration_seconds, section.notes) == (33, "keep me")
        assert section.style.avoid == ("noise",)
        # ... but the exportable part is the new text, all as style tags
        assert (section.delivery, section.style.tags) == ((), ("loud", "big"))

    def test_a_matched_section_with_an_unchanged_header_is_copied_verbatim(self) -> None:
        verse = make_section(
            K.BRIDGE,
            label="Verse",
            delivery=("breathy",),
            tags=("minimal",),
            avoid=("n",),
            duration_seconds=5,
            notes="n",
            instrumental=True,
        )
        parent = make_plan(verse)
        header = "[Verse: instrumental, breathy, minimal]"
        plan, _, _ = import_text(lyrics=header, parent_plan=parent)
        assert plan.sections[0] == verse  # kind bridge, the delivery/tag split, instrumental

    def test_idea_and_references_are_always_carried(self) -> None:
        parent = make_plan(idea="the idea", references=("ref one",))
        plan, _, _ = import_text(lyrics="[Verse]", parent_plan=parent)
        assert (plan.idea, plan.references) == ("the idea", ("ref one",))

    def test_the_tempo_and_title_come_from_the_text_and_the_request(self) -> None:
        parent = make_plan(tempo_bpm=100, title="Old")
        plan, _, _ = import_text(style="90 bpm", title="New", parent_plan=parent)
        assert (plan.tempo_bpm, plan.title) == (90, "New")
        plan, _, _ = import_text(style="", parent_plan=parent)
        assert plan.tempo_bpm is None

    def test_line_syllables_are_carried_from_the_current_lyrics(self) -> None:
        verse = make_section(K.VERSE)
        parent_lyrics = msgspec.structs.replace(
            make_lyrics((verse.id, ["old"])), line_syllables=SyllableRange(min=3, max=9)
        )
        _, lyrics, _ = import_text(
            lyrics="[Verse]\nnew", parent_plan=make_plan(verse), parent_lyrics=parent_lyrics
        )
        assert lyrics is not None
        assert lyrics.line_syllables == SyllableRange(min=3, max=9)

    def test_a_kept_label_is_pinned_when_new_neighbours_change_the_numbering(self) -> None:
        # The parent's only verse rendered as "Verse" (label None). Pasting a second "[Verse]"
        # makes the default labels "Verse 1" / "Verse 2"; the kept section must still read "Verse".
        verse = make_section(K.VERSE)
        plan, _, _ = import_text(lyrics="[Verse]\na\n\n[Verse]\nb", parent_plan=make_plan(verse))
        assert plan.sections[0].id == verse.id
        assert rendered_labels((s.kind, s.label) for s in plan.sections) == ["Verse", "Verse"]


class TestLabelsAndKinds:
    def test_a_label_is_none_exactly_when_it_equals_the_default(self) -> None:
        plan, _, _ = import_text(lyrics="[Verse 1]\n\n[Verse 2]\n\n[Verse 3]")
        assert [s.label for s in plan.sections] == [None, None, None]
        # the same names with a Chorus between: the third verse is "Verse 2" by default
        plan, _, _ = import_text(lyrics="[Verse 1]\n\n[Chorus]\n\n[Verse 3]")
        assert [s.label for s in plan.sections] == [None, None, "Verse 3"]

    def test_the_comparison_is_exact_and_case_sensitive(self) -> None:
        plan, _, _ = import_text(lyrics="[verse]\n\n[Chorus]")
        assert [s.label for s in plan.sections] == ["verse", None]

    def test_numbered_defaults(self) -> None:
        plan, _, _ = import_text(lyrics="[Verse 1]\n\n[Chorus]\n\n[Verse 2]")
        assert [s.label for s in plan.sections] == [None, None, None]

    def test_a_number_the_default_would_not_give_is_kept_as_a_label(self) -> None:
        plan, _, _ = import_text(lyrics="[Chorus 2]\n\n[Verse 2]")
        assert [s.label for s in plan.sections] == ["Chorus 2", "Verse 2"]

    def test_unknown_names_keep_their_label_and_warn(self) -> None:
        plan, _, built = import_text(lyrics="[Guitar Duel]\n\n[Verse]")
        assert (plan.sections[0].kind, plan.sections[0].label) == (K.INTERLUDE, "Guitar Duel")
        assert codes(built) == [("unknown_section_name", 1), ("no_lyric_lines", None)]

    def test_the_flag_is_set_by_the_cue_but_not_by_the_kind(self) -> None:
        plan, _, _ = import_text(lyrics="[Drop: instrumental]\n\n[Instrumental]")
        assert [s.instrumental for s in plan.sections] == [True, False]


class TestArtistTags:
    ARTIST = ArtistRenderInput(style_tags=("dreamy", "warm"), global_avoid=("autotune",))

    def test_the_artists_tags_are_not_copied_into_the_plan(self) -> None:
        plan, _, _ = import_text(
            style="synth, dreamy, warm, 100 BPM", exclude="noise, autotune", artist=self.ARTIST
        )
        assert plan.global_style.tags == ("synth",)
        assert plan.global_style.avoid == ("noise",)

    def test_without_an_artist_every_token_is_the_plans(self) -> None:
        plan, _, _ = import_text(style="synth, dreamy", exclude="noise")
        assert plan.global_style.tags == ("synth", "dreamy")
        assert plan.global_style.avoid == ("noise",)

    def test_a_plan_tag_the_artist_also_has_stays_in_the_unchanged_plan(self) -> None:
        # Rendering is not injective: plan [warm] + artist [dreamy, warm] writes "dreamy, warm"
        # exactly like plan [] does. The current plan's own tags win when they explain the text.
        parent = make_plan(tags=("dreamy",))
        plan, _, _ = import_text(style="dreamy, warm", artist=self.ARTIST, parent_plan=parent)
        assert plan.global_style.tags == ("dreamy",)

    def test_a_changed_text_falls_back_to_the_pure_inverse(self) -> None:
        parent = make_plan(tags=("dreamy",))
        plan, _, _ = import_text(style="lofi, dreamy, warm", artist=self.ARTIST, parent_plan=parent)
        assert plan.global_style.tags == ("lofi",)


class TestLyricsDocument:
    def test_headers_only_gives_no_lyrics_and_a_warning(self) -> None:
        plan, lyrics, built = import_text(lyrics="[Verse]\n\n[Chorus]")
        assert len(plan.sections) == 2
        assert lyrics is None
        assert codes(built) == [("no_lyric_lines", None)]

    def test_headers_only_on_an_instrumental_has_no_warning(self) -> None:
        plan, lyrics, built = import_text(lyrics="[Drop]\n\n[Outro]", instrumental=True)
        assert (plan.instrumental, lyrics, built.warnings) == (True, None, ())

    def test_an_empty_lyrics_box_is_a_plan_without_sections(self) -> None:
        plan, lyrics, built = import_text(style="lofi", lyrics="")
        assert (plan.sections, lyrics, built.warnings) == ((), None, ())

    def test_instrumental_is_never_inferred(self) -> None:
        plan, _, _ = import_text(lyrics="[Verse]\n\n[Chorus]")
        assert plan.instrumental is False

    def test_lines_on_an_instrumental_track_are_refused_with_every_line(self) -> None:
        with pytest.raises(ImportInvalid) as exc:
            import_text(lyrics="[Verse]\nsing\nmore\n\n[Chorus]\nagain", instrumental=True)
        assert [(i.path, i.code) for i in exc.value.issues] == [
            ("lyrics", "instrumental_with_lyric_lines")
        ]
        assert exc.value.detail is not None
        assert exc.value.detail["source_lines"] == {"lyrics": [2, 3, 6]}

    def test_lines_need_a_language(self) -> None:
        with pytest.raises(ImportInvalid) as exc:
            import_text(lyrics="[Verse]\nsing", language=None)
        assert [(i.path, i.code) for i in exc.value.issues] == [("language", "language_required")]
        import_text(lyrics="[Verse]", language=None)  # no lines: no language needed

    def test_lyrics_sections_follow_the_plan_and_skip_sections_without_lines(self) -> None:
        plan, lyrics, _ = import_text(lyrics="[Verse]\na\n\n[Chorus]\n\n[Outro]\nz")
        assert lyrics is not None
        assert [s.section_id for s in lyrics.sections] == [plan.sections[0].id, plan.sections[2].id]

    def test_delivery_lands_on_the_line(self) -> None:
        _, lyrics, _ = import_text(lyrics="[Verse]\n[soft, close]\na")
        assert lyrics is not None
        assert lyrics.sections[0].lines == (LyricLine(text="a", delivery=("soft", "close")),)


class TestErrorsPointAtTheirLines:
    def test_a_bad_label_is_addressed_to_the_header_line(self) -> None:
        with pytest.raises(ImportInvalid) as exc:
            import_text(lyrics="[Verse]\na\n\n[A [B]\nb")
        assert [(i.path, i.code) for i in exc.value.issues] == [
            ("plan.sections[1].label", "reserved_characters")
        ]
        assert exc.value.source_lines == {"plan.sections[1].label": 4}

    def test_a_pair_issue_is_addressed_to_its_section_header(self) -> None:
        with pytest.raises(ImportInvalid) as exc:
            import_text(lyrics="[Verse]\na\n\n[Instrumental]\nsung over a solo")
        assert [(i.path, i.code) for i in exc.value.issues] == [
            ("lyrics.sections[1]", "instrumental_section_has_lines")
        ]
        assert exc.value.source_lines == {"lyrics.sections[1]": 4}

    def test_style_issues_have_a_path_and_no_line(self) -> None:
        with pytest.raises(ImportInvalid) as exc:
            import_text(style="Lofi", exclude="lofi", lyrics="[Verse]")
        assert [(i.path, i.code) for i in exc.value.issues] == [
            ("plan.global_style.avoid[0]", "tag_conflict")
        ]
        assert exc.value.source_lines == {}

    def test_too_many_tags_is_an_issue_not_a_crash(self) -> None:
        with pytest.raises(ImportInvalid) as exc:
            import_text(style=", ".join(f"tag{i}" for i in range(41)))
        assert [(i.path, i.code) for i in exc.value.issues] == [
            ("plan.global_style.tags", "too_many_items")
        ]

    def test_too_many_sections_and_lines(self) -> None:
        with pytest.raises(ImportInvalid) as exc:
            import_text(lyrics="\n\n".join(f"[Verse {i}]" for i in range(65)))
        assert [(i.path, i.code) for i in exc.value.issues] == [("plan.sections", "too_many_items")]
        lines = "\n".join(f"line {i}" for i in range(65))
        with pytest.raises(ImportInvalid) as exc:
            import_text(lyrics=f"[Verse]\n{lines}")
        assert [(i.path, i.code) for i in exc.value.issues] == [
            ("lyrics.sections[0].lines", "too_many_items")
        ]
        assert exc.value.source_lines == {"lyrics.sections[0].lines": 1}

    def test_an_over_long_line_is_addressed_to_the_line(self) -> None:
        with pytest.raises(ImportInvalid) as exc:
            import_text(lyrics="[Verse]\nok\n" + "x" * 301)
        assert [(i.path, i.code) for i in exc.value.issues] == [
            ("lyrics.sections[0].lines[1].text", "too_long_after_normalization")
        ]
        assert exc.value.source_lines == {"lyrics.sections[0].lines[1].text": 3}

    def test_the_error_never_contains_pasted_text(self) -> None:
        with pytest.raises(ImportInvalid) as exc:
            import_text(lyrics="[Verse]\nSECRET-LINE\n\n[A [SECRET-B]\nmore")
        assert "SECRET" not in repr(exc.value.detail)
        assert "SECRET" not in str(exc.value)
