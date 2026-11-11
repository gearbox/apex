"""Suno export through the service (invariants 16, 23) against real rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

from src.api.schemas.studio import ArtistCreate
from src.core.enums import StudioExportTarget
from src.core.enums import StudioSectionKind as K
from src.studio.documents import LyricLine
from src.studio.errors import LyricsIncompatibleWithPlan, PlanRequired, StudioNotFound
from tests.studio_support import make_lyrics, make_plan, make_section

if TYPE_CHECKING:
    from tests.studio_world import StudioWorld


async def _song(w: StudioWorld, **plan_kwargs: Any):
    verse, chorus = make_section(K.VERSE, delivery=("soft",)), make_section(K.CHORUS)
    track = await w.new_track()
    plan, _ = await w.plan(track, make_plan(verse, chorus, title="Exported", **plan_kwargs))
    lyr, _ = await w.lyrics(
        track,
        make_lyrics(
            (verse.id, ["one", LyricLine(text="two", delivery=("whisper",))]), (chorus.id, ["hook"])
        ),
        plan.version.id,
    )
    return track, plan.version.id, lyr.version.id, verse, chorus


async def test_renders_the_current_pair(enabled_studio: StudioWorld) -> None:
    w = enabled_studio
    track, plan_id, lyrics_id, *_ = await _song(
        w, tags=("synthwave",), avoid=("metal",), tempo_bpm=110
    )
    out = await w.exports.render_suno(track, w.user_id, w.product_id)
    assert out.target is StudioExportTarget.SUNO
    assert (out.plan_version_id, out.lyrics_version_id) == (plan_id, lyrics_id)
    assert out.title == "Exported"
    assert out.style == "synthwave, 110 BPM"
    assert out.exclude == "metal"
    assert out.lyrics == "[Verse: soft]\none\n[whisper]\ntwo\n\n[Chorus]\nhook"
    assert {k: (v.limit, v.within_limit) for k, v in out.fields.items()} == {
        "title": (None, True),
        "style": (1000, True),
        "exclude": (1000, True),
        "lyrics": (5000, True),
    }
    assert out.fields["lyrics"].length == len(out.lyrics)
    assert out.warnings == []


async def test_render_is_a_pure_read_and_repeatable(enabled_studio: StudioWorld) -> None:
    w = enabled_studio
    track, *_ = await _song(w, tags=("a",))
    before = (await w.track_row(track)).updated_at
    first = await w.exports.render_suno(track, w.user_id, w.product_id)
    second = await w.exports.render_suno(track, w.user_id, w.product_id)
    assert first == second
    assert (await w.track_row(track)).updated_at == before


async def test_artist_style_is_merged_after_the_plans_tags_and_deduped(
    enabled_studio: StudioWorld,
) -> None:
    w = enabled_studio
    artist = await w.artists.create(
        w.user_id,
        w.product_id,
        ArtistCreate(
            name="Nova",
            style_tags=["husky vocals", "Synthwave"],
            global_avoid=["Autotune", "metal"],
            persona="MARKER-PERSONA",
            voice_notes="MARKER-VOICE",
            aesthetic_references=["MARKER-AESTHETIC"],
        ),
    )
    await w.enable(artist_id=artist.id)
    track, *_ = await _song(
        w,
        tags=("synthwave", "dreamy"),
        avoid=("metal",),
        tempo_bpm=100,
        idea="MARKER-IDEA",
        references=("MARKER-REF",),
    )
    out = await w.exports.render_suno(track, w.user_id, w.product_id)
    assert out.style == "synthwave, dreamy, husky vocals, 100 BPM"
    assert out.exclude == "metal, Autotune"
    everything = f"{out.title}\n{out.style}\n{out.exclude}\n{out.lyrics}"
    assert "MARKER" not in everything


async def test_a_project_without_an_artist_exports_the_plan_alone(
    enabled_studio: StudioWorld,
) -> None:
    w = enabled_studio
    track, *_ = await _song(w, tags=("only",))
    assert (await w.exports.render_suno(track, w.user_id, w.product_id)).style == "only"


async def test_over_limit_fields_are_rendered_in_full_and_flagged(
    enabled_studio: StudioWorld,
) -> None:
    w = enabled_studio
    verse = make_section(K.VERSE)
    track = await w.new_track()
    tags = tuple(f"descriptive tag number {i:03d} with padding" for i in range(30))
    plan, _ = await w.plan(track, make_plan(verse, title="Long", tags=tags))
    await w.lyrics(track, make_lyrics((verse.id, ["z" * 300] * 64)), plan.version.id)
    out = await w.exports.render_suno(track, w.user_id, w.product_id)
    assert out.fields["style"].within_limit is False
    assert out.fields["lyrics"].within_limit is False
    assert out.fields["lyrics"].length == len(out.lyrics) > 5000
    assert out.lyrics.count("z" * 300) == 64
    assert [w.code for w in out.warnings] == ["lyrics_above_recommended_length"]


async def test_explicit_versions_override_the_current_pair(enabled_studio: StudioWorld) -> None:
    w = enabled_studio
    track, _, lyrics_id, verse, chorus = await _song(w)
    other, _ = await w.plan(track, make_plan(verse, chorus, title="Other"), make_current=False)
    out = await w.exports.render_suno(
        track,
        w.user_id,
        w.product_id,
        plan_version_id=other.version.id,
        lyrics_version_id=lyrics_id,
    )
    assert (out.title, out.plan_version_id) == ("Other", other.version.id)
    assert out.lyrics_version_id == lyrics_id
    assert "hook" in out.lyrics


async def test_lyrics_must_fit_the_chosen_plan(enabled_studio: StudioWorld) -> None:
    w = enabled_studio
    track, _, lyrics_id, verse, _ = await _song(w)
    versed, _ = await w.plan(track, make_plan(verse, title="NoChorus"), make_current=False)
    with pytest.raises(LyricsIncompatibleWithPlan):
        await w.exports.render_suno(
            track,
            w.user_id,
            w.product_id,
            plan_version_id=versed.version.id,
            lyrics_version_id=lyrics_id,
        )
    with pytest.raises(LyricsIncompatibleWithPlan):  # current lyrics are used by default
        await w.exports.render_suno(
            track, w.user_id, w.product_id, plan_version_id=versed.version.id
        )


async def test_a_plan_alone_renders_the_section_skeleton(enabled_studio: StudioWorld) -> None:
    w = enabled_studio
    track = await w.new_track()
    await w.plan(track, make_plan(make_section(K.INTRO), make_section(K.VERSE), title="Skeleton"))
    out = await w.exports.render_suno(track, w.user_id, w.product_id)
    assert out.lyrics_version_id is None
    assert out.lyrics == "[Intro]\n\n[Verse]"


async def test_plan_required(enabled_studio: StudioWorld) -> None:
    w = enabled_studio
    with pytest.raises(PlanRequired):
        await w.exports.render_suno(await w.new_track(), w.user_id, w.product_id)


async def test_wrong_kind_or_foreign_versions_are_not_found(enabled_studio: StudioWorld) -> None:
    w = enabled_studio
    track, plan_id, lyrics_id, *_ = await _song(w)
    other = await w.new_track("other")
    foreign, _ = await w.plan(other, make_plan(title="foreign"))
    for kwargs in (
        {"plan_version_id": lyrics_id},
        {"lyrics_version_id": plan_id},
        {"plan_version_id": foreign.version.id},
        {"lyrics_version_id": uuid4()},
    ):
        with pytest.raises(StudioNotFound):
            await w.exports.render_suno(track, w.user_id, w.product_id, **kwargs)
