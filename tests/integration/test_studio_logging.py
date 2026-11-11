"""Invariant 18: logs carry ids, kinds, counts and codes — never names, tags, lyrics, ideas or concepts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from structlog.testing import capture_logs

from src.api.schemas.studio import ArtistCreate, ArtistPatch, ProjectProfilePut, TrackPatch
from src.core.enums import StudioReleaseKind
from src.core.enums import StudioSectionKind as K
from src.studio.documents import LyricLine
from src.studio.errors import StudioDocumentInvalid
from tests.studio_support import make_lyrics, make_plan, make_section

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tests.studio_world import StudioWorld

MARK = "ZXQ-SECRET"


def _assert_clean(logs: Sequence[Mapping[str, object]]) -> None:
    assert logs, "the scenario should have logged something"
    for entry in logs:
        assert MARK.lower() not in repr(entry).lower(), entry


async def test_a_full_journey_logs_no_user_content(enabled_studio: StudioWorld) -> None:
    w = enabled_studio
    verse, chorus = (
        make_section(
            K.VERSE, label=f"{MARK}-label", notes=f"{MARK}-notes", tags=(f"{MARK}-local",)
        ),
        make_section(K.CHORUS),
    )
    with capture_logs() as logs:
        artist = await w.artists.create(
            w.user_id,
            w.product_id,
            ArtistCreate(
                name=f"{MARK}-artist",
                persona=f"{MARK}-persona",
                voice_notes=f"{MARK}-voice",
                aesthetic_references=[f"{MARK}-aesthetic"],
                style_tags=[f"{MARK}-style"],
                global_avoid=[f"{MARK}-avoid"],
            ),
        )
        await w.artists.patch(
            artist.id, w.user_id, w.product_id, ArtistPatch(name=f"{MARK}-renamed")
        )
        await w.projects.upsert_profile(
            w.project_id,
            w.user_id,
            w.product_id,
            ProjectProfilePut(
                release_kind=StudioReleaseKind.EP, artist_id=artist.id, concept=f"{MARK}-concept"
            ),
        )
        track = await w.new_track(f"{MARK}-track")
        await w.tracks.patch(
            track, w.user_id, w.product_id, TrackPatch(title=f"{MARK}-renamed-track")
        )
        plan_doc = make_plan(
            verse,
            chorus,
            title=f"{MARK}-title",
            idea=f"{MARK}-idea",
            references=(f"{MARK}-reference",),
            tags=(f"{MARK}-global",),
            avoid=(f"{MARK}-global-avoid",),
        )
        plan, _ = await w.plan(track, plan_doc)
        await w.plan(track, plan_doc)  # replay
        lyrics = make_lyrics(
            (
                verse.id,
                [f"{MARK} line one", LyricLine(text=f"{MARK} line two", delivery=(f"{MARK}-d",))],
            )
        )
        lyr, _ = await w.lyrics(track, lyrics, plan.version.id)
        await w.versions.set_current(
            track, w.user_id, w.product_id, plan_version_id=plan.version.id, lyrics_version_id=None
        )
        await w.exports.render_suno(
            track, w.user_id, w.product_id, lyrics_version_id=lyr.version.id
        )
        other, _ = await w.plan(
            track, make_plan(verse, chorus, title=f"{MARK}-other"), make_current=False
        )
        await w.versions.delete(other.version.id, w.user_id, w.product_id)
        await w.tracks.reorder(w.project_id, w.user_id, w.product_id, [track])
        await w.tracks.delete(track, w.user_id, w.product_id)
        await w.projects.disable(w.project_id, w.user_id, w.product_id)
    _assert_clean(logs)
    events = {entry["event"] for entry in logs}
    assert {
        "studio.artist_created",
        "studio.project_updated",  # the fixture already enabled the project
        "studio.track_created",
        "studio.version_created",
        "studio.version_replayed",
        "studio.current_set",
        "studio.export_rendered",
        "studio.version_deleted",
        "studio.track_reordered",
        "studio.track_deleted",
        "studio.project_disabled",
    } <= events


async def test_validation_failures_log_nothing_from_the_document(
    enabled_studio: StudioWorld,
) -> None:
    w = enabled_studio
    track = await w.new_track()
    with capture_logs() as logs, pytest.raises(StudioDocumentInvalid):
        await w.plan(track, make_plan(title=f"{MARK}", tags=(f"{MARK}-tag", f"{MARK.lower()}-tag")))
    for entry in logs:
        assert MARK.lower() not in repr(entry).lower(), entry
