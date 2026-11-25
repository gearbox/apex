"""Invariant 14: hard-deleting a user removes the whole studio graph without FK errors."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import delete, func, select, text

from src.core.enums import StudioSectionKind as K
from src.db.models.library import LibraryAssetHold, LibraryAudioProfile, LibraryProject
from src.db.models.storage import UserImage
from src.db.models.studio import (
    StudioArtist,
    StudioDocumentVersion,
    StudioProjectProfile,
    StudioTrack,
    StudioTrackAudio,
)
from src.db.models.user import User
from tests.studio_support import make_lyrics, make_plan, make_section
from tests.studio_world import StudioWorld
from tests.track_audio_support import attach_body

if TYPE_CHECKING:
    from tests.library_holds_support import HoldWorld

_TABLES = (StudioArtist, StudioProjectProfile, StudioTrack, StudioDocumentVersion, LibraryProject)


async def _counts(world: StudioWorld, user_id) -> dict[str, int]:
    return {
        model.__tablename__: (
            await world.session.execute(
                select(func.count()).select_from(model).where(model.user_id == user_id)  # type: ignore[attr-defined]
            )
        ).scalar_one()
        for model in _TABLES
    }


async def _populate(w: StudioWorld, name: str) -> None:
    from src.api.schemas.studio import ArtistCreate

    artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name=name))
    await w.enable(artist_id=artist.id)
    for n in range(2):
        track = await w.new_track(f"{name}-{n}")
        verse = make_section(K.VERSE)
        plan, _ = await w.plan(track, make_plan(verse, title="v1"))
        child, _ = await w.plan(track, make_plan(verse, title="v2"), parent=plan.version.id)
        await w.lyrics(track, make_lyrics((verse.id, ["la la"])), child.version.id)


async def test_hard_delete_removes_the_whole_graph_and_only_that_users(
    enabled_studio: StudioWorld, make_user, make_library_project
) -> None:
    w = enabled_studio
    await _populate(w, "alpha")

    bystander = await make_user(email=f"by-{uuid4().hex[:6]}@example.com")
    other = StudioWorld(w.session, bystander, await make_library_project(user=bystander))
    await _populate(other, "beta")

    before = await _counts(w, w.user_id)
    assert before == {
        "studio_artists": 1,
        "studio_project_profiles": 1,
        "studio_tracks": 2,
        "studio_document_versions": 6,
        "library_projects": 1,
    }

    await w.session.execute(
        delete(User).where(User.id == w.user_id)
    )  # the cascade, in one statement

    assert set((await _counts(w, w.user_id)).values()) == {0}
    assert (await _counts(other, bystander.id))["studio_document_versions"] == 6
    assert (await _counts(other, bystander.id))["studio_tracks"] == 2


async def test_cascade_works_when_only_part_of_the_graph_exists(
    studio: StudioWorld,
) -> None:
    w = studio
    from src.api.schemas.studio import ArtistCreate

    await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="solo"))  # artist only
    await w.session.execute(delete(User).where(User.id == w.user_id))
    assert set((await _counts(w, w.user_id)).values()) == {0}


async def test_one_statement_hard_delete_removes_a_track_with_attached_audio(
    hold_world: HoldWorld,
) -> None:
    """The user, upload/profile, track, audio version and hold are committed before deletion."""
    user = await hold_world.user()
    asset = await hold_world.audio(user)
    async with hold_world.factory() as session:
        project = LibraryProject(
            user_id=user.id,
            product_id=user.product_id,
            name=f"Hard delete {user.id}",
        )
        session.add(project)
        await session.flush()
        world = StudioWorld(session=session, user=user, library_project=project)
        await world.enable()
        track_id = await world.new_track("Has attached audio")
        await world.audio.create(
            track_id,
            user.id,
            user.product_id,
            attach_body(asset),
        )
        await session.commit()

    async with hold_world.factory() as session:
        await session.execute(
            text("DELETE FROM users WHERE id = :id"),
            {"id": user.id},
        )
        await session.commit()

    async with hold_world.factory() as session:
        assert (
            await session.execute(
                select(func.count())
                .select_from(StudioTrackAudio)
                .where(StudioTrackAudio.user_id == user.id)
            )
        ).scalar_one() == 0
        assert (
            await session.execute(
                select(func.count())
                .select_from(LibraryAssetHold)
                .where(LibraryAssetHold.user_id == user.id)
            )
        ).scalar_one() == 0
        assert (
            await session.execute(
                select(func.count())
                .select_from(LibraryAudioProfile)
                .join(UserImage, UserImage.id == LibraryAudioProfile.upload_id)
                .where(UserImage.user_id == user.id)
            )
        ).scalar_one() == 0
