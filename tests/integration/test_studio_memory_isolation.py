"""What memories must never touch: the Suno export (invariant 7) and what they leave with (16).

Memories are planner-only (P2-M6): the renderer sees ``RenderablePlan`` and ``ArtistRenderInput``,
which have no memory field, so an artist's memories cannot change a byte of an export. They belong
to their artist: deleting the artist - or the user - deletes them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
from sqlalchemy import delete, func, select

from src.api.schemas.studio import ArtistCreate, MemoryCreate
from src.core.enums import StudioSectionKind as K
from src.db.models.studio import StudioArtistMemory
from src.db.models.user import User
from src.studio.services.memory import MemoryService
from tests.studio_support import make_lyrics, make_plan, make_section

if TYPE_CHECKING:
    from uuid import UUID

    from tests.studio_world import StudioWorld


async def _memories_of(w: StudioWorld, artist_id: UUID) -> int:
    return (
        await w.session.execute(
            select(func.count())
            .select_from(StudioArtistMemory)
            .where(StudioArtistMemory.artist_id == artist_id)
        )
    ).scalar_one()


async def test_the_suno_export_is_byte_identical_with_and_without_memories(
    studio: StudioWorld,
) -> None:
    w = studio
    artist = await w.artists.create(
        w.user_id,
        w.product_id,
        ArtistCreate(name="Nova", style_tags=["dreamy"], global_avoid=["autotune"]),
    )
    await w.enable(artist_id=artist.id)
    verse, chorus = make_section(K.VERSE), make_section(K.CHORUS, tags=["anthemic"])
    track = await w.new_track("Night Drive")
    plan, _ = await w.plan(track, make_plan(verse, chorus, title="Neon Rain", tempo_bpm=100))
    await w.lyrics(
        track,
        make_lyrics((verse.id, ["la la la la"]), (chorus.id, ["na na na na"])),
        plan.version.id,
    )

    before = await w.exports.render_suno(track, w.user_id, w.product_id)

    memories = MemoryService(w.session)
    for n in range(5):
        await memories.create(
            artist.id,
            w.user_id,
            w.product_id,
            MemoryCreate(
                title=f"memory {n}",
                body="STYLE: neon / EXCLUDE: dreamy " * 20,
                themes=["dreamy", "autotune"],
                imagery=["Neon Rain"],
                salience=5,
            ),
        )
    assert await _memories_of(w, artist.id) == 5

    after = await w.exports.render_suno(track, w.user_id, w.product_id)
    assert msgspec.json.encode(after) == msgspec.json.encode(before)
    assert "memory" not in msgspec.json.encode(after).decode()


async def test_deleting_an_artist_deletes_its_memories_and_only_its_own(
    studio: StudioWorld,
) -> None:
    w = studio
    memories = MemoryService(w.session)
    first = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="First"))
    second = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Second"))
    for artist in (first, second):
        for n in range(3):
            await memories.create(
                artist.id, w.user_id, w.product_id, MemoryCreate(title=f"m{n}", body="b")
            )

    await w.artists.delete(first.id, w.user_id, w.product_id)

    assert await _memories_of(w, first.id) == 0
    assert await _memories_of(w, second.id) == 3


async def test_deleting_the_user_deletes_the_artists_and_through_them_the_memories(
    studio: StudioWorld,
) -> None:
    w = studio
    artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
    await MemoryService(w.session).create(
        artist.id, w.user_id, w.product_id, MemoryCreate(title="m", body="b")
    )
    assert await _memories_of(w, artist.id) == 1

    await w.session.execute(delete(User).where(User.id == w.user_id))

    assert await _memories_of(w, artist.id) == 0
