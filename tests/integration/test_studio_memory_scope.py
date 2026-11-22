"""A memory carries its own ``user_id`` / ``product_id`` and every query filters them (R1).

Two guarantees, both below the service layer:

* **The database** refuses a memory whose owner or product is not its artist's (the composite FK
  ``fk_studio_artist_memories_artist_owner``), so the row can never contradict the artist.
* **The repository** filters the memory's own scope in every method, by-id and by-artist alike: a
  caller that holds an ``artist_id`` it did not resolve through its own scope - a future chat
  assembler, a video director, an admin tool - reads nothing of another tenant's, and changes
  nothing.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from src.api.schemas.studio import ArtistCreate, MemoryCreate
from src.core.enums import StudioMemoryOrigin, StudioMemoryStatus
from src.db.models.studio import StudioArtistMemory
from src.db.repositories.studio import StudioArtistMemoryRepository
from src.studio.services.memory import MemoryService

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any
    from uuid import UUID

    from src.db.models.user import User
    from tests.studio_world import StudioWorld

    UserFactory = Callable[..., Coroutine[Any, Any, User]]

OWNER_FK = "fk_studio_artist_memories_artist_owner"

_INSERT = text(
    "INSERT INTO studio_artist_memories (id, user_id, product_id, artist_id, title, body) "
    "VALUES (:id, :user, :product, :artist, 't', 'b')"
)


async def test_a_memory_whose_owner_differs_from_its_artists_is_refused_by_the_database(
    studio: StudioWorld, make_user: UserFactory
) -> None:
    w = studio
    artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
    other_user = await make_user(email=f"other-{uuid4().hex[:8]}@example.com")

    # the artist's own owner pair is accepted: the control for the two refusals below
    async with w.session.begin_nested():
        await w.session.execute(
            _INSERT,
            {"id": uuid4(), "user": w.user_id, "product": w.product_id, "artist": artist.id},
        )

    # another user (who exists, so only the composite FK can object) ...
    with pytest.raises(IntegrityError, match=OWNER_FK):
        async with w.session.begin_nested():
            await w.session.execute(
                _INSERT,
                {
                    "id": uuid4(),
                    "user": other_user.id,
                    "product": w.product_id,
                    "artist": artist.id,
                },
            )
    # ... and, separately, another product for the right user
    with pytest.raises(IntegrityError, match=OWNER_FK):
        async with w.session.begin_nested():
            await w.session.execute(
                _INSERT,
                {"id": uuid4(), "user": w.user_id, "product": "synthara", "artist": artist.id},
            )


async def test_the_owner_pair_cannot_be_changed_away_from_the_artists(
    studio: StudioWorld, make_user: UserFactory
) -> None:
    """The FK also guards an UPDATE: a memory cannot be handed to another tenant."""
    w = studio
    artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
    memory = await MemoryService(w.session).create(
        artist.id, w.user_id, w.product_id, MemoryCreate(title="m", body="b")
    )
    other_user = await make_user(email=f"other-{uuid4().hex[:8]}@example.com")

    with pytest.raises(IntegrityError, match=OWNER_FK):
        async with w.session.begin_nested():
            await w.session.execute(
                text("UPDATE studio_artist_memories SET user_id = :user WHERE id = :id"),
                {"user": other_user.id, "id": memory.id},
            )


async def test_the_service_stores_the_artists_owner_pair_on_the_row(studio: StudioWorld) -> None:
    w = studio
    artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
    created = await MemoryService(w.session).create(
        artist.id, w.user_id, w.product_id, MemoryCreate(title="m", body="b")
    )

    row = (
        await w.session.execute(
            select(StudioArtistMemory).where(StudioArtistMemory.id == created.id)
        )
    ).scalar_one()

    assert (row.user_id, row.product_id, row.artist_id) == (w.user_id, "vex", artist.id)


async def test_every_by_artist_read_is_scoped_to_the_caller(
    studio: StudioWorld, make_user: UserFactory
) -> None:
    """User A's artist id with user B's scope - and with A's user but another product - reads
    nothing, counts 0, finds nothing to update and leaves the row as it was."""
    w = studio
    artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
    service = MemoryService(w.session)
    first = await service.create(
        artist.id,
        w.user_id,
        w.product_id,
        MemoryCreate(title="first", body="b", occurred_on=date(2020, 1, 1)),
    )
    second = await service.create(
        artist.id, w.user_id, w.product_id, MemoryCreate(title="second", body="b")
    )
    ids = [first.id, second.id]
    repo = StudioArtistMemoryRepository(w.session)
    other_user = await make_user(email=f"other-{uuid4().hex[:8]}@example.com")

    # the control: the rightful scope sees everything (so an empty result below means "scoped")
    mine = {"user_id": w.user_id, "product_id": w.product_id}
    assert (
        len(await repo.list_page(artist.id, **mine, status=StudioMemoryStatus.ACTIVE, limit=9)) == 2
    )
    assert await repo.count_active(artist.id, **mine) == 2
    assert len(await repo.get_many_owned(ids, artist_id=artist.id, **mine)) == 2
    assert len(await repo.active_in_selection_order(artist.id, **mine, exclude=[], limit=9)) == 2

    for scope in (
        {"user_id": other_user.id, "product_id": w.product_id},  # another user
        {"user_id": w.user_id, "product_id": "synthara"},  # the same user, another product
    ):
        assert not await repo.list_page(
            artist.id, **scope, status=StudioMemoryStatus.ACTIVE, limit=9
        )
        assert await repo.count_active(artist.id, **scope) == 0
        assert not await repo.get_many_owned(ids, artist_id=artist.id, **scope)
        assert not await repo.active_in_selection_order(artist.id, **scope, exclude=[], limit=9)
        # update: no row, with and without changes - and the row is untouched
        assert (
            await repo.update(first.id, artist_id=artist.id, **scope, changes={"title": "X"})
            is None
        )
        assert await repo.update(first.id, artist_id=artist.id, **scope, changes={}) is None

    row = await repo.get(first.id, **mine)
    assert row is not None
    assert (row.title, row.salience, row.status) == ("first", 3, StudioMemoryStatus.ACTIVE.value)


async def test_every_by_id_read_is_scoped_to_the_caller(
    studio: StudioWorld, make_user: UserFactory
) -> None:
    w = studio
    artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
    created = await MemoryService(w.session).create(
        artist.id, w.user_id, w.product_id, MemoryCreate(title="m", body="b")
    )
    repo = StudioArtistMemoryRepository(w.session)
    other_user = await make_user(email=f"other-{uuid4().hex[:8]}@example.com")

    for scope in (
        {"user_id": other_user.id, "product_id": w.product_id},
        {"user_id": w.user_id, "product_id": "synthara"},
    ):
        assert await repo.get(created.id, **scope) is None
        assert await repo.lock(created.id, **scope) is None
        assert await repo.delete(created.id, **scope) is None  # nothing deleted ...
    assert (
        await repo.get(created.id, user_id=w.user_id, product_id=w.product_id) is not None
    )  # ... so it is still there


async def test_delete_answers_the_artist_the_memory_belonged_to(studio: StudioWorld) -> None:
    w = studio
    artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
    created = await MemoryService(w.session).create(
        artist.id, w.user_id, w.product_id, MemoryCreate(title="m", body="b")
    )
    repo = StudioArtistMemoryRepository(w.session)

    deleted = await repo.delete(created.id, user_id=w.user_id, product_id=w.product_id)

    assert deleted == artist.id
    assert await repo.delete(created.id, user_id=w.user_id, product_id=w.product_id) is None


async def test_the_repository_inserts_only_for_the_artists_owner(
    studio: StudioWorld, make_user: UserFactory
) -> None:
    """The repository passes the pair through: the database is what refuses a wrong one."""
    w = studio
    artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
    other_user = await make_user(email=f"other-{uuid4().hex[:8]}@example.com")
    repo = StudioArtistMemoryRepository(w.session)

    def insert(user_id: UUID, product_id: str) -> Coroutine[Any, Any, StudioArtistMemory]:
        return repo.insert(
            user_id=user_id,
            product_id=product_id,
            artist_id=artist.id,
            title="t",
            body="b",
            occurred_on=None,
            themes=[],
            imagery=[],
            salience=3,
            origin=StudioMemoryOrigin.USER,
        )

    assert (await insert(w.user_id, w.product_id)).artist_id == artist.id
    with pytest.raises(IntegrityError, match=OWNER_FK):
        async with w.session.begin_nested():
            await insert(other_user.id, w.product_id)
