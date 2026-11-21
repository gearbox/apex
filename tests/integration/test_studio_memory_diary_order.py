"""The memory list is in diary order: ``occurred_on DESC NULLS LAST, created_at DESC, id DESC``.

The artist page's default tab is a diary, so the list follows the episode date, not the time the
entry was typed. Paging is keyset: walking the cursor with any page size returns every memory
exactly once, in the Python-sorted order, whatever mix of dates, undated rows and ties there is.
The rows are seeded with explicit ``created_at`` values (a few of them equal) and drawn ids, so the
three tie-breakers are all exercised - the rows of one transaction would all share a timestamp.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from src.api.schemas.pagination import encode_cursor
from src.api.schemas.studio import ArtistCreate
from src.core.enums import StudioMemoryStatus
from src.db.models.studio import StudioArtistMemory
from src.db.models.user import User
from src.studio.errors import InvalidCursor
from src.studio.services.artist import ArtistService
from src.studio.services.memory import MemoryService

if TYPE_CHECKING:
    from tests.studio_world import StudioWorld

PRODUCT = "vex"
_BASE = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Seed:
    """One memory to insert: its episode date (or none), when it was written and its id."""

    occurred_on: date | None
    created_at: datetime
    id: UUID


def diary_order(seeds: list[Seed]) -> list[UUID]:
    """The expected order, from the definition: stable sorts from the last key to the first."""
    rows = sorted(seeds, key=lambda s: s.id, reverse=True)
    rows.sort(key=lambda s: s.created_at, reverse=True)
    rows.sort(
        key=lambda s: (s.occurred_on is None, -s.occurred_on.toordinal() if s.occurred_on else 0)
    )
    return [s.id for s in rows]


async def seed_memories(
    session: AsyncSession,
    user_id: UUID,
    artist_id: UUID,
    seeds: list[Seed],
    *,
    status: str = "active",
) -> None:
    if not seeds:
        return
    await session.execute(
        insert(StudioArtistMemory),
        [
            {
                "id": s.id,
                "user_id": user_id,
                "product_id": PRODUCT,
                "artist_id": artist_id,
                "title": "t",
                "body": "b",
                "occurred_on": s.occurred_on,
                "created_at": s.created_at,
                "status": status,
            }
            for s in seeds
        ],
    )


async def walk(
    service: MemoryService, artist_id: UUID, user_id: UUID, limit: int, *, max_pages: int = 50
) -> list[list[UUID]]:
    """Every page of the active list, following ``next_cursor`` until ``has_more`` is false."""
    pages: list[list[UUID]] = []
    cursor: str | None = None
    for _ in range(max_pages):
        page = await service.list_page(
            artist_id,
            user_id,
            PRODUCT,
            status=StudioMemoryStatus.ACTIVE,
            limit=limit,
            cursor=cursor,
        )
        pages.append([m.id for m in page.items])
        assert (page.next_cursor is None) == (not page.has_more)
        if not page.has_more:
            return pages
        cursor = page.next_cursor
    pytest.fail("the cursor never reached the end of the list")


def _at(minutes: int) -> datetime:
    return _BASE + timedelta(minutes=minutes)


async def test_the_list_is_in_diary_order_with_undated_last(studio: StudioWorld) -> None:
    w = studio
    artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
    day, earlier = date(2021, 5, 1), date(2019, 3, 9)
    # ids chosen so the id tie-break is visible: low and high, in an order the sort must undo
    low, high = UUID(int=1), UUID(int=2**100)
    seeds = [
        Seed(earlier, _at(50), UUID(int=10)),  # older episode, written last
        Seed(day, _at(10), UUID(int=11)),  # same day, written first
        Seed(day, _at(20), UUID(int=12)),  # same day, written later: comes first
        Seed(day, _at(20), low),  # same day, same minute: the higher id first
        Seed(day, _at(20), high),
        Seed(None, _at(90), UUID(int=13)),  # undated, written last: first among the undated
        Seed(None, _at(5), UUID(int=14)),
        Seed(None, _at(5), UUID(int=15)),  # undated tie: the higher id first
    ]
    await seed_memories(w.session, w.user_id, artist.id, seeds)

    page = await MemoryService(w.session).list_page(
        artist.id, w.user_id, PRODUCT, status=StudioMemoryStatus.ACTIVE, limit=50, cursor=None
    )

    got = [m.id for m in page.items]
    assert got == diary_order(seeds)
    assert got == [
        high, UUID(int=12), low,  # the day: later written first, then the higher id
        UUID(int=11),
        UUID(int=10),  # the older episode, although it was written last
        UUID(int=13), UUID(int=15), UUID(int=14),  # undated last, newest written first
    ]  # fmt: skip


async def test_a_page_boundary_inside_a_tie_loses_and_repeats_nothing(studio: StudioWorld) -> None:
    w = studio
    artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
    day = date(2021, 5, 1)
    seeds = [Seed(day, _at(1), UUID(int=n)) for n in range(1, 6)]  # five rows, all tied but the id
    seeds += [Seed(None, _at(1), UUID(int=n)) for n in range(6, 9)]  # three undated, all tied
    await seed_memories(w.session, w.user_id, artist.id, seeds)

    pages = await walk(MemoryService(w.session), artist.id, w.user_id, 2)

    assert [len(p) for p in pages] == [2, 2, 2, 2]
    assert [i for page in pages for i in page] == diary_order(seeds)


async def test_the_list_filters_by_status_and_pages_each_in_its_own_order(
    studio: StudioWorld,
) -> None:
    w = studio
    artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
    active = [Seed(date(2020, 1, n), _at(n), uuid4()) for n in (1, 2, 3)]
    archived = [Seed(date(2020, 6, n), _at(n), uuid4()) for n in (1, 2)]
    await seed_memories(w.session, w.user_id, artist.id, active)
    await seed_memories(w.session, w.user_id, artist.id, archived, status="archived")
    service = MemoryService(w.session)

    listed_active = await service.list_page(
        artist.id, w.user_id, PRODUCT, status=StudioMemoryStatus.ACTIVE, limit=50, cursor=None
    )
    listed_archived = await service.list_page(
        artist.id, w.user_id, PRODUCT, status=StudioMemoryStatus.ARCHIVED, limit=50, cursor=None
    )

    assert [m.id for m in listed_active.items] == diary_order(active)
    assert [m.id for m in listed_archived.items] == diary_order(archived)


async def test_a_cursor_from_another_endpoint_is_invalid(studio: StudioWorld) -> None:
    """An artist-list cursor is ``{created_at, id}``: no ``occurred_on``, so not ours."""
    w = studio
    artist = await w.artists.create(w.user_id, w.product_id, ArtistCreate(name="Nova"))
    foreign = encode_cursor(_at(1), uuid4())

    with pytest.raises(InvalidCursor):
        await MemoryService(w.session).list_page(
            artist.id, w.user_id, PRODUCT, status=StudioMemoryStatus.ACTIVE, limit=5, cursor=foreign
        )


# ---------------------------------------------------------------------------
# The property: any data, any page size
# ---------------------------------------------------------------------------

_seeds = st.lists(
    st.builds(
        Seed,
        # few distinct dates and minutes, so equal dates and equal timestamps are the norm
        occurred_on=st.none() | st.integers(0, 3).map(lambda n: date(2020, 1, 1 + n)),
        created_at=st.integers(0, 3).map(_at),
        id=st.uuids(),
    ),
    unique_by=lambda s: s.id,
    max_size=14,
)


async def _page_through(url: str, seeds: list[Seed], limit: int) -> tuple[list[UUID], list[UUID]]:
    """Seed one throwaway artist in a rolled-back transaction and walk its list."""
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()
            try:
                async with AsyncSession(bind=connection, expire_on_commit=False) as session:
                    user = User(
                        id=uuid4(),
                        email=f"diary-{uuid4().hex}@example.com",
                        password_hash="x",
                        product_id=PRODUCT,
                    )
                    session.add(user)
                    await session.flush()
                    artist = await ArtistService(session).create(
                        user.id, PRODUCT, ArtistCreate(name="Nova")
                    )
                    await seed_memories(session, user.id, artist.id, seeds)
                    pages = await walk(MemoryService(session), artist.id, user.id, limit)
                    return [i for page in pages for i in page], diary_order(seeds)
            finally:
                await transaction.rollback()
    finally:
        await engine.dispose()


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(seeds=_seeds, limit=st.integers(1, 7))
def test_paging_with_any_page_size_returns_every_memory_once_in_order(
    test_database_url: str, seeds: list[Seed], limit: int
) -> None:
    walked, expected = asyncio.run(_page_through(test_database_url, seeds, limit))

    assert walked == expected
    assert len(set(walked)) == len(seeds)  # nothing twice, nothing missing
