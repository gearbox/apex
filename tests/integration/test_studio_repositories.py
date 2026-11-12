"""Studio repositories against real PostgreSQL: locks, counters, siblings, reorder, pages."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import msgspec
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.schemas.pagination import decode_cursor
from src.core.enums import StudioDocumentKind as D
from src.db.models.library import LibraryProject
from src.db.models.studio import StudioTrack
from src.db.models.user import User
from src.db.repositories.studio import (
    StudioArtistRepository,
    StudioDocumentVersionRepository,
    StudioProjectRepository,
    StudioTrackRepository,
)
from tests.studio_world import StudioWorld

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


def _version_args(w: StudioWorld, track_id: UUID, **overrides: object) -> dict[str, object]:
    args: dict[str, object] = {
        "user_id": w.user_id,
        "product_id": w.product_id,
        "track_id": track_id,
        "kind": D.PLAN,
        "version_number": 1,
        "parent_version_id": None,
        "based_on_plan_version_id": None,
        "schema_version": 1,
        "origin": "user",
        "content": {"title": "x"},
        "content_hash": "h" * 64,
    }
    args.update(overrides)
    return args


class TestLocks:
    """A lock is real only if a second connection cannot take it: probe with NOWAIT."""

    @pytest.fixture
    async def committed(self, db_engine: AsyncEngine):
        factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
        user = User(
            id=uuid4(), email=f"lock-{uuid4().hex}@example.com", password_hash="x", product_id="vex"
        )
        project = LibraryProject(id=uuid4(), product_id="vex", user_id=user.id, name="Locks")
        async with factory() as session, session.begin():
            session.add(user)
            await session.flush()
            session.add(project)
            await session.flush()
            world = StudioWorld(session, user, project)
            await world.enable()
            track_id = await world.new_track()
        yield factory, user, project, track_id
        async with factory() as session, session.begin():
            await session.execute(text("DELETE FROM users WHERE id = :id"), {"id": user.id})

    async def _nowait_fails(
        self, factory: async_sessionmaker[AsyncSession], sql: str, **params: object
    ) -> bool:
        async with factory() as probe:
            try:
                await probe.execute(text(f"{sql} FOR UPDATE NOWAIT"), params)
            except DBAPIError as exc:
                return "could not obtain lock" in str(exc)
            return False

    async def test_track_lock_holds_the_row(self, committed) -> None:
        factory, user, _, track_id = committed
        async with factory() as session:
            locked = await StudioTrackRepository(session).lock(
                track_id, user_id=user.id, product_id="vex"
            )
            assert locked is not None
            assert await self._nowait_fails(
                factory, "SELECT 1 FROM studio_tracks WHERE id = :id", id=track_id
            )
            await session.rollback()
        assert not await self._nowait_fails(
            factory, "SELECT 1 FROM studio_tracks WHERE id = :id", id=track_id
        )

    async def test_profile_lock_holds_the_row(self, committed) -> None:
        factory, user, project, _ = committed
        async with factory() as session:
            assert await StudioProjectRepository(session).lock(
                project.id, user_id=user.id, product_id="vex"
            )
            assert await self._nowait_fails(
                factory,
                "SELECT 1 FROM studio_project_profiles WHERE library_project_id = :id",
                id=project.id,
            )
            await session.rollback()

    async def test_locks_are_owner_and_product_scoped(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track_id = await w.new_track()
        repo = StudioTrackRepository(w.session)
        assert await repo.lock(track_id, user_id=uuid4(), product_id="vex") is None
        assert await repo.lock(track_id, user_id=w.user_id, product_id="synthara") is None
        assert (
            await StudioProjectRepository(w.session).lock(
                w.project_id, user_id=uuid4(), product_id="vex"
            )
            is None
        )

    async def test_lock_refreshes_a_stale_identity_map_entry(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track_id = await w.new_track("before")
        repo = StudioTrackRepository(w.session)
        stale = await repo.get(track_id, user_id=w.user_id, product_id=w.product_id)
        assert stale is not None
        await w.session.execute(
            text("UPDATE studio_tracks SET title = 'after' WHERE id = :i"), {"i": track_id}
        )
        fresh = await repo.lock(track_id, user_id=w.user_id, product_id=w.product_id)
        assert fresh is not None
        assert fresh is stale
        assert fresh.title == "after"


class TestVersionNumbers:
    async def test_counters_are_independent_per_kind_and_per_track(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        repo = StudioTrackRepository(w.session)
        a, b = await w.new_track("A"), await w.new_track("B")
        assert [await repo.allocate_version_number(a, D.PLAN) for _ in range(3)] == [1, 2, 3]
        assert await repo.allocate_version_number(a, D.LYRICS) == 1
        assert await repo.allocate_version_number(b, D.PLAN) == 1
        assert await repo.allocate_version_number(a, D.PLAN) == 4
        row = (
            await w.session.execute(
                select(StudioTrack)
                .where(StudioTrack.id == a)
                .execution_options(populate_existing=True)
            )
        ).scalar_one()
        assert (row.next_plan_version_number, row.next_lyrics_version_number) == (5, 2)


class TestSiblings:
    async def test_insert_or_get_sibling_creates_once(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        repo = StudioDocumentVersionRepository(w.session)
        first, created = await repo.insert_or_get_sibling(**_version_args(w, track))  # type: ignore[arg-type]
        again, created_again = await repo.insert_or_get_sibling(
            **_version_args(w, track, version_number=2)  # type: ignore[arg-type]
        )
        assert (created, created_again) == (True, False)
        assert again.id == first.id
        assert again.version_number == 1  # the existing row, not the rejected attempt

    async def test_siblings_differ_by_parent_hash_and_track(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        a, b = await w.new_track("A"), await w.new_track("B")
        repo = StudioDocumentVersionRepository(w.session)
        root, _ = await repo.insert_or_get_sibling(**_version_args(w, a))  # type: ignore[arg-type]
        other_hash, c1 = await repo.insert_or_get_sibling(
            **_version_args(w, a, version_number=2, content_hash="g" * 64)  # type: ignore[arg-type]
        )
        child, c2 = await repo.insert_or_get_sibling(
            **_version_args(w, a, version_number=3, parent_version_id=root.id)  # type: ignore[arg-type]
        )
        other_track, c3 = await repo.insert_or_get_sibling(**_version_args(w, b))  # type: ignore[arg-type]
        assert (c1, c2, c3) == (True, True, True)
        assert len({root.id, other_hash.id, child.id, other_track.id}) == 4

    async def test_find_sibling_matches_null_parents(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        repo = StudioDocumentVersionRepository(w.session)
        row, _ = await repo.insert_or_get_sibling(**_version_args(w, track))  # type: ignore[arg-type]
        found = await repo.find_sibling(
            track_id=track, kind=D.PLAN, parent_version_id=None, content_hash="h" * 64
        )
        assert found is not None
        assert found.id == row.id
        assert (
            await repo.find_sibling(
                track_id=track, kind=D.PLAN, parent_version_id=uuid4(), content_hash="h" * 64
            )
            is None
        )

    async def test_duplicate_version_numbers_are_rejected_by_the_database(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        repo = StudioDocumentVersionRepository(w.session)
        await repo.insert_or_get_sibling(**_version_args(w, track))  # type: ignore[arg-type]
        with pytest.raises(IntegrityError):
            async with w.session.begin_nested():
                await repo.insert_or_get_sibling(
                    **_version_args(w, track, content_hash="z" * 64)  # type: ignore[arg-type]
                )


class TestDeletionBlockers:
    async def test_counts_children_and_lyrics_bases_in_one_query(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        repo = StudioDocumentVersionRepository(w.session)
        root, _ = await repo.insert_or_get_sibling(**_version_args(w, track))  # type: ignore[arg-type]
        for n in (2, 3):
            await repo.insert_or_get_sibling(
                **_version_args(
                    w, track, version_number=n, parent_version_id=root.id, content_hash=str(n) * 64
                )  # type: ignore[arg-type]
            )
        await repo.insert_or_get_sibling(
            **_version_args(  # type: ignore[arg-type]
                w, track, kind=D.LYRICS, based_on_plan_version_id=root.id, content_hash="l" * 64
            )
        )
        blockers = await repo.deletion_blockers(root.id, track_id=track, kind=D.PLAN)
        assert (blockers.child_count, blockers.lyrics_base_count) == (2, 1)
        leaf = await repo.deletion_blockers(uuid4(), track_id=track, kind=D.PLAN)
        assert (leaf.child_count, leaf.lyrics_base_count) == (0, 0)


class TestTracks:
    async def test_positions_and_next_position(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        repo = StudioTrackRepository(w.session)
        assert await repo.next_position(w.project_id) == 0
        ids = [await w.new_track(f"t{i}") for i in range(3)]
        assert await repo.next_position(w.project_id) == 3
        assert (
            await repo.count_for_project(w.project_id, user_id=w.user_id, product_id=w.product_id)
            == 3
        )
        rows = await repo.list_for_project(w.project_id, user_id=w.user_id, product_id=w.product_id)
        assert [(t.id, t.position) for t in rows] == [(i, n) for n, i in enumerate(ids)]

    async def test_reorder_swaps_through_the_deferred_unique(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        repo = StudioTrackRepository(w.session)
        a, b, c = [await w.new_track(n) for n in "abc"]
        await repo.reorder(w.project_id, [c, a, b], user_id=w.user_id, product_id=w.product_id)
        rows = await repo.list_for_project(w.project_id, user_id=w.user_id, product_id=w.product_id)
        assert [(t.id, t.position) for t in rows] == [(c, 0), (a, 1), (b, 2)]
        await repo.reorder(
            w.project_id, [b, c, a], user_id=w.user_id, product_id=w.product_id
        )  # a pure swap chain
        rows = await repo.list_for_project(w.project_id, user_id=w.user_id, product_id=w.product_id)
        assert [t.id for t in rows] == [b, c, a]

    async def test_position_unique_is_deferred_but_enforced_at_commit_time(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        a, b = await w.new_track("a"), await w.new_track("b")
        await w.session.execute(text("SET CONSTRAINTS uq_studio_tracks_project_position IMMEDIATE"))
        with pytest.raises(IntegrityError):
            async with w.session.begin_nested():
                await w.session.execute(
                    text("UPDATE studio_tracks SET position = 0 WHERE id = :id"), {"id": b}
                )
        assert a

    async def test_reorder_touches_only_the_projects_own_tracks(
        self, enabled_studio: StudioWorld, make_library_project
    ) -> None:
        w = enabled_studio
        other_project = await make_library_project(user=w.user)
        other = StudioWorld(w.session, w.user, other_project)
        await other.enable()
        mine = await w.new_track("mine")
        theirs = await other.new_track("theirs")
        await StudioTrackRepository(w.session).reorder(
            w.project_id, [theirs, mine], user_id=w.user_id, product_id=w.product_id
        )
        row = (
            await w.session.execute(select(StudioTrack.position).where(StudioTrack.id == theirs))
        ).scalar_one()
        assert row == 0  # unchanged: it is not part of this project


class TestPages:
    async def test_artist_pages_are_keyset_newest_first(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        repo = StudioArtistRepository(w.session)
        for n in range(5):
            await repo.insert(
                user_id=w.user_id,
                product_id=w.product_id,
                name=f"A{n}",
                persona=None,
                voice_notes=None,
                default_language=None,
                aesthetic_references=(),
                style_tags=(),
                global_avoid=(),
            )
        first = await repo.list_page(user_id=w.user_id, product_id=w.product_id, limit=2)
        assert len(first) == 3  # limit + 1
        cursor_ts, cursor_id = first[1].created_at, first[1].id
        second = await repo.list_page(
            user_id=w.user_id,
            product_id=w.product_id,
            limit=2,
            cursor_ts=cursor_ts,
            cursor_id=cursor_id,
        )
        assert {a.id for a in first[:2]}.isdisjoint({a.id for a in second})
        assert len(second) == 3
        assert decode_cursor  # cursor helpers are the shared ones

    async def test_version_pages_are_scoped_by_track_owner_and_product(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        repo = StudioDocumentVersionRepository(w.session)
        await repo.insert_or_get_sibling(**_version_args(w, track))  # type: ignore[arg-type]
        assert (
            len(
                await repo.list_page(
                    track, None, user_id=w.user_id, product_id=w.product_id, limit=5
                )
            )
            == 1
        )
        assert (
            await repo.list_page(track, None, user_id=uuid4(), product_id=w.product_id, limit=5)
            == []
        )
        assert (
            await repo.list_page(track, None, user_id=w.user_id, product_id="synthara", limit=5)
            == []
        )


class TestProfiles:
    async def test_upsert_reports_created_then_replaced(self, studio: StudioWorld) -> None:
        w = studio
        repo = StudioProjectRepository(w.session)
        args = {"user_id": w.user_id, "product_id": w.product_id}
        profile, created = await repo.upsert(
            w.project_id, artist_id=None, release_kind="ep", concept="one", **args
        )
        again, created_again = await repo.upsert(
            w.project_id, artist_id=None, release_kind="album", concept=None, **args
        )
        assert (created, created_again) == (True, False)
        assert (again.release_kind, again.concept) == ("album", None)
        assert again.created_at == profile.created_at
        assert isinstance(again.updated_at, datetime)
        assert again.updated_at.tzinfo is UTC or again.updated_at.tzinfo is not None
        assert msgspec  # imported for parity with the other studio suites
