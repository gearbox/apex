"""Document versions against real PostgreSQL: numbering, idempotency, pairs, deletion, integrity.

Covers invariants 6-13, 17, 19-21 and 24 of the studio foundation. Most tests drive the
``DocumentVersionService`` on the savepoint-wrapped ``db_session``; the concurrency tests use
committed rows and separate sessions.
"""

from __future__ import annotations

import asyncio
import unicodedata
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import msgspec
import pytest
from sqlalchemy import delete, insert, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.schemas.studio import PlanVersionCreate
from src.core.enums import StudioDocumentKind, StudioIncompatibleLyricsPolicy
from src.core.enums import StudioSectionKind as K
from src.db.models.library import LibraryProject
from src.db.models.studio import StudioDocumentVersion, StudioTrack
from src.db.models.user import User
from src.db.repositories.studio import VersionDeletionBlockers
from src.studio.documents import CURRENT_SCHEMA_VERSION
from src.studio.errors import (
    InvalidCursor,
    LyricsIncompatibleWithPlan,
    ParentVersionNotFound,
    PlanRequired,
    PlanVersionNotFound,
    StudioDocumentInvalid,
    StudioDocumentSchemaError,
    StudioNotFound,
    VersionHasChildren,
    VersionIsCurrent,
    VersionIsLyricsBase,
    VersionLimitReached,
    VersionUnchanged,
)
from src.studio.services import version as version_module
from src.studio.services.version import DocumentVersionService
from src.studio.validation import (
    content_hash,
    normalize_plan,
    to_content,
)
from tests.studio_support import make_lyrics, make_plan, make_section
from tests.studio_world import StudioWorld

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

DETACH = StudioIncompatibleLyricsPolicy.DETACH


def _song(title: str = "Song", *, chorus: bool = True):
    verse = make_section(K.VERSE)
    chorus_section = make_section(K.CHORUS)
    sections = (verse, chorus_section) if chorus else (verse,)
    return make_plan(*sections, title=title), verse, chorus_section


def _raw_version(world: StudioWorld, track_id: UUID, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "id": uuid4(),
        "product_id": world.product_id,
        "user_id": world.user_id,
        "track_id": track_id,
        "kind": "plan",
        "version_number": 900,
        "schema_version": 1,
        "origin": "user",
        "content": {},
        "content_hash": uuid4().hex * 2,
    }
    values.update(overrides)
    return values


async def _expect_integrity_error(session: AsyncSession, statement: object) -> None:
    with pytest.raises(IntegrityError):
        async with session.begin_nested():
            await session.execute(statement)  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# Numbering (invariant 6)
# ---------------------------------------------------------------------------


class TestNumbering:
    async def test_numbers_start_at_one_per_kind_and_increase(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan, verse, _ = _song()
        first, created = await w.plan(track, plan)
        assert created is True
        assert first.version.version_number == 1
        child, _ = await w.plan(track, make_plan(verse, title="Child"), parent=first.version.id)
        assert child.version.version_number == 2
        lyr, _ = await w.lyrics(track, make_lyrics((verse.id, ["la"])), first.version.id)
        assert lyr.version.version_number == 1  # lyrics have their own counter

    async def test_numbers_are_never_reused_after_a_delete(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        numbers = []
        for title in ("a", "b", "c"):
            out, _ = await w.plan(track, make_plan(title=title), make_current=False)
            numbers.append(out.version.version_number)
        assert numbers == [1, 2, 3]

        await w.versions.delete(out.version.id, w.user_id, w.product_id)  # drop #3
        again, _ = await w.plan(track, make_plan(title="d"), make_current=False)
        assert again.version.version_number == 4  # max()+1 would have reused 3

    async def test_numbers_are_per_track(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        a, b = await w.new_track("A"), await w.new_track("B")
        one, _ = await w.plan(a, make_plan(title="x"))
        two, _ = await w.plan(b, make_plan(title="x"))
        assert one.version.version_number == two.version.version_number == 1

    async def test_a_replay_consumes_no_number(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan = make_plan(title="same")
        await w.plan(track, plan)
        _, created = await w.plan(track, plan)
        assert created is False
        nxt, _ = await w.plan(track, make_plan(title="other"))
        assert nxt.version.version_number == 2

    async def test_rejected_creates_burn_no_number(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        with pytest.raises(StudioDocumentInvalid):
            await w.plan(track, make_plan(title=" "))
        out, _ = await w.plan(track, make_plan(title="ok"))
        assert out.version.version_number == 1

    async def test_cap_per_kind_blocks_new_rows_but_not_replays_and_deletion_frees_room(
        self, enabled_studio: StudioWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(version_module, "MAX_VERSIONS_PER_KIND", 2)
        w = enabled_studio
        track = await w.new_track()
        a, _ = await w.plan(track, make_plan(title="a"), make_current=False)
        await w.plan(track, make_plan(title="b"), make_current=False)
        with pytest.raises(VersionLimitReached):
            await w.plan(track, make_plan(title="c"), make_current=False)
        _, created = await w.plan(track, make_plan(title="a"), make_current=False)
        assert created is False  # idempotent replay is exempt from the cap
        await w.versions.delete(a.version.id, w.user_id, w.product_id)
        out, _ = await w.plan(track, make_plan(title="c"), make_current=False)
        assert out.version.version_number == 3


class TestConcurrency:
    """Committed rows + separate sessions: the track lock serializes numbering and replay."""

    @pytest.fixture
    async def committed_track(self, db_engine: AsyncEngine):
        factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
        user = User(
            id=uuid4(),
            email=f"studio-conc-{uuid4().hex}@example.com",
            password_hash="x",
            product_id="vex",
        )
        project = LibraryProject(id=uuid4(), product_id="vex", user_id=user.id, name="Conc")
        async with factory() as session, session.begin():
            session.add(user)
            await session.flush()
            session.add(project)
            await session.flush()
            world = StudioWorld(session, user, project)
            await world.enable()
            track_id = await world.new_track()
        yield factory, user, track_id
        async with factory() as session, session.begin():
            await session.execute(delete(User).where(User.id == user.id))

    async def test_two_concurrent_creates_get_distinct_numbers(self, committed_track) -> None:
        factory, user, track_id = committed_track

        async def create(title: str) -> tuple[int, bool]:
            async with factory() as session:
                out, created = await DocumentVersionService(session).create(
                    track_id,
                    PlanVersionCreate(content=make_plan(title=title), make_current=False),
                    user.id,
                    "vex",
                )
                await session.commit()
                return out.version.version_number, created

        results = await asyncio.gather(create("one"), create("two"))
        assert sorted(number for number, _ in results) == [1, 2]
        assert all(created for _, created in results)

    async def test_two_concurrent_identical_creates_yield_one_row(self, committed_track) -> None:
        factory, user, track_id = committed_track
        plan = make_plan(title="identical")

        async def create() -> tuple[UUID, bool]:
            async with factory() as session:
                out, created = await DocumentVersionService(session).create(
                    track_id, PlanVersionCreate(content=plan), user.id, "vex"
                )
                await session.commit()
                return out.version.id, created

        (id_a, created_a), (id_b, created_b) = await asyncio.gather(create(), create())
        assert id_a == id_b
        assert sorted([created_a, created_b]) == [False, True]
        async with factory() as session:
            rows = (
                (
                    await session.execute(
                        select(StudioDocumentVersion).where(
                            StudioDocumentVersion.track_id == track_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            track = (
                await session.execute(select(StudioTrack).where(StudioTrack.id == track_id))
            ).scalar_one()
        assert len(rows) == 1
        assert track.current_plan_version_id == id_a
        assert track.next_plan_version_number == 2  # the replay burned nothing

    async def test_create_waits_for_the_track_lock(
        self, committed_track, db_engine: AsyncEngine
    ) -> None:
        factory, user, track_id = committed_track
        async with factory() as holder:
            await holder.execute(
                select(StudioTrack).where(StudioTrack.id == track_id).with_for_update()
            )

            async def create() -> int:
                async with factory() as session:
                    out, _ = await DocumentVersionService(session).create(
                        track_id,
                        PlanVersionCreate(content=make_plan(title="blocked"), make_current=False),
                        user.id,
                        "vex",
                    )
                    await session.commit()
                    return out.version.version_number

            task = asyncio.create_task(create())
            async with db_engine.connect() as probe:
                for _ in range(500):
                    waiting = (
                        await probe.execute(
                            text(
                                "SELECT count(*) FROM pg_stat_activity "
                                "WHERE wait_event_type = 'Lock' AND datname = current_database()"
                            )
                        )
                    ).scalar_one()
                    if waiting:
                        break
                    await asyncio.sleep(0.01)
                else:
                    pytest.fail("create never blocked on the track row lock")
            assert not task.done()
            await holder.commit()
        assert await asyncio.wait_for(task, 5) == 1


# ---------------------------------------------------------------------------
# Idempotent create / replay (invariants 7, 8, 24)
# ---------------------------------------------------------------------------


class TestIdempotentCreate:
    async def test_replay_returns_the_same_version_without_a_new_row(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan, *_ = _song()
        first, created = await w.plan(track, plan)
        again, created_again = await w.plan(track, plan)
        assert (created, created_again) == (True, False)
        assert again.version.id == first.version.id
        assert await w.version_count(track) == 1

    async def test_replay_reapplies_make_current(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        a, _ = await w.plan(track, make_plan(title="a"))
        b, _ = await w.plan(track, make_plan(title="b"))
        assert (await w.track_row(track)).current_plan_version_id == b.version.id

        replay, created = await w.plan(track, make_plan(title="a"))  # make_current=True default
        assert created is False
        assert replay.version.id == a.version.id
        assert replay.version.is_current is True
        assert (await w.track_row(track)).current_plan_version_id == a.version.id

    async def test_replay_with_make_current_false_leaves_the_pointer(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        a, _ = await w.plan(track, make_plan(title="a"))
        b, _ = await w.plan(track, make_plan(title="b"))
        await w.plan(track, make_plan(title="a"), make_current=False)
        assert (await w.track_row(track)).current_plan_version_id == b.version.id
        assert a.version.id != b.version.id

    async def test_root_versions_collide_on_null_parent(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan = make_plan(title="root")
        (a, _), (b, created) = await w.plan(track, plan), await w.plan(track, plan)
        assert a.version.parent_version_id is None
        assert (a.version.id == b.version.id, created) == (True, False)

    async def test_same_content_under_another_parent_is_a_distinct_version(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        root, _ = await w.plan(track, make_plan(title="root"))
        other, _ = await w.plan(track, make_plan(title="other"))
        under_root, c1 = await w.plan(track, make_plan(title="leaf"), parent=root.version.id)
        under_other, c2 = await w.plan(track, make_plan(title="leaf"), parent=other.version.id)
        assert c1 and c2
        assert under_root.version.id != under_other.version.id

    async def test_content_equal_to_the_parent_is_unchanged_and_writes_nothing(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        parent, _ = await w.plan(track, make_plan(title="same"))
        with pytest.raises(VersionUnchanged):
            await w.plan(track, make_plan(title="  same "), parent=parent.version.id)
        assert await w.version_count(track) == 1
        nxt, _ = await w.plan(track, make_plan(title="next"))
        assert nxt.version.version_number == 2  # nothing was burned

    async def test_whitespace_and_unicode_composition_variants_are_the_same_version(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        section = make_section(K.VERSE)
        a, _ = await w.plan(track, make_plan(section, title="Café  Noir", tags=("synth  pop",)))
        nfd = unicodedata.normalize("NFD", "Café Noir")
        b, created = await w.plan(track, make_plan(section, title=f" {nfd} ", tags=(" synth pop",)))
        assert created is False
        assert a.version.id == b.version.id

    async def test_stored_content_and_hash_are_the_normalized_form(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        raw = make_plan(make_section(K.VERSE, label=" Big   Verse "), title="  My   Song ")
        out, _ = await w.plan(track, raw)
        row = await w.version_row(out.version.id)
        normalized = normalize_plan(raw)
        assert row.content == to_content(normalized)
        assert row.content["title"] == "My Song"
        assert row.content_hash == content_hash(StudioDocumentKind.PLAN, None, normalized)
        assert row.schema_version == CURRENT_SCHEMA_VERSION[StudioDocumentKind.PLAN]
        assert row.origin == "user"

    async def test_validation_reports_every_issue_at_once_and_writes_nothing(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        shared = uuid4()
        bad = make_plan(
            make_section(section_id=shared, tags=("a", "A")),
            make_section(section_id=shared, label="[x]"),
            title=" ",
        )
        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.plan(track, bad)
        assert {(i.path, i.code) for i in exc.value.issues} == {
            ("title", "empty_after_normalization"),
            ("sections[0].style.tags[1]", "duplicate_tag"),
            ("sections[1].id", "duplicate_section_id"),
            ("sections[1].label", "reserved_characters"),
        }
        assert await w.version_count(track) == 0


# ---------------------------------------------------------------------------
# Parent / base integrity (invariants 9, 10, 21)
# ---------------------------------------------------------------------------


class TestParentIntegrity:
    async def test_parent_from_another_track_is_rejected_by_the_service(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        a, b = await w.new_track("A"), await w.new_track("B")
        on_a, _ = await w.plan(a, make_plan(title="a"))
        with pytest.raises(ParentVersionNotFound):
            await w.plan(b, make_plan(title="b"), parent=on_a.version.id)

    async def test_parent_of_the_other_kind_is_rejected_by_the_service(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan, verse, _ = _song()
        p, _ = await w.plan(track, plan)
        lyr, _ = await w.lyrics(track, make_lyrics((verse.id, ["la"])), p.version.id)
        with pytest.raises(ParentVersionNotFound):
            await w.plan(track, make_plan(title="x"), parent=lyr.version.id)
        with pytest.raises(ParentVersionNotFound):
            await w.lyrics(
                track, make_lyrics((verse.id, ["lb"])), p.version.id, parent=p.version.id
            )

    async def test_parent_must_exist(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        with pytest.raises(ParentVersionNotFound):
            await w.plan(track, make_plan(), parent=uuid4())

    async def test_composite_fk_rejects_a_cross_track_parent_on_a_raw_insert(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        a, b = await w.new_track("A"), await w.new_track("B")
        on_a, _ = await w.plan(a, make_plan(title="a"))
        await _expect_integrity_error(
            w.session,
            insert(StudioDocumentVersion).values(
                _raw_version(w, b, parent_version_id=on_a.version.id)
            ),
        )

    async def test_composite_fk_rejects_a_parent_of_another_kind_on_a_raw_insert(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan, verse, _ = _song()
        p, _ = await w.plan(track, plan)
        lyr, _ = await w.lyrics(track, make_lyrics((verse.id, ["la"])), p.version.id)
        await _expect_integrity_error(
            w.session,
            insert(StudioDocumentVersion).values(
                _raw_version(w, track, parent_version_id=lyr.version.id)  # kind=plan, parent=lyrics
            ),
        )


class TestLyricsBase:
    async def test_base_must_be_a_plan_of_the_same_track(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        a, b = await w.new_track("A"), await w.new_track("B")
        plan, verse, _ = _song()
        plan_a, _ = await w.plan(a, plan)
        plan_b, _ = await w.plan(b, plan)
        lyr_a, _ = await w.lyrics(a, make_lyrics((verse.id, ["la"])), plan_a.version.id)
        doc = make_lyrics((verse.id, ["lb"]))
        with pytest.raises(PlanVersionNotFound):
            await w.lyrics(a, doc, plan_b.version.id)  # another track's plan
        with pytest.raises(PlanVersionNotFound):
            await w.lyrics(a, doc, lyr_a.version.id)  # a lyrics version is not a plan
        with pytest.raises(PlanVersionNotFound):
            await w.lyrics(a, doc, uuid4())

    async def test_lyrics_carry_the_base_and_plans_never_do(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan, verse, _ = _song()
        p, _ = await w.plan(track, plan)
        lyr, _ = await w.lyrics(track, make_lyrics((verse.id, ["la"])), p.version.id)
        assert (await w.version_row(lyr.version.id)).based_on_plan_version_id == p.version.id
        assert (await w.version_row(p.version.id)).based_on_plan_version_id is None

    async def test_db_check_ties_the_base_to_the_lyrics_kind(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        p, _ = await w.plan(track, make_plan())
        await _expect_integrity_error(
            w.session,
            insert(StudioDocumentVersion).values(_raw_version(w, track, kind="lyrics")),
        )
        await _expect_integrity_error(
            w.session,
            insert(StudioDocumentVersion).values(
                _raw_version(w, track, kind="plan", based_on_plan_version_id=p.version.id)
            ),
        )

    async def test_composite_fk_rejects_a_cross_track_base_on_a_raw_insert(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        a, b = await w.new_track("A"), await w.new_track("B")
        plan_a, _ = await w.plan(a, make_plan(title="a"))
        await _expect_integrity_error(
            w.session,
            insert(StudioDocumentVersion).values(
                _raw_version(w, b, kind="lyrics", based_on_plan_version_id=plan_a.version.id)
            ),
        )


class TestCrossTrackPointers:
    async def test_current_pointers_cannot_reference_another_tracks_version(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        a, b = await w.new_track("A"), await w.new_track("B")
        plan_a, _ = await w.plan(a, make_plan(title="a"))
        await w.plan(b, make_plan(title="b"))
        await _expect_integrity_error(
            w.session,
            update(StudioTrack)
            .where(StudioTrack.id == b)
            .values(current_plan_version_id=plan_a.version.id),
        )
        await _expect_integrity_error(
            w.session,
            update(StudioTrack)
            .where(StudioTrack.id == b)
            .values(current_lyrics_version_id=plan_a.version.id),
        )

    async def test_lyrics_pointer_requires_a_plan_pointer(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan, verse, _ = _song()
        p, _ = await w.plan(track, plan, make_current=False)
        lyr, _ = await w.lyrics(
            track, make_lyrics((verse.id, ["la"])), p.version.id, make_current=False
        )
        await _expect_integrity_error(
            w.session,
            update(StudioTrack)
            .where(StudioTrack.id == track)
            .values(current_lyrics_version_id=lyr.version.id),
        )


# ---------------------------------------------------------------------------
# The current (plan, lyrics) pair (invariants 11, 19)
# ---------------------------------------------------------------------------


async def _track_with_pair(w: StudioWorld):
    """A track whose current pair is (plan with verse+chorus, lyrics for the chorus)."""
    track = await w.new_track()
    plan, verse, chorus = _song()
    p, _ = await w.plan(track, plan)
    lyr, _ = await w.lyrics(track, make_lyrics((chorus.id, ["sing along"])), p.version.id)
    return track, p, lyr, verse, chorus


class TestCurrentPair:
    async def test_make_current_defaults_to_true_and_flags_the_response(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track, p, lyr, *_ = await _track_with_pair(w)
        row = await w.track_row(track)
        assert (row.current_plan_version_id, row.current_lyrics_version_id) == (
            p.version.id,
            lyr.version.id,
        )
        assert lyr.version.is_current is True
        assert lyr.lyrics_detached is False

    async def test_make_current_false_changes_no_pointer(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track, p, lyr, verse, _ = await _track_with_pair(w)
        new_plan, _ = await w.plan(track, make_plan(verse, title="v2"), make_current=False)
        assert new_plan.version.is_current is False
        row = await w.track_row(track)
        assert (row.current_plan_version_id, row.current_lyrics_version_id) == (
            p.version.id,
            lyr.version.id,
        )

    async def test_lyrics_made_current_need_a_current_plan(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan, verse, _ = _song()
        p, _ = await w.plan(track, plan, make_current=False)
        with pytest.raises(PlanRequired):
            await w.lyrics(track, make_lyrics((verse.id, ["la"])), p.version.id)
        assert await w.version_count(track) == 1  # nothing written

    async def test_incompatible_plan_with_reject_writes_nothing(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track, p, lyr, verse, _ = await _track_with_pair(w)
        before = await w.version_count()
        counter_before = (await w.track_row(track)).next_plan_version_number

        chorusless = make_plan(verse, title="no chorus")  # the lyrics sing the chorus
        with pytest.raises(LyricsIncompatibleWithPlan) as exc:
            await w.plan(track, chorusless)
        assert exc.value.detail == {
            "issues": [{"path": "sections[0].section_id", "code": "unknown_section"}]
        }
        assert await w.version_count() == before
        row = await w.track_row(track)
        assert row.next_plan_version_number == counter_before
        assert (row.current_plan_version_id, row.current_lyrics_version_id) == (
            p.version.id,
            lyr.version.id,
        )

    async def test_incompatible_plan_is_storable_when_not_made_current(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track, p, lyr, verse, _ = await _track_with_pair(w)
        out, created = await w.plan(track, make_plan(verse, title="draft"), make_current=False)
        assert created is True
        row = await w.track_row(track)
        assert (row.current_plan_version_id, row.current_lyrics_version_id) == (
            p.version.id,
            lyr.version.id,
        )
        assert out.lyrics_detached is False

    async def test_detach_clears_only_the_lyrics_pointer(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track, _, lyr, verse, _ = await _track_with_pair(w)
        out, _ = await w.plan(track, make_plan(verse, title="no chorus"), incompatible=DETACH)
        assert out.lyrics_detached is True
        row = await w.track_row(track)
        assert row.current_plan_version_id == out.version.id
        assert row.current_lyrics_version_id is None
        assert (await w.version_row(lyr.version.id)).id == lyr.version.id  # the version survives

    async def test_detach_without_make_current_is_refused_and_writes_nothing(
        self, enabled_studio: StudioWorld
    ) -> None:
        # Review probe: this used to be accepted, did nothing, and reported lyrics_detached=false.
        w = enabled_studio
        track, p, lyr, verse, _ = await _track_with_pair(w)
        before = await w.version_count()
        counter_before = (await w.track_row(track)).next_plan_version_number

        with pytest.raises(StudioDocumentInvalid) as exc:
            await w.plan(
                track, make_plan(verse, title="no chorus"), make_current=False, incompatible=DETACH
            )
        assert exc.value.detail == {
            "issues": [{"path": "incompatible_lyrics", "code": "requires_make_current"}]
        }
        assert await w.version_count() == before
        row = await w.track_row(track)
        assert row.next_plan_version_number == counter_before
        assert (row.current_plan_version_id, row.current_lyrics_version_id) == (
            p.version.id,
            lyr.version.id,
        )

    async def test_detach_without_make_current_is_refused_before_any_read(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        with pytest.raises(StudioDocumentInvalid):  # not StudioNotFound: no track was looked up
            await w.plan(uuid4(), make_plan(), make_current=False, incompatible=DETACH)

    async def test_detach_with_compatible_lyrics_changes_nothing(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track, _, lyr, verse, chorus = await _track_with_pair(w)
        out, _ = await w.plan(track, make_plan(verse, chorus, title="v2"), incompatible=DETACH)
        assert out.lyrics_detached is False
        row = await w.track_row(track)
        assert row.current_plan_version_id == out.version.id
        assert row.current_lyrics_version_id == lyr.version.id

    async def test_plan_that_makes_a_lyrics_section_instrumental_is_incompatible(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track, _, _, verse, chorus = await _track_with_pair(w)
        instrumental = make_section(K.CHORUS, section_id=chorus.id, instrumental=True)
        with pytest.raises(LyricsIncompatibleWithPlan) as exc:
            await w.plan(track, make_plan(verse, instrumental, title="inst"))
        assert exc.value.detail == {
            "issues": [{"path": "sections[0]", "code": "instrumental_section_has_lines"}]
        }

    async def test_lyrics_checked_against_the_current_plan_not_only_their_base(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        old_plan, verse, chorus = _song("old")
        old, _ = await w.plan(track, old_plan)
        new, _ = await w.plan(track, make_plan(verse, title="new"))  # current plan: verse only
        doc = make_lyrics((chorus.id, ["chorus words"]))  # valid for the old base...
        with pytest.raises(LyricsIncompatibleWithPlan):  # ...but not for the current plan
            await w.lyrics(track, doc, old.version.id)
        assert await w.version_count() == 2
        stored, created = await w.lyrics(track, doc, old.version.id, make_current=False)
        assert created is True
        assert stored.version.is_current is False
        assert (await w.track_row(track)).current_plan_version_id == new.version.id


# ---------------------------------------------------------------------------
# Switching current (set_current)
# ---------------------------------------------------------------------------


class TestSetCurrent:
    async def test_switch_between_versions(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        a, _ = await w.plan(track, make_plan(title="a"))
        b, _ = await w.plan(track, make_plan(title="b"))
        out = await w.versions.set_current(
            track,
            w.user_id,
            w.product_id,
            plan_version_id=a.version.id,
            lyrics_version_id=msgspec.UNSET,
        )
        assert out.current_plan_version_id == a.version.id
        out = await w.versions.set_current(
            track,
            w.user_id,
            w.product_id,
            plan_version_id=b.version.id,
            lyrics_version_id=msgspec.UNSET,
        )
        assert out.current_plan_version_id == b.version.id

    async def test_versions_must_be_of_the_matching_kind_and_track(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track, p, lyr, *_ = await _track_with_pair(w)
        other = await w.new_track("other")
        foreign, _ = await w.plan(other, make_plan(title="foreign"))
        unset = msgspec.UNSET
        for plan_id, lyrics_id in (
            (lyr.version.id, unset),  # lyrics given as the plan
            (unset, p.version.id),  # plan given as the lyrics
            (foreign.version.id, unset),  # another track's version
            (uuid4(), unset),
        ):
            with pytest.raises(StudioNotFound):
                await w.versions.set_current(
                    track,
                    w.user_id,
                    w.product_id,
                    plan_version_id=plan_id,
                    lyrics_version_id=lyrics_id,
                )

    async def test_resulting_pair_must_be_compatible(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track, _, lyr, verse, _ = await _track_with_pair(w)
        chorusless, _ = await w.plan(track, make_plan(verse, title="v"), make_current=False)
        with pytest.raises(LyricsIncompatibleWithPlan):
            await w.versions.set_current(
                track,
                w.user_id,
                w.product_id,
                plan_version_id=chorusless.version.id,
                lyrics_version_id=msgspec.UNSET,
            )
        # the explicit way out: switch plan and detach lyrics in one call
        out = await w.versions.set_current(
            track,
            w.user_id,
            w.product_id,
            plan_version_id=chorusless.version.id,
            lyrics_version_id=None,
        )
        assert (out.current_plan_version_id, out.current_lyrics_version_id) == (
            chorusless.version.id,
            None,
        )
        assert (await w.version_row(lyr.version.id)).id == lyr.version.id

    async def test_lyrics_without_a_plan_is_plan_required(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan, verse, _ = _song()
        p, _ = await w.plan(track, plan, make_current=False)
        lyr, _ = await w.lyrics(
            track, make_lyrics((verse.id, ["la"])), p.version.id, make_current=False
        )
        with pytest.raises(PlanRequired):
            await w.versions.set_current(
                track,
                w.user_id,
                w.product_id,
                plan_version_id=msgspec.UNSET,
                lyrics_version_id=lyr.version.id,
            )

    async def test_setting_the_same_pair_is_a_noop(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track, p, lyr, *_ = await _track_with_pair(w)
        out = await w.versions.set_current(
            track,
            w.user_id,
            w.product_id,
            plan_version_id=p.version.id,
            lyrics_version_id=lyr.version.id,
        )
        assert (out.current_plan_version_id, out.current_lyrics_version_id) == (
            p.version.id,
            lyr.version.id,
        )


# ---------------------------------------------------------------------------
# Deleting versions (invariants 13, 20)
# ---------------------------------------------------------------------------


class TestDeleteVersion:
    async def test_a_leaf_that_is_not_current_can_be_deleted(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        a, _ = await w.plan(track, make_plan(title="a"))
        b, _ = await w.plan(track, make_plan(title="b"))  # b is current
        await w.versions.delete(a.version.id, w.user_id, w.product_id)
        assert await w.version_count(track) == 1
        with pytest.raises(StudioNotFound):
            await w.versions.get(a.version.id, w.user_id, w.product_id)
        assert b.version.is_current

    async def test_current_versions_cannot_be_deleted(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        _, p, lyr, *_ = await _track_with_pair(w)
        with pytest.raises(VersionIsCurrent):
            await w.versions.delete(lyr.version.id, w.user_id, w.product_id)
        with pytest.raises(VersionIsCurrent):
            await w.versions.delete(p.version.id, w.user_id, w.product_id)

    async def test_versions_with_children_cannot_be_deleted(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        parent, _ = await w.plan(track, make_plan(title="parent"), make_current=False)
        await w.plan(track, make_plan(title="c1"), parent=parent.version.id, make_current=False)
        await w.plan(track, make_plan(title="c2"), parent=parent.version.id, make_current=False)
        with pytest.raises(VersionHasChildren) as exc:
            await w.versions.delete(parent.version.id, w.user_id, w.product_id)
        assert exc.value.detail == {"count": 2}

    async def test_plans_that_lyrics_are_based_on_cannot_be_deleted(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan, verse, _ = _song()
        base, _ = await w.plan(track, plan, make_current=False)
        await w.lyrics(track, make_lyrics((verse.id, ["la"])), base.version.id, make_current=False)
        with pytest.raises(VersionIsLyricsBase) as exc:
            await w.versions.delete(base.version.id, w.user_id, w.product_id)
        assert exc.value.detail == {"count": 1}

    async def test_the_foreign_keys_enforce_the_same_rules_on_a_raw_delete(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan, verse, _ = _song()
        base, _ = await w.plan(track, plan)  # current plan
        child, _ = await w.plan(
            track, make_plan(verse, title="child"), parent=base.version.id, make_current=False
        )
        lyr, _ = await w.lyrics(
            track, make_lyrics((verse.id, ["la"])), base.version.id
        )  # based on base
        for victim in (base.version.id, lyr.version.id):  # parent+base+current / current lyrics
            await _expect_integrity_error(
                w.session, delete(StudioDocumentVersion).where(StudioDocumentVersion.id == victim)
            )
        assert child.version.id  # untouched leaf still deletable below
        await w.session.execute(
            delete(StudioDocumentVersion).where(StudioDocumentVersion.id == child.version.id)
        )

    async def test_a_blocker_that_appears_after_the_check_is_still_reported_precisely(
        self, enabled_studio: StudioWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        parent, _ = await w.plan(track, make_plan(title="parent"), make_current=False)
        await w.plan(track, make_plan(title="child"), parent=parent.version.id, make_current=False)

        repo = w.versions._versions
        real = repo.deletion_blockers
        calls = 0

        async def stale_once(
            version_id: UUID, *, track_id: UUID, kind: StudioDocumentKind
        ) -> VersionDeletionBlockers:
            nonlocal calls
            calls += 1
            if calls == 1:  # the first (pre-delete) check "misses" the child
                return VersionDeletionBlockers(child_count=0, lyrics_base_count=0)
            return await real(version_id, track_id=track_id, kind=kind)

        monkeypatch.setattr(repo, "deletion_blockers", stale_once)
        with pytest.raises(VersionHasChildren):  # the FK fired; the re-check named the reason
            await w.versions.delete(parent.version.id, w.user_id, w.product_id)
        assert calls == 2

    async def test_deleting_a_track_removes_its_versions(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track, p, _, verse, _ = await _track_with_pair(w)
        await w.plan(
            track,
            make_plan(verse, make_section(K.CHORUS), title="child"),
            parent=p.version.id,
            make_current=False,
        )
        other = await w.new_track("keep")
        await w.plan(other, make_plan(title="keep"))
        assert await w.version_count(track) == 3

        await w.tracks.delete(
            track, w.user_id, w.product_id
        )  # parent/base/current FKs must not block

        assert await w.version_count(track) == 0
        assert await w.version_count(other) == 1
        with pytest.raises(StudioNotFound):
            await w.tracks.get_detail(track, w.user_id, w.product_id)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestReads:
    async def test_list_is_newest_first_with_keyset_pages_and_kind_filter(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track = await w.new_track()
        plan, verse, _ = _song()
        p, _ = await w.plan(track, plan)
        for title in ("b", "c"):
            await w.plan(track, make_plan(verse, title=title), make_current=False)
        await w.lyrics(track, make_lyrics((verse.id, ["la"])), p.version.id)

        page1 = await w.versions.list_page(
            track, None, w.user_id, w.product_id, limit=2, cursor=None
        )
        assert (len(page1.items), page1.has_more) == (2, True)
        assert page1.next_cursor
        page2 = await w.versions.list_page(
            track, None, w.user_id, w.product_id, limit=2, cursor=page1.next_cursor
        )
        assert (len(page2.items), page2.has_more, page2.next_cursor) == (2, False, None)
        seen = [v.id for v in (*page1.items, *page2.items)]
        assert len(set(seen)) == 4

        plans = await w.versions.list_page(
            track, StudioDocumentKind.PLAN, w.user_id, w.product_id, limit=10, cursor=None
        )
        assert {v.kind for v in plans.items} == {StudioDocumentKind.PLAN}
        assert [v.version_number for v in plans.items] == [3, 2, 1]
        assert [v.is_current for v in plans.items] == [False, False, True]

    async def test_invalid_cursor(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track = await w.new_track()
        with pytest.raises(InvalidCursor):
            await w.versions.list_page(
                track, None, w.user_id, w.product_id, limit=5, cursor="garbage"
            )

    async def test_get_decodes_the_stored_content(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        _, p, lyr, *_ = await _track_with_pair(w)
        plan_detail = await w.versions.get(p.version.id, w.user_id, w.product_id)
        assert plan_detail.content.title == "Song"  # type: ignore[union-attr]
        lyrics_detail = await w.versions.get(lyr.version.id, w.user_id, w.product_id)
        assert lyrics_detail.content.sections[0].lines[0].text == "sing along"  # type: ignore[union-attr]
        assert lyrics_detail.based_on_plan_version_id == p.version.id  # type: ignore[union-attr]

    async def test_unknown_schema_version_fails_loudly_on_every_read_path(
        self, enabled_studio: StudioWorld
    ) -> None:
        w = enabled_studio
        track, p, *_ = await _track_with_pair(w)
        await w.session.execute(
            update(StudioDocumentVersion)
            .where(StudioDocumentVersion.id == p.version.id)
            .values(schema_version=7)
        )
        with pytest.raises(StudioDocumentSchemaError):
            await w.versions.get(p.version.id, w.user_id, w.product_id)
        with pytest.raises(StudioDocumentSchemaError):
            await w.tracks.get_detail(track, w.user_id, w.product_id)
        with pytest.raises(StudioDocumentSchemaError):
            await w.exports.render_suno(track, w.user_id, w.product_id)
