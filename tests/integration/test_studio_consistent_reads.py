"""Remediation round 2, RF1 part A: a track and the rows its pointers name are read together.

The user action "switch current + delete the old version" commits between two statements of a
request that reads the track first and the versions after. Each statement has its own READ
COMMITTED snapshot, so the first read's pointer then names a row that is gone. Invariants 27 and 29:
a queued run is never failed for it, ``GET`` track detail and the import dry run never answer 500,
and the export of the current pair never answers a spurious 404.

The races are deterministic (``tests/studio_race_support.py``): the second session commits at an
exact point of the request, never after a sleep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.api.schemas.studio import (
    LyricsVersionCreate,
    PlanVersionCreate,
    SunoImportRequest,
)
from src.core.enums import StudioDocumentOrigin
from src.core.uid import new_id
from src.db.repositories.studio import StudioTrackRepository
from src.studio.services.export import ExportService
from src.studio.services.importer import SunoImportService
from src.studio.services.run_context import MemoryArtistContextProvider
from src.studio.services.track import TrackService
from src.studio.services.version import DocumentVersionService
from tests.llm_support import FakeLlm
from tests.studio_race_support import after_first_track_read
from tests.studio_runs_support import PRODUCT
from tests.studio_support import DEFAULT_SECTION_ID, make_lyrics, make_plan

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.studio.planner.context import ArtistContext, RunContextRequest
    from tests.studio_runs_support import Actor, RunWorld
    from tests.studio_world import StudioWorld


# ---------------------------------------------------------------------------
# The repository helper
# ---------------------------------------------------------------------------


class TestGetWithCurrentVersions:
    async def _read(self, w: StudioWorld, track_id: UUID, **overrides: object):  # type: ignore[no-untyped-def]
        args = {"user_id": w.user_id, "product_id": w.product_id} | overrides
        return await StudioTrackRepository(w.session).get_with_current_versions(track_id, **args)  # type: ignore[arg-type]

    async def test_a_track_without_pointers_has_no_rows(self, enabled_studio: StudioWorld) -> None:
        track_id = await enabled_studio.new_track()

        found = await self._read(enabled_studio, track_id)

        assert found is not None
        track, plan_row, lyrics_row = found
        assert (track.id, plan_row, lyrics_row) == (track_id, None, None)

    async def test_a_plan_only_track_has_the_plan_row(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track_id = await w.new_track()
        created, _ = await w.plan(track_id, make_plan(title="Only plan"))

        found = await self._read(w, track_id)

        assert found is not None
        track, plan_row, lyrics_row = found
        assert track.current_plan_version_id == created.version.id
        assert plan_row is not None
        assert (plan_row.id, plan_row.kind, plan_row.content["title"]) == (
            created.version.id,
            "plan",
            "Only plan",
        )
        assert lyrics_row is None

    async def test_a_track_with_both_has_both_rows(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track_id = await w.new_track()
        plan, _ = await w.plan(track_id, make_plan())
        lyrics, _ = await w.lyrics(
            track_id, make_lyrics((DEFAULT_SECTION_ID, ["la la"])), plan.version.id
        )

        found = await self._read(w, track_id)

        assert found is not None
        track, plan_row, lyrics_row = found
        assert plan_row is not None
        assert lyrics_row is not None
        assert (plan_row.id, lyrics_row.id) == (plan.version.id, lyrics.version.id)
        assert (plan_row.kind, lyrics_row.kind) == ("plan", "lyrics")
        assert (track.current_plan_version_id, track.current_lyrics_version_id) == (
            plan_row.id,
            lyrics_row.id,
        )

    async def test_it_is_one_statement(self, enabled_studio: StudioWorld) -> None:
        w = enabled_studio
        track_id = await w.new_track()
        await w.plan(track_id, make_plan())
        statements: list[str] = []
        original = w.session.execute

        async def counting(statement: Any, *args: Any, **kwargs: Any) -> Any:
            statements.append(str(statement))
            return await original(statement, *args, **kwargs)

        w.session.execute = counting  # type: ignore[method-assign]
        await self._read(w, track_id)

        assert len(statements) == 1
        assert (
            "FOR UPDATE" not in statements[0]
        )  # a plain read: no lock, the lock order is untouched

    @pytest.mark.parametrize("who", ["another_user", "another_product"])
    async def test_a_track_that_is_not_the_callers_is_none(
        self, enabled_studio: StudioWorld, who: str
    ) -> None:
        w = enabled_studio
        track_id = await w.new_track()
        await w.plan(track_id, make_plan())

        overrides = {"user_id": new_id()} if who == "another_user" else {"product_id": "synthara"}

        assert await self._read(w, track_id, **overrides) is None
        assert await self._read(w, new_id()) is None  # no such track either


# ---------------------------------------------------------------------------
# Invariant 27: the claim
# ---------------------------------------------------------------------------


class _SwitchingProvider:
    """A context provider that performs the user's "switch current + delete old" before returning."""

    def __init__(self, world: RunWorld, actor: Actor, track_id: UUID) -> None:
        self._world = world
        self._actor = actor
        self._track_id = track_id
        self.replacement: UUID | None = None

    async def for_run(self, request: RunContextRequest, *, session: AsyncSession) -> ArtistContext:
        self.replacement = await self._world.replace_current_plan(self._actor, self._track_id)
        return await MemoryArtistContextProvider().for_run(request, session=session)


class TestTheClaimReadsAConsistentSnapshot:
    async def test_a_switch_and_delete_during_the_claim_does_not_fail_the_run(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor, plan=make_plan(title="Old", empty=True))
        (old,) = await run_world.versions(track_id)
        write = await run_world.submit(actor, track_id)
        provider = _SwitchingProvider(run_world, actor, track_id)

        claimed = await run_world.executor(FakeLlm(), provider=provider).claim(1)

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == ("running", None)
        (item,) = claimed
        # The snapshot is the one the claim read: the plan that was current at that moment.
        assert item.context.plan_version_id == old.id
        assert run.context_snapshot is not None
        assert run.context_snapshot["plan_version_id"] == str(old.id)
        assert run.context_snapshot["current_plan"]["title"] == "Old"
        # The race really happened, and the claim neither failed nor refunded the run.
        assert provider.replacement is not None
        assert [v.id for v in await run_world.versions(track_id)] == [provider.replacement]
        assert await run_world.refunds_and_debits(actor) == (1, 0)

    async def test_a_dangling_pointer_that_is_not_a_race_still_fails_loudly(
        self, run_world: RunWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-null pointer whose joined row is missing is a broken invariant, not "no plan"."""
        actor = await run_world.add_actor()
        track_id = await run_world.new_track(actor, plan=make_plan(title="Old", empty=True))
        write = await run_world.submit(actor, track_id)
        real = StudioTrackRepository.get_with_current_versions

        async def dangling(self: StudioTrackRepository, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            track, _, lyrics = await real(self, *args, **kwargs)  # type: ignore[misc, arg-type]
            return track, None, lyrics

        monkeypatch.setattr(StudioTrackRepository, "get_with_current_versions", dangling)

        assert await run_world.executor(FakeLlm()).claim(1) == []

        run = await run_world.run(write.run.id)
        assert (run.status, run.error_code) == ("failed", "internal_error")
        assert await run_world.refunds_and_debits(actor) == (1, 1)


# ---------------------------------------------------------------------------
# Invariant 29: the same pattern on three request paths
# ---------------------------------------------------------------------------


async def _track_with_pair(world: RunWorld, actor: Actor) -> tuple[UUID, UUID, UUID]:
    """A track whose current pair is committed: ``(track_id, plan_id, lyrics_id)``."""
    track_id = await world.new_track(actor, "Pair")
    async with world.factory() as session, session.begin():
        versions = DocumentVersionService(session)
        plan, _ = await versions.create(
            track_id,
            PlanVersionCreate(content=make_plan(title="Pair plan"), make_current=True),
            actor.user_id,
            PRODUCT,
            origin=StudioDocumentOrigin.USER,
        )
        lyrics, _ = await versions.create(
            track_id,
            LyricsVersionCreate(
                content=make_lyrics((DEFAULT_SECTION_ID, ["la la"])),
                based_on_plan_version_id=plan.version.id,
                make_current=True,
            ),
            actor.user_id,
            PRODUCT,
            origin=StudioDocumentOrigin.USER,
        )
    return track_id, plan.version.id, lyrics.version.id


class TestRequestsUnderTheSameRace:
    async def test_track_detail_answers_with_one_consistent_snapshot(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, plan_id, lyrics_id = await _track_with_pair(run_world, actor)

        async with run_world.factory() as session:
            after_first_track_read(session, lambda: run_world.replace_current_plan(actor, track_id))
            detail = await TrackService(session).get_detail(track_id, actor.user_id, PRODUCT)

        assert (detail.current_plan_version_id, detail.current_lyrics_version_id) == (
            plan_id,
            lyrics_id,
        )
        assert detail.current_plan is not None
        assert detail.current_lyrics is not None
        assert (detail.current_plan.id, detail.current_lyrics.id) == (plan_id, lyrics_id)
        # The race really happened: the pair read is already gone.
        assert (await run_world.track_row(track_id)).current_plan_version_id != plan_id

    async def test_the_import_dry_run_does_not_fail(self, run_world: RunWorld) -> None:
        actor = await run_world.add_actor()
        track_id, plan_id, _ = await _track_with_pair(run_world, actor)

        async with run_world.factory() as session:
            after_first_track_read(session, lambda: run_world.replace_current_plan(actor, track_id))
            out = await SunoImportService(
                session, versions=DocumentVersionService(session)
            ).import_suno(
                track_id,
                SunoImportRequest(
                    style="synthwave", exclude="", lyrics="[Verse]\nla la", dry_run=True
                ),
                actor.user_id,
                PRODUCT,
            )

        assert out.dry_run is True
        assert out.plan_version is None  # a dry run writes nothing
        assert (await run_world.track_row(track_id)).current_plan_version_id != plan_id

    async def test_the_export_of_the_current_pair_is_that_pair_not_a_404(
        self, run_world: RunWorld
    ) -> None:
        actor = await run_world.add_actor()
        track_id, plan_id, lyrics_id = await _track_with_pair(run_world, actor)

        async with run_world.factory() as session:
            after_first_track_read(session, lambda: run_world.replace_current_plan(actor, track_id))
            out = await ExportService(session).render_suno(track_id, actor.user_id, PRODUCT)

        assert (out.plan_version_id, out.lyrics_version_id) == (plan_id, lyrics_id)
        assert "la la" in out.lyrics
        assert (await run_world.track_row(track_id)).current_plan_version_id != plan_id

    async def test_an_explicit_version_that_was_deleted_is_still_a_404(
        self, run_world: RunWorld
    ) -> None:
        """Only the *current* pair is read consistently; a named version keeps its own lookup."""
        from src.studio.errors import StudioNotFound

        actor = await run_world.add_actor()
        track_id, plan_id, _ = await _track_with_pair(run_world, actor)
        await run_world.replace_current_plan(actor, track_id)  # the pair above is now deleted

        async with run_world.factory() as session:
            with pytest.raises(StudioNotFound):
                await ExportService(session).render_suno(
                    track_id, actor.user_id, PRODUCT, plan_version_id=plan_id
                )
