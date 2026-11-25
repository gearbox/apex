"""Helpers shared by the track audio tests (S2a): the reconciliation check and builders.

Invariant 4: after every committed transaction there is **one hold per track audio row and one row
per ``studio.track_audio`` hold**. ``assert_audio_reconciled`` runs after every integration test
that creates audio versions, and re-runs the cast's check too (the two share ``library_asset_holds``).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import text

from src.api.schemas.studio import ProjectProfilePut, TrackAudioCreate
from src.api.services.library_holds import AssetHoldService
from src.core.enums import StudioReleaseKind
from src.core.uid import new_id
from src.db.models.library import LibraryProject
from src.studio.services.project import StudioProjectService
from src.studio.services.track import TrackService
from src.studio.services.track_audio import TrackAudioService
from tests.cast_support import assert_reconciled, wire
from tests.db_locks import backend_pid, wait_until_blocked
from tests.library_holds_support import RETENTION_DAYS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.core.library_ref import AssetRef
    from src.db.models.storage import GenerationOutput, UserImage
    from src.db.models.user import User
    from tests.library_holds_support import HoldWorld

__all__ = ["Racer", "assert_audio_reconciled", "attach_body", "audio_service", "committed_track"]

_ORPHAN_VERSIONS = text(
    """
    SELECT a.id FROM studio_track_audio a
    LEFT JOIN library_asset_holds h ON h.id = a.hold_id
    WHERE h.id IS NULL
       OR h.holder_kind <> 'studio.track_audio'
       OR h.holder_id <> a.id
    """
)
_ORPHAN_HOLDS = text(
    """
    SELECT h.id FROM library_asset_holds h
    LEFT JOIN studio_track_audio a ON a.hold_id = h.id
    WHERE h.holder_kind = 'studio.track_audio'
      AND (a.id IS NULL OR a.id <> h.holder_id)
    """
)
_BAD_NUMBERS = text(
    """
    SELECT track_id FROM studio_track_audio
    GROUP BY track_id
    HAVING count(DISTINCT version_number) <> count(*)
    """
)
_COUNTER_BEHIND = text(
    """
    SELECT t.id FROM studio_tracks t
    JOIN studio_track_audio a ON a.track_id = t.id
    GROUP BY t.id, t.next_audio_version_number
    HAVING max(a.version_number) >= t.next_audio_version_number
    """
)


async def assert_audio_reconciled(session: AsyncSession) -> None:
    """One hold per audio version, one audio version per audio hold, numbers unique and behind
    the counter. Also runs the cast's check: both kinds of holder share the holds table."""
    orphan_versions = (await session.execute(_ORPHAN_VERSIONS)).scalars().all()
    assert not orphan_versions, f"audio versions without their hold: {orphan_versions}"
    orphan_holds = (await session.execute(_ORPHAN_HOLDS)).scalars().all()
    assert not orphan_holds, f"track audio holds without their version: {orphan_holds}"
    assert not (await session.execute(_BAD_NUMBERS)).scalars().all()
    behind = (await session.execute(_COUNTER_BEHIND)).scalars().all()
    assert not behind, f"tracks whose counter is not past their highest audio number: {behind}"
    await assert_reconciled(session)


async def committed_track(world: HoldWorld, user: User, title: str = "Track") -> UUID:
    """A committed studio-enabled project of ``user`` with one track; the user's cascade removes it."""
    async with world.factory() as session:
        project = LibraryProject(
            id=new_id(), product_id=user.product_id, user_id=user.id, name=f"P {uuid4().hex[:8]}"
        )
        session.add(project)
        await session.flush()
        await StudioProjectService(session).upsert_profile(
            project.id,
            user.id,
            user.product_id,
            ProjectProfilePut(release_kind=StudioReleaseKind.ALBUM),
        )
        track = await TrackService(
            session, AssetHoldService(session, retention_days=RETENTION_DAYS)
        ).create(project.id, user.id, user.product_id, title)
        await session.commit()
        return track.id


class Racer:
    """A request that reports its backend pid, then runs ``work(session)`` in that session.

    The test proves the request is **blocked on a lock** by asking PostgreSQL about exactly that
    backend (``tests.db_locks``), lets the lock holder finish, and only then gathers the racer.
    """

    def __init__(self, world: HoldWorld, work: Callable[[AsyncSession], Awaitable[Any]]) -> None:
        self._world = world
        self._work = work
        self.pid = 0
        self.reported = asyncio.Event()
        self.task: asyncio.Task[Any] | None = None

    async def _run(self) -> Any:
        async with self._world.factory() as session:
            self.pid = await backend_pid(session)
            self.reported.set()
            try:
                result = await self._work(session)
                await session.commit()
            except BaseException:
                await session.rollback()
                raise
            return result

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run())
        await asyncio.wait_for(self.reported.wait(), 5)

    async def blocked(self, engine: AsyncEngine) -> None:
        assert self.task is not None
        await wait_until_blocked(engine, self.pid)
        assert not self.task.done()

    async def outcome(self) -> Any:
        assert self.task is not None
        (result,) = await asyncio.wait_for(asyncio.gather(self.task, return_exceptions=True), 15)
        return result


def audio_service(session: AsyncSession) -> TrackAudioService:
    return TrackAudioService(session, AssetHoldService(session, retention_days=RETENTION_DAYS))


def attach_body(
    asset: AssetRef | UserImage | GenerationOutput,
    *,
    label: str | None = None,
    plan_version_id: UUID | None = None,
    lyrics_version_id: UUID | None = None,
) -> TrackAudioCreate:
    return TrackAudioCreate(
        asset=wire(asset),
        label=label,
        plan_version_id=plan_version_id,
        lyrics_version_id=lyrics_version_id,
    )
