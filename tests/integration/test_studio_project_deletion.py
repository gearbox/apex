"""Invariant 4: deleting a library project through the ProjectDeletionGuard port."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from src.api.services.library_project import LibraryProjectInUseError, LibraryProjectService
from src.api.services.library_project_guards import ProjectDeletionBlocker
from src.db.models.library import LibraryAssetMetadata, LibraryProject
from src.db.models.studio import StudioProjectProfile, StudioTrack
from src.studio.services.project_deletion_guard import StudioProjectDeletionGuard
from src.studio.services.track import TrackService

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from tests.studio_world import StudioWorld


def _service(session: AsyncSession, *guards: object) -> LibraryProjectService:
    return LibraryProjectService(
        session,
        deletion_guards=guards or (StudioProjectDeletionGuard(),),  # type: ignore[arg-type]
    )


async def _delete(w: StudioWorld, service: LibraryProjectService) -> bool:
    return await service.delete(w.project_id, w.user_id, w.product_id, session=w.session)


async def _exists(session: AsyncSession, model: type, key: UUID, column: str) -> bool:
    return (
        await session.execute(
            select(func.count()).select_from(model).where(getattr(model, column) == key)
        )
    ).scalar_one() > 0


async def test_a_project_with_tracks_is_blocked_with_typed_blockers(
    enabled_studio: StudioWorld,
) -> None:
    w = enabled_studio
    await w.new_track("one")
    await w.new_track("two")
    with pytest.raises(LibraryProjectInUseError) as exc:
        await _delete(w, _service(w.session))
    assert exc.value.blockers == (ProjectDeletionBlocker("studio_tracks", 2),)
    assert await _exists(w.session, LibraryProject, w.project_id, "id")
    assert await _exists(w.session, StudioProjectProfile, w.project_id, "library_project_id")


async def test_a_project_without_tracks_is_deleted_and_its_profile_cascades(
    enabled_studio: StudioWorld,
) -> None:
    w = enabled_studio
    assert await _delete(w, _service(w.session)) is True
    assert not await _exists(w.session, LibraryProject, w.project_id, "id")
    assert not await _exists(w.session, StudioProjectProfile, w.project_id, "library_project_id")


async def test_a_project_that_never_enabled_studio_deletes_as_before(studio: StudioWorld) -> None:
    assert await _delete(studio, _service(studio.session)) is True


async def test_deleting_the_last_track_unblocks_the_project(enabled_studio: StudioWorld) -> None:
    w = enabled_studio
    track = await w.new_track()
    with pytest.raises(LibraryProjectInUseError):
        await _delete(w, _service(w.session))
    await w.tracks.delete(track, w.user_id, w.product_id)
    assert await _delete(w, _service(w.session)) is True


async def test_assets_are_unassigned_as_before(enabled_studio: StudioWorld) -> None:
    w = enabled_studio
    meta = LibraryAssetMetadata(
        id=uuid4(),
        product_id=w.product_id,
        user_id=w.user_id,
        asset_type="upload",
        asset_id=uuid4(),
        project_id=w.project_id,
    )
    w.session.add(meta)
    await w.session.flush()
    assert await _delete(w, _service(w.session)) is True
    await w.session.refresh(meta)
    assert meta.project_id is None  # ON DELETE SET NULL still applies


async def test_another_owners_project_is_simply_not_found(
    enabled_studio: StudioWorld, make_user
) -> None:
    w = enabled_studio
    intruder = await make_user(email=f"x-{uuid4().hex[:6]}@example.com")
    assert (
        await _service(w.session).delete(w.project_id, intruder.id, w.product_id, session=w.session)
        is False
    )
    assert await _exists(w.session, LibraryProject, w.project_id, "id")


async def test_a_track_inserted_after_the_guard_ran_is_a_409_not_a_500(
    enabled_studio: StudioWorld,
) -> None:
    """The race backstop: guards report nothing, a track appears, the NO ACTION FK refuses."""
    w = enabled_studio

    class RacyGuard:
        """First call: a concurrent request adds a track *and* the guard saw nothing."""

        def __init__(self) -> None:
            self.calls = 0

        async def blockers(
            self, project_id: UUID, *, user_id: UUID, product_id: str, session: AsyncSession
        ) -> list[ProjectDeletionBlocker]:
            self.calls += 1
            if self.calls == 1:
                await TrackService(session).create(project_id, user_id, product_id, "sneaky")
                return []
            return list(
                await StudioProjectDeletionGuard().blockers(
                    project_id, user_id=user_id, product_id=product_id, session=session
                )
            )

    guard = RacyGuard()
    with pytest.raises(LibraryProjectInUseError) as exc:
        await _delete(w, _service(w.session, guard))
    assert exc.value.blockers == (ProjectDeletionBlocker("studio_tracks", 1),)
    assert guard.calls == 2
    # the savepoint rolled back: project, profile and the new track are all intact
    assert await _exists(w.session, LibraryProject, w.project_id, "id")
    assert await _exists(w.session, StudioTrack, w.project_id, "library_project_id")
