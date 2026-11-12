"""LibraryProjectService x ProjectDeletionGuard port: blockers veto a delete, no guards = old behaviour."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from src.api.services.library_project import LibraryProjectInUseError, LibraryProjectService
from src.api.services.library_project_guards import ProjectDeletionBlocker

pytestmark = pytest.mark.unit

USER, PRODUCT = uuid4(), "vex"


class _Guard:
    def __init__(self, *results: list[ProjectDeletionBlocker]) -> None:
        self._results = list(results)
        self.calls = 0

    async def blockers(self, *_: object, **__: object) -> list[ProjectDeletionBlocker]:
        self.calls += 1
        return self._results[min(self.calls - 1, len(self._results) - 1)]


def _session() -> MagicMock:
    session = MagicMock()
    nested = MagicMock()
    nested.__aenter__ = AsyncMock(return_value=None)
    nested.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested)
    return session


def _service(*guards: _Guard) -> LibraryProjectService:
    return LibraryProjectService(MagicMock(), deletion_guards=guards)  # type: ignore[arg-type]


def _repo(*, exists: bool = True, delete: AsyncMock | None = None) -> MagicMock:
    repo = MagicMock()
    repo.get = AsyncMock(return_value=object() if exists else None)
    repo.delete = delete or AsyncMock(return_value=True)
    return repo


async def _delete(service: LibraryProjectService, repo: MagicMock) -> bool:
    with patch("src.api.services.library_project.LibraryProjectRepository", return_value=repo):
        return await service.delete(uuid4(), USER, PRODUCT, session=_session())


async def test_no_guards_keeps_the_old_behaviour() -> None:
    repo = _repo()
    assert await _delete(_service(), repo) is True
    repo.delete.assert_awaited_once()


async def test_missing_project_is_false_and_guards_are_not_consulted() -> None:
    guard = _Guard([ProjectDeletionBlocker("studio_tracks", 2)])
    assert await _delete(_service(guard), _repo(exists=False)) is False
    assert guard.calls == 0


async def test_blockers_veto_the_delete_before_anything_is_deleted() -> None:
    repo = _repo()
    guard = _Guard([ProjectDeletionBlocker("studio_tracks", 3)])
    with pytest.raises(LibraryProjectInUseError) as exc:
        await _delete(_service(guard), repo)
    assert exc.value.blockers == (ProjectDeletionBlocker("studio_tracks", 3),)
    repo.delete.assert_not_awaited()


async def test_blockers_from_every_guard_are_aggregated() -> None:
    a = _Guard([ProjectDeletionBlocker("studio_tracks", 1)])
    b = _Guard([ProjectDeletionBlocker("other", 4)])
    with pytest.raises(LibraryProjectInUseError) as exc:
        await _delete(_service(a, b), _repo())
    assert [(x.code, x.count) for x in exc.value.blockers] == [("studio_tracks", 1), ("other", 4)]


async def test_guards_with_no_blockers_allow_the_delete() -> None:
    repo = _repo()
    assert await _delete(_service(_Guard([])), repo) is True


async def test_fk_race_after_the_guards_ran_becomes_in_use_not_a_500() -> None:
    guard = _Guard([], [ProjectDeletionBlocker("studio_tracks", 1)])  # clear first, blocked after
    repo = _repo(delete=AsyncMock(side_effect=IntegrityError("DELETE", {}, Exception("fk"))))
    with pytest.raises(LibraryProjectInUseError):
        await _delete(_service(guard), repo)
    assert guard.calls == 2


async def test_integrity_error_without_a_blocker_is_not_masked() -> None:
    repo = _repo(delete=AsyncMock(side_effect=IntegrityError("DELETE", {}, Exception("other"))))
    with pytest.raises(IntegrityError):
        await _delete(_service(_Guard([])), repo)


def test_guards_are_a_required_argument() -> None:
    with pytest.raises(TypeError):
        LibraryProjectService(MagicMock())  # type: ignore[call-arg]
