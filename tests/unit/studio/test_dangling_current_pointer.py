"""Invariant 29 (RD5): a current pointer whose row cannot be read raises, never reads as "none".

The composite same-track foreign keys make a dangling pointer impossible in a real database, so
these tests stub the repositories. Before this fix the services returned ``None`` for such a
pointer: ``set_current`` then skipped the compatibility check and ``get_detail`` reported a track
with no current document.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest

from src.core.enums import StudioSectionKind as K
from src.core.uid import new_id
from src.studio.errors import LyricsIncompatibleWithPlan
from src.studio.services.mappers import DANGLING_CURRENT_POINTER, require_current_row
from src.studio.services.track import TrackService
from src.studio.services.version import DocumentVersionService
from tests.studio_support import make_lyrics, make_plan, make_section, to_json

if TYPE_CHECKING:
    from uuid import UUID

pytestmark = pytest.mark.unit

USER_ID = new_id()
PRODUCT_ID = "vex"


def _track(*, plan: UUID | None = None, lyrics: UUID | None = None) -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=new_id(),
        library_project_id=new_id(),
        position=0,
        title="Track",
        current_plan_version_id=plan,
        current_lyrics_version_id=lyrics,
        created_at=now,
        updated_at=now,
    )


def _row(kind: str, document: Any, row_id: UUID | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=row_id or new_id(), kind=kind, schema_version=1, content=to_json(document)
    )


def _version_service(track: SimpleNamespace, rows: dict[UUID, Any]) -> DocumentVersionService:
    """A service whose repositories know only ``rows`` (every other id reads as missing)."""
    service = DocumentVersionService(MagicMock())

    async def get_on_track(version_id: UUID, _track_id: UUID, **_: object) -> Any:
        return rows.get(version_id)

    service._tracks = MagicMock(lock=AsyncMock(return_value=track))
    service._versions = MagicMock(get_on_track=AsyncMock(side_effect=get_on_track))
    return service


def _track_service(rows: dict[UUID, Any], track: SimpleNamespace) -> tuple[TrackService, AsyncMock]:
    service = TrackService(MagicMock())

    async def get_on_track(version_id: UUID, _track_id: UUID, **_: object) -> Any:
        return rows.get(version_id)

    lookup = AsyncMock(side_effect=get_on_track)
    service._tracks = MagicMock(get=AsyncMock(return_value=track))
    service._versions = MagicMock(get_on_track=lookup)
    return service, lookup


class TestRequireCurrentRow:
    def test_a_row_passes_through(self) -> None:
        row = _row("plan", make_plan())
        assert require_current_row(row) is row  # type: ignore[arg-type]

    def test_none_is_a_runtime_error_with_a_static_message(self) -> None:
        with pytest.raises(RuntimeError) as exc:
            require_current_row(None)
        assert str(exc.value) == DANGLING_CURRENT_POINTER == "studio current pointer without a row"


class TestCurrentDocuments:
    async def test_a_dangling_plan_pointer_raises(self) -> None:
        track = _track(plan=new_id())
        service = _version_service(track, {})
        with pytest.raises(RuntimeError, match="without a row"):
            await service._current_plan_doc(track, USER_ID, PRODUCT_ID)  # type: ignore[arg-type]

    async def test_a_dangling_lyrics_pointer_raises(self) -> None:
        track = _track(plan=new_id(), lyrics=new_id())
        service = _version_service(track, {})
        with pytest.raises(RuntimeError, match="without a row"):
            await service._current_lyrics_doc(track, USER_ID, PRODUCT_ID)  # type: ignore[arg-type]

    async def test_a_null_pointer_is_still_simply_no_document(self) -> None:
        track = _track()
        service = _version_service(track, {})
        assert await service._current_plan_doc(track, USER_ID, PRODUCT_ID) is None  # type: ignore[arg-type]
        assert await service._current_lyrics_doc(track, USER_ID, PRODUCT_ID) is None  # type: ignore[arg-type]

    async def test_a_readable_pointer_decodes(self) -> None:
        plan = make_plan(title="Readable")
        plan_row = _row("plan", plan)
        track = _track(plan=plan_row.id)
        service = _version_service(track, {plan_row.id: plan_row})
        doc = await service._current_plan_doc(track, USER_ID, PRODUCT_ID)  # type: ignore[arg-type]
        assert doc is not None
        assert doc.title == "Readable"


class TestSetCurrent:
    """The compatibility check must not be skipped because a document could not be read."""

    async def test_both_current_pointers_dangling(self) -> None:
        track = _track(plan=new_id(), lyrics=new_id())
        service = _version_service(track, {})
        with pytest.raises(RuntimeError, match="without a row"):
            await service.set_current(
                track.id,
                USER_ID,
                PRODUCT_ID,
                plan_version_id=msgspec.UNSET,
                lyrics_version_id=msgspec.UNSET,
            )

    async def test_current_lyrics_row_missing_while_the_plan_is_switched(self) -> None:
        verse = make_section(K.VERSE)
        new_plan = _row("plan", make_plan(verse, title="new"))
        track = _track(plan=new_id(), lyrics=new_id())
        service = _version_service(track, {new_plan.id: new_plan})
        with pytest.raises(RuntimeError, match="without a row"):
            await service.set_current(
                track.id,
                USER_ID,
                PRODUCT_ID,
                plan_version_id=new_plan.id,
                lyrics_version_id=msgspec.UNSET,
            )

    async def test_current_plan_row_missing_while_lyrics_are_switched(self) -> None:
        verse = make_section(K.VERSE)
        new_lyrics = _row("lyrics", make_lyrics((verse.id, ["la"])))
        track = _track(plan=new_id(), lyrics=None)
        service = _version_service(track, {new_lyrics.id: new_lyrics})
        with pytest.raises(RuntimeError, match="without a row"):
            await service.set_current(
                track.id,
                USER_ID,
                PRODUCT_ID,
                plan_version_id=msgspec.UNSET,
                lyrics_version_id=new_lyrics.id,
            )

    async def test_the_check_still_runs_when_every_row_is_readable(self) -> None:
        verse, chorus = make_section(K.VERSE), make_section(K.CHORUS)
        plan = _row("plan", make_plan(verse, title="verse only"))
        lyrics = _row("lyrics", make_lyrics((chorus.id, ["la"])))
        track = _track(plan=plan.id, lyrics=lyrics.id)
        service = _version_service(track, {plan.id: plan, lyrics.id: lyrics})
        with pytest.raises(LyricsIncompatibleWithPlan):
            await service.set_current(
                track.id,
                USER_ID,
                PRODUCT_ID,
                plan_version_id=msgspec.UNSET,
                lyrics_version_id=msgspec.UNSET,
            )


class TestTrackDetail:
    async def test_a_dangling_plan_pointer_raises_instead_of_returning_none(self) -> None:
        track = _track(plan=new_id())
        service, lookup = _track_service({}, track)
        with pytest.raises(RuntimeError, match="without a row"):
            await service.get_detail(track.id, USER_ID, PRODUCT_ID)
        lookup.assert_awaited_once()

    async def test_a_dangling_lyrics_pointer_raises_instead_of_returning_none(self) -> None:
        track = _track(lyrics=new_id())
        service, _ = _track_service({}, track)
        with pytest.raises(RuntimeError, match="without a row"):
            await service.get_detail(track.id, USER_ID, PRODUCT_ID)

    async def test_null_pointers_are_a_track_without_documents(self) -> None:
        track = _track()
        service, lookup = _track_service({}, track)
        detail = await service.get_detail(track.id, USER_ID, PRODUCT_ID)
        assert (detail.current_plan, detail.current_lyrics) == (None, None)
        lookup.assert_not_awaited()
