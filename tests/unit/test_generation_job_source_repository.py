"""Unit contracts for ordered generation source persistence."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.api.services.generation.source_media import ResolvedSourceMedia
from src.core.enums import MediaKind
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref
from src.db.repositories.generation_job_source import GenerationJobSourceRepository


def _source(position: int) -> ResolvedSourceMedia:
    asset_id = uuid4()
    return ResolvedSourceMedia(
        position=position,
        ref=AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=asset_id),
        asset_ref=format_asset_ref(LibraryAssetSource.UPLOAD, asset_id),
        media_kind=MediaKind.IMAGE,
        content_type="image/png",
        storage_key=f"uploads/{asset_id}.png",
        size_bytes=10,
        job_id=None,
    )


async def test_create_many_persists_all_sources_in_request_order() -> None:
    session = MagicMock()
    session.flush = AsyncMock()
    repository = GenerationJobSourceRepository(session)
    job_id = uuid4()
    sources = [_source(0), _source(1), _source(2)]

    rows = await repository.create_many(job_id=job_id, product_id="vex", sources=sources)

    assert [row.position for row in rows] == [0, 1, 2]
    assert [row.asset_ref for row in rows] == [source.asset_ref for source in sources]
    assert [row.source_upload_id for row in rows] == [source.ref.asset_id for source in sources]
    session.add_all.assert_called_once_with(rows)
    session.flush.assert_awaited_once()


async def test_list_for_job_returns_sources_ordered_by_position() -> None:
    expected_rows = [MagicMock(), MagicMock()]
    result = MagicMock()
    result.scalars.return_value.all.return_value = expected_rows
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    repository = GenerationJobSourceRepository(session)

    rows = await repository.list_for_job(uuid4(), product_id="vex")

    assert rows == expected_rows
    session.execute.assert_awaited_once()
