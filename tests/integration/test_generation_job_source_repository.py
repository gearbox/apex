"""Integration contracts for ordered generation source provenance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.services.generation.source_media import ResolvedSourceMedia
from src.core.enums import MediaKind
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref
from src.db.repositories.generation_job_source import GenerationJobSourceRepository

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.storage import GenerationJob, UserImage
    from src.db.models.user import User


async def test_create_many_preserves_every_source_position_and_asset_ref(
    db_session: AsyncSession,
    make_job: Callable[..., Coroutine[object, object, GenerationJob]],
    make_user: Callable[..., Coroutine[object, object, User]],
    make_user_image: Callable[..., Coroutine[object, object, UserImage]],
) -> None:
    user = await make_user(email="source-provenance@example.com")
    job = await make_job(user=user)
    uploads = [await make_user_image(user=user) for _ in range(3)]
    sources = [
        ResolvedSourceMedia(
            position=position,
            ref=AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=upload.id),
            asset_ref=format_asset_ref(LibraryAssetSource.UPLOAD, upload.id),
            media_kind=MediaKind.IMAGE,
            content_type=upload.content_type,
            storage_key=upload.storage_key,
            size_bytes=upload.size_bytes,
            job_id=None,
        )
        for position, upload in enumerate(uploads)
    ]
    repository = GenerationJobSourceRepository(db_session)

    rows = await repository.create_many(job_id=job.id, product_id="vex", sources=sources)

    assert [row.position for row in rows] == [0, 1, 2]
    persisted = await repository.list_for_job(job.id, product_id="vex")
    assert [row.position for row in persisted] == [0, 1, 2]
    assert [row.asset_ref for row in persisted] == [source.asset_ref for source in sources]
