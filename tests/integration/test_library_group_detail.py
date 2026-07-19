"""Integration test: LibraryService.get_group_detail parity with GalleryService.

get_group_detail was ported (not re-derived) from GalleryService.get_gallery_detail
for the same job — this asserts the two views agree on every shared field for
an identical GenerationJob + outputs, and that GalleryService is unaffected
(still importable/functional, per the prompt's "leave Gallery untouched"
requirement).
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest_asyncio

from src.api.services.gallery import GalleryService
from src.api.services.library import LibraryService
from src.core.enums import LibraryBadge
from src.core.library_ref import LibraryAssetSource, format_asset_ref

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.storage import GenerationJob, GenerationOutput
    from src.db.models.user import User

OutputFactory = Callable[..., Coroutine[Any, Any, "GenerationOutput"]]


@pytest_asyncio.fixture
async def make_output(
    db_session: AsyncSession,
    make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    make_user: Callable[..., Coroutine[Any, Any, User]],
) -> OutputFactory:
    from src.db.models.storage import GenerationOutput as _GenerationOutput

    async def _factory(
        *,
        job: GenerationJob | None = None,
        user: User | None = None,
        output_index: int = 0,
    ) -> GenerationOutput:
        if user is None:
            user = await make_user(email=f"grpdetail-{uuid4().hex[:8]}@example.com")
        if job is None:
            job = await make_job(user=user, status="completed", product_id="vex")
        oid = uuid4()
        out = _GenerationOutput(
            id=oid,
            user_id=user.id,
            job_id=job.id,
            storage_key=f"users/{user.id}/outputs/{job.id}/{oid}.jpeg",
            content_type="image/jpeg",
            size_bytes=1000,
            format="jpeg",
            output_index=output_index,
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
        )
        db_session.add(out)
        await db_session.flush()
        return out

    return _factory


async def test_group_detail_matches_gallery_detail_for_same_job(
    db_session: AsyncSession,
    make_user: Callable[..., Coroutine[Any, Any, User]],
    make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    make_output: OutputFactory,
) -> None:
    user = await make_user(email=f"parity-{uuid4().hex[:8]}@example.com")
    job = await make_job(
        user=user,
        status="completed",
        generation_type="t2i",
        prompt="a cat wearing sunglasses",
        model="grok-imagine-image",
        provider="grok",
        product_id="vex",
    )
    out1 = await make_output(job=job, user=user, output_index=0)
    out2 = await make_output(job=job, user=user, output_index=1)

    gallery_detail = await GalleryService(session=db_session).get_gallery_detail(
        job.id, user.id, "vex", session=db_session
    )
    library_detail = await LibraryService(session=db_session).get_group_detail(
        job.id, user.id, "vex", session=db_session
    )

    assert gallery_detail is not None
    assert library_detail is not None

    assert library_detail.job_id == gallery_detail.job_id == job.id
    assert library_detail.prompt == gallery_detail.prompt
    assert library_detail.negative_prompt == gallery_detail.negative_prompt
    assert library_detail.model == gallery_detail.model
    assert library_detail.provider == gallery_detail.provider
    assert library_detail.generation_type == gallery_detail.generation_type
    assert library_detail.aspect_ratio == gallery_detail.aspect_ratio
    assert library_detail.token_cost == gallery_detail.token_cost
    assert library_detail.created_at == gallery_detail.created_at
    assert library_detail.completed_at == gallery_detail.completed_at
    assert library_detail.media_type == gallery_detail.media_type

    # badge is a distinct-but-equivalent enum (D1) — same string value
    assert library_detail.badge.value == gallery_detail.badge.value == LibraryBadge.PROMPT.value

    assert len(library_detail.outputs) == len(gallery_detail.outputs) == 2
    for lib_out, gal_out in zip(library_detail.outputs, gallery_detail.outputs, strict=True):
        assert lib_out.id == gal_out.id
        assert lib_out.output_index == gal_out.output_index
        assert lib_out.media.original.url == gal_out.media.original.url

    # Library-only addition: asset_ref present on each output item.
    assert library_detail.outputs[0].asset_ref == format_asset_ref(
        LibraryAssetSource.OUTPUT, out1.id
    )
    assert library_detail.outputs[1].asset_ref == format_asset_ref(
        LibraryAssetSource.OUTPUT, out2.id
    )

    # Gallery lineage is None here (no remix source) — library group lineage too.
    assert gallery_detail.lineage is None
    assert library_detail.lineage is None


async def test_group_detail_returns_none_for_missing_job(db_session: AsyncSession) -> None:
    service = LibraryService(session=db_session)
    result = await service.get_group_detail(uuid4(), uuid4(), "vex", session=db_session)
    assert result is None


async def test_group_detail_returns_none_for_wrong_user(
    db_session: AsyncSession,
    make_user: Callable[..., Coroutine[Any, Any, User]],
    make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
) -> None:
    owner = await make_user(email=f"owner-{uuid4().hex[:8]}@example.com")
    other = await make_user(email=f"other-{uuid4().hex[:8]}@example.com")
    job = await make_job(user=owner, status="completed", product_id="vex")

    service = LibraryService(session=db_session)
    result = await service.get_group_detail(job.id, other.id, "vex", session=db_session)
    assert result is None
