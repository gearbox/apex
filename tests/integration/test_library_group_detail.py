"""Integration tests: LibraryService.get_group_detail against a real database."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest_asyncio

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


async def test_group_detail_for_job_with_two_outputs(
    db_session: AsyncSession,
    make_user: Callable[..., Coroutine[Any, Any, User]],
    make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    make_output: OutputFactory,
) -> None:
    user = await make_user(email=f"grpdetail2-{uuid4().hex[:8]}@example.com")
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

    detail = await LibraryService(session=db_session).get_group_detail(
        job.id, user.id, "vex", session=db_session
    )

    assert detail is not None
    assert detail.job_id == job.id
    assert detail.prompt == "a cat wearing sunglasses"
    assert detail.negative_prompt is None
    assert detail.model == "grok-imagine-image"
    assert detail.provider == "grok"
    assert detail.badge == LibraryBadge.PROMPT
    assert detail.lineage is None

    assert len(detail.outputs) == 2
    assert detail.outputs[0].id == out1.id
    assert detail.outputs[1].id == out2.id
    assert detail.outputs[0].asset_ref == format_asset_ref(LibraryAssetSource.OUTPUT, out1.id)
    assert detail.outputs[1].asset_ref == format_asset_ref(LibraryAssetSource.OUTPUT, out2.id)


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
