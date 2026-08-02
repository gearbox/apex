"""Integration test for SAVEPOINT isolation of thumbnail insert failures (FIX-1).

Uses the SAVEPOINT-scoped ``db_session`` fixture (see conftest.py) — it never
commits, so assertions here read back rows via the same live session rather
than a real commit (read-your-own-writes within the open transaction is
sufficient to prove the rows landed correctly and the session survived).

A mocked session cannot exercise this: ``OutputRepository.create`` flushing a
duplicate primary key must actually round-trip to PostgreSQL and abort the
underlying transaction/savepoint for this test to mean anything.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import structlog.testing
from sqlalchemy import insert, select

from src.api.services.grok.job_service import GrokJobService
from src.api.services.image_thumbnail import GeneratedThumbnail, ThumbnailResult
from src.api.services.storage.r2 import R2StorageService, R2StorageSettings
from src.core.thumbnails import ThumbnailSpec
from src.db.models.storage import GenerationOutput

if TYPE_CHECKING:
    from collections.abc import Generator

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repositories.output import OutputRepository

_FAKE_WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20
_SM_SPEC = ThumbnailSpec("sm", 150)
_MD_SPEC = ThumbnailSpec("md", 512)


def _make_thumbnails() -> list[GeneratedThumbnail]:
    return [
        GeneratedThumbnail(
            spec=_SM_SPEC, result=ThumbnailResult(data=_FAKE_WEBP, width=100, height=56)
        ),
        GeneratedThumbnail(
            spec=_MD_SPEC, result=ThumbnailResult(data=_FAKE_WEBP, width=400, height=225)
        ),
    ]


@pytest.fixture
def noop_storage() -> Generator[R2StorageService]:
    """Real R2StorageService with the boto client boundary patched to a no-op recorder."""
    mock_client = AsyncMock()
    mock_client.put_object = AsyncMock(return_value={})

    settings = R2StorageSettings(
        account_id="test",
        access_key_id="test",
        secret_access_key="test",
        bucket_name="apex-user-content",
    )
    storage = R2StorageService(settings)

    @asynccontextmanager
    async def _fake_get_client():
        yield mock_client

    with patch.object(storage, "_get_client", _fake_get_client):
        yield storage


async def test_thumbnail_insert_failure_isolated_by_savepoint(
    db_session: AsyncSession,
    output_repo: OutputRepository,
    make_user: object,
    make_job: object,
    noop_storage: R2StorageService,
) -> None:
    """A PK-colliding thumbnail INSERT must not poison the outer transaction —
    the parent output row and the second (valid) thumbnail must survive."""
    user = await make_user(email=f"savepoint-{uuid4().hex[:8]}@example.com")  # type: ignore[operator]
    job = await make_job(user=user)  # type: ignore[operator]

    parent_id = uuid4()
    expires_at = datetime.now(UTC) + timedelta(days=7)
    await output_repo.create(
        id=parent_id,
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{parent_id}.jpg",
        content_type="image/jpeg",
        size_bytes=100,
        format="jpg",
        output_index=0,
        expires_at=expires_at,
        product_id="vex",
    )

    # Pre-insert a row occupying the primary key the "sm" thumbnail insert will
    # attempt to reuse (patched new_id below). Inserted via a Core statement so
    # it never enters the ORM identity map — the later ORM insert must round-trip
    # to PostgreSQL and hit a genuine primary-key violation, not a client-side
    # identity-map guard.
    colliding_id = uuid4()
    await db_session.execute(
        insert(GenerationOutput).values(
            id=colliding_id,
            user_id=user.id,
            job_id=job.id,
            storage_key=f"users/{user.id}/outputs/{job.id}/{colliding_id}-decoy.webp",
            content_type="image/webp",
            size_bytes=1,
            format="webp",
            # This is a decoy used solely to reserve the primary key below;
            # it must remain distinct from the parent full output.
            output_index=1,
            expires_at=expires_at,
            is_thumbnail=False,
            product_id="vex",
        )
    )
    await db_session.flush()

    md_thumb_id = uuid4()

    svc = GrokJobService(grok_client=MagicMock(), storage=noop_storage, retention_days=7)

    with (
        patch(
            "src.api.services.grok.job_service.new_id",
            side_effect=[colliding_id, md_thumb_id],
        ),
        patch(
            "src.api.services.grok.job_service.make_image_thumbnails",
            new=AsyncMock(return_value=_make_thumbnails()),
        ),
        structlog.testing.capture_logs() as logs,
    ):
        await svc._store_output_thumbnails(
            session=db_session,
            output_repo=output_repo,
            user_id=user.id,
            job_id=job.id,
            parent_output_id=parent_id,
            parent_output_index=0,
            source_bytes=b"source-bytes",
            expires_at=expires_at,
            product_id="vex",
        )

    # No exception propagated. Session must still be usable — a poisoned
    # session raises PendingRollbackError on the next statement.
    await db_session.flush()
    await db_session.execute(select(GenerationOutput.id).limit(1))

    skip_events = [log for log in logs if log.get("event") == "grok.thumbnail_skipped"]
    assert len(skip_events) == 1

    result = await db_session.execute(
        select(GenerationOutput).where(GenerationOutput.job_id == job.id)
    )
    rows = {row.id: row for row in result.scalars().all()}

    assert parent_id in rows

    # Decoy row (occupying the colliding PK) is untouched by the failed insert.
    assert colliding_id in rows
    assert rows[colliding_id].is_thumbnail is False
    assert rows[colliding_id].parent_output_id is None

    # Second thumbnail variant succeeded despite the first's failure.
    assert md_thumb_id in rows
    assert rows[md_thumb_id].is_thumbnail is True
    assert rows[md_thumb_id].parent_output_id == parent_id
    assert rows[md_thumb_id].thumbnail_max_edge == 512
    assert rows[md_thumb_id].output_index == 0

    # No row exists for the failed sm variant.
    assert all(row.thumbnail_max_edge != 150 for row in rows.values())
