"""Integration tests for LibraryService.get_lineage_graph (T8) against a real database.

Covers: upload → i2v output → extracted frame → i2i output chain walks
correctly with relations and source_timestamp_ms; depth-cap truncation;
SET-NULL-severed chain terminates cleanly (T10); descendant caps + totals;
cross-user/missing asset returns None (404 semantics).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

import src.api.services.library as library_module
from src.api.services.library import LibraryService
from src.core.library_ref import LibraryAssetSource, format_asset_ref
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.user import User

pytestmark = pytest.mark.asyncio


async def _make_upload(
    session: AsyncSession,
    *,
    user: User,
    source_output_id: object = None,
    source_upload_id: object = None,
    source_timestamp_ms: int | None = None,
    content_type: str = "image/png",
) -> UserImage:
    img_id = uuid4()
    image = UserImage(
        id=img_id,
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/{img_id}.png",
        original_filename="frame.png",
        content_type=content_type,
        size_bytes=100,
        format="png",
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id=user.product_id,
        source_output_id=source_output_id,
        source_upload_id=source_upload_id,
        source_timestamp_ms=source_timestamp_ms,
    )
    session.add(image)
    await session.flush()
    return image


async def _make_job(
    session: AsyncSession,
    *,
    user: User,
    generation_type: str = "t2i",
    input_image_id: object = None,
    source_output_id: object = None,
    source_job_id: object = None,
    status: str = "completed",
) -> GenerationJob:
    job = GenerationJob(
        id=uuid4(),
        user_id=user.id,
        product_id=user.product_id,
        name="Lineage Job",
        prompt="a test prompt",
        status=status,
        generation_type=generation_type,
        provider="grok",
        input_image_id=input_image_id,
        source_output_id=source_output_id,
        source_job_id=source_job_id,
    )
    session.add(job)
    await session.flush()
    return job


async def _make_output(
    session: AsyncSession, *, user: User, job: GenerationJob
) -> GenerationOutput:
    oid = uuid4()
    out = GenerationOutput(
        id=oid,
        user_id=user.id,
        job_id=job.id,
        storage_key=f"users/{user.id}/outputs/{job.id}/{oid}.jpeg",
        content_type="image/jpeg",
        size_bytes=1000,
        format="jpeg",
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        is_thumbnail=False,
        product_id=user.product_id,
    )
    session.add(out)
    await session.flush()
    return out


class TestAncestorChainWalk:
    async def test_upload_to_i2v_output_to_frame_to_i2i_output(
        self,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"lineagechain-{uuid4().hex[:8]}@example.com")
        service = LibraryService(session=db_session)

        upload_a = await _make_upload(db_session, user=user)
        job1 = await _make_job(
            db_session, user=user, generation_type="i2v", input_image_id=upload_a.id
        )
        output_b = await _make_output(db_session, user=user, job=job1)
        frame_c = await _make_upload(
            db_session,
            user=user,
            source_output_id=output_b.id,
            source_timestamp_ms=1500,
            content_type="image/png",
        )
        job2 = await _make_job(
            db_session, user=user, generation_type="i2i", input_image_id=frame_c.id
        )
        output_d = await _make_output(db_session, user=user, job=job2)

        graph = await service.get_lineage_graph(
            format_asset_ref(LibraryAssetSource.OUTPUT, output_d.id),
            user.id,
            "vex",
            session=db_session,
        )

        assert graph is not None
        assert graph.focus.asset_ref == format_asset_ref(LibraryAssetSource.OUTPUT, output_d.id)
        assert graph.ancestors_truncated is False

        relations = [(e.relation.value, e.node.asset_ref) for e in graph.ancestors]
        assert relations == [
            ("generated_from_upload", format_asset_ref(LibraryAssetSource.UPLOAD, frame_c.id)),
            ("frame_of_output", format_asset_ref(LibraryAssetSource.OUTPUT, output_b.id)),
            ("generated_from_upload", format_asset_ref(LibraryAssetSource.UPLOAD, upload_a.id)),
        ]
        # The frame edge must carry the extraction timestamp.
        frame_edge = graph.ancestors[1]
        assert frame_edge.source_timestamp_ms == 1500

    async def test_depth_cap_truncates_long_chain(
        self,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"lineagedepth-{uuid4().hex[:8]}@example.com")
        service = LibraryService(session=db_session)

        # 12-upload chain (image_0 root .. image_11 focus) — walking from
        # image_11 needs 11 hops to reach the root, one more than the cap.
        images = [await _make_upload(db_session, user=user)]
        for _ in range(11):
            images.append(await _make_upload(db_session, user=user, source_upload_id=images[-1].id))

        graph = await service.get_lineage_graph(
            format_asset_ref(LibraryAssetSource.UPLOAD, images[-1].id),
            user.id,
            "vex",
            session=db_session,
        )

        assert graph is not None
        assert graph.ancestors_truncated is True
        assert len(graph.ancestors) == library_module._LINEAGE_DEPTH_CAP

    async def test_set_null_severed_chain_terminates_cleanly(
        self,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"lineagesevered-{uuid4().hex[:8]}@example.com")
        service = LibraryService(session=db_session)

        upload_a = await _make_upload(db_session, user=user)
        job1 = await _make_job(db_session, user=user, input_image_id=upload_a.id)
        output_b = await _make_output(db_session, user=user, job=job1)

        # Deleting the upload triggers ON DELETE SET NULL on the job's FK.
        await db_session.delete(upload_a)
        await db_session.flush()
        await db_session.refresh(job1)
        assert job1.input_image_id is None

        graph = await service.get_lineage_graph(
            format_asset_ref(LibraryAssetSource.OUTPUT, output_b.id),
            user.id,
            "vex",
            session=db_session,
        )

        assert graph is not None
        assert graph.ancestors == ()
        assert graph.ancestors_truncated is False


class TestDescendantsCapAndTotals:
    async def test_descendants_cap_and_truncated_flag(
        self,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(library_module, "_LINEAGE_DESCENDANTS_CAP", 2)

        user = await make_user(email=f"lineagedesccap-{uuid4().hex[:8]}@example.com")
        service = LibraryService(session=db_session)

        upload_a = await _make_upload(db_session, user=user)
        for _ in range(3):
            job = await _make_job(db_session, user=user, input_image_id=upload_a.id)
            await _make_output(db_session, user=user, job=job)

        graph = await service.get_lineage_graph(
            format_asset_ref(LibraryAssetSource.UPLOAD, upload_a.id),
            user.id,
            "vex",
            session=db_session,
        )

        assert graph is not None
        assert graph.descendants_truncated is True
        assert len(graph.descendants) == 2
        # Totals are independent of the capped list — all 3 are counted.
        assert graph.descendant_totals.job_count == 3

    async def test_frame_and_output_descendants_both_present(
        self,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"lineagedescmix-{uuid4().hex[:8]}@example.com")
        service = LibraryService(session=db_session)

        upload_a = await _make_upload(db_session, user=user)
        job = await _make_job(db_session, user=user, input_image_id=upload_a.id)
        output_b = await _make_output(db_session, user=user, job=job)
        frame = await _make_upload(
            db_session, user=user, source_upload_id=upload_a.id, source_timestamp_ms=250
        )

        graph = await service.get_lineage_graph(
            format_asset_ref(LibraryAssetSource.UPLOAD, upload_a.id),
            user.id,
            "vex",
            session=db_session,
        )

        assert graph is not None
        refs = {e.node.asset_ref: e for e in graph.descendants}
        assert format_asset_ref(LibraryAssetSource.OUTPUT, output_b.id) in refs
        assert refs[format_asset_ref(LibraryAssetSource.OUTPUT, output_b.id)].relation.value == (
            "generated_from_upload"
        )
        assert format_asset_ref(LibraryAssetSource.UPLOAD, frame.id) in refs
        frame_edge = refs[format_asset_ref(LibraryAssetSource.UPLOAD, frame.id)]
        assert frame_edge.relation.value == "frame_of_upload"
        assert frame_edge.source_timestamp_ms == 250

        assert graph.descendant_totals.job_count == 1
        assert graph.descendant_totals.frame_count == 1


class TestLineageNotFound:
    async def test_missing_asset_returns_none(
        self,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"lineagemissing-{uuid4().hex[:8]}@example.com")
        service = LibraryService(session=db_session)

        graph = await service.get_lineage_graph(
            format_asset_ref(LibraryAssetSource.UPLOAD, uuid4()),
            user.id,
            "vex",
            session=db_session,
        )
        assert graph is None

    async def test_cross_user_asset_returns_none(
        self,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        owner = await make_user(email=f"lineageowner-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"lineageother-{uuid4().hex[:8]}@example.com")
        service = LibraryService(session=db_session)

        upload = await _make_upload(db_session, user=owner)

        graph = await service.get_lineage_graph(
            format_asset_ref(LibraryAssetSource.UPLOAD, upload.id),
            other.id,
            "vex",
            session=db_session,
        )
        assert graph is None

    async def test_malformed_ref_returns_none(
        self,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"lineagemalformed-{uuid4().hex[:8]}@example.com")
        service = LibraryService(session=db_session)

        graph = await service.get_lineage_graph(
            "not-a-valid-ref", user.id, "vex", session=db_session
        )
        assert graph is None
