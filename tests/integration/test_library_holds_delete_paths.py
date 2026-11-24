"""Every delete path refuses a held asset (S1-P3, W4, invariants 1 and 2).

The paths: library single delete, bulk delete, job soft-delete, and the foreign-key backstop under
the first two. A refused delete changes **nothing**: no metadata, no tag, no job flag, no R2 object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from litestar.testing import AsyncTestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.routes.jobs import UnifiedJobController
from src.api.routes.library import LibraryController
from src.api.schemas.library import BulkDelete
from src.api.services.content_proxy import ContentNotFoundError, ContentProxyService
from src.api.services.library import LibraryBulkValidationError, LibraryService
from src.api.services.library_holds import AssetHolder, AssetHoldService
from src.core.enums import AssetDeleteOutcome, JobStatus, SoftDeleteOutcome
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref
from src.db.models.library import LibraryAssetMetadata, LibraryAssetTag, LibraryTag
from src.db.models.storage import GenerationJob
from src.db.repositories.job import JobRepository
from src.db.repositories.library import LibraryRepository
from tests.library_holds_support import (
    RETENTION_DAYS,
    HoldWorld,
    bearer,
    build_app,
    output_ref,
    upload_ref,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from litestar import Litestar
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.db.models.user import User

KIND = "studio.cast_reference"


@pytest.fixture
def factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest.fixture
def storage() -> MagicMock:
    mock = MagicMock()
    mock.delete = AsyncMock()
    return mock


@pytest.fixture
def content_proxy(storage: MagicMock) -> ContentProxyService:
    settings = MagicMock()
    settings.content_url_ttl = 60
    return ContentProxyService(storage=storage, settings=settings)


async def _hold(
    world: HoldWorld, ref: AssetRef, user: User, *, kind: str = KIND, product_id: str = "vex"
) -> None:
    async with world.factory() as session:
        await AssetHoldService(session, retention_days=RETENTION_DAYS).hold(
            ref,
            holder=AssetHolder(kind=kind, id=uuid4()),
            user_id=user.id,
            product_id=product_id,
        )
        await session.commit()


async def _favorite_and_tag(world: HoldWorld, user: User, ref: AssetRef) -> None:
    """Give the asset metadata (favorite + title) and a tag: what a refused delete must keep."""
    async with world.factory() as session:
        await LibraryRepository(session).upsert_metadata(
            user.id,
            "vex",
            ref.source,
            ref.asset_id,
            is_favorite=True,
            display_title="Keep me",
        )
        tag = LibraryTag(id=uuid4(), user_id=user.id, product_id="vex", name=f"t-{uuid4().hex[:6]}")
        session.add(tag)
        await session.flush()
        session.add(
            LibraryAssetTag(
                tag_id=tag.id,
                asset_type=ref.source.value,
                asset_id=ref.asset_id,
                user_id=user.id,
                product_id="vex",
            )
        )
        await session.commit()


async def _metadata_and_tag_counts(world: HoldWorld, ref: AssetRef) -> tuple[int, int]:
    async with world.factory() as session:
        metadata = (
            await session.execute(
                select(func.count())
                .select_from(LibraryAssetMetadata)
                .where(LibraryAssetMetadata.asset_id == ref.asset_id)
            )
        ).scalar_one()
        tags = (
            await session.execute(
                select(func.count())
                .select_from(LibraryAssetTag)
                .where(LibraryAssetTag.asset_id == ref.asset_id)
            )
        ).scalar_one()
    return metadata, tags


async def _delete(
    world: HoldWorld, content_proxy: ContentProxyService, user: User, ref: AssetRef
) -> AssetDeleteOutcome:
    async with world.factory() as session:
        return await LibraryService(session=session).delete_asset(
            format_asset_ref(ref.source, ref.asset_id),
            user.id,
            "vex",
            session=session,
            content_proxy=content_proxy,
        )


class TestLibrarySingleDelete:
    async def test_a_held_upload_is_refused_and_everything_is_kept(
        self, hold_world: HoldWorld, content_proxy: ContentProxyService, storage: MagicMock
    ) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user)
        ref = upload_ref(image)
        await _favorite_and_tag(hold_world, user, ref)
        await _hold(hold_world, ref, user)

        outcome = await _delete(hold_world, content_proxy, user, ref)

        assert outcome is AssetDeleteOutcome.HELD
        assert await hold_world.upload_exists(image)
        assert await _metadata_and_tag_counts(hold_world, ref) == (1, 1)
        storage.delete.assert_not_awaited()

    async def test_a_held_output_is_refused_and_everything_is_kept(
        self, hold_world: HoldWorld, content_proxy: ContentProxyService, storage: MagicMock
    ) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user)
        output = await hold_world.output(user, job)
        ref = output_ref(output)
        await _favorite_and_tag(hold_world, user, ref)
        await _hold(hold_world, ref, user)

        outcome = await _delete(hold_world, content_proxy, user, ref)

        assert outcome is AssetDeleteOutcome.HELD
        assert await hold_world.output_exists(output)
        assert await _metadata_and_tag_counts(hold_world, ref) == (1, 1)
        storage.delete.assert_not_awaited()

    async def test_the_same_delete_succeeds_once_the_hold_is_released(
        self, hold_world: HoldWorld, content_proxy: ContentProxyService
    ) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user)
        ref = upload_ref(image)
        async with hold_world.factory() as session:
            service = AssetHoldService(session, retention_days=RETENTION_DAYS)
            hold_id = await service.hold(
                ref, holder=AssetHolder(KIND, uuid4()), user_id=user.id, product_id="vex"
            )
            await session.commit()
        assert await _delete(hold_world, content_proxy, user, ref) is AssetDeleteOutcome.HELD

        async with hold_world.factory() as session:
            await AssetHoldService(session, retention_days=RETENTION_DAYS).release([hold_id])
            await session.commit()

        assert await _delete(hold_world, content_proxy, user, ref) is AssetDeleteOutcome.DELETED
        assert not await hold_world.upload_exists(image)

    async def test_a_hold_that_appears_after_the_check_is_caught_by_the_foreign_key(
        self, hold_world: HoldWorld, content_proxy: ContentProxyService, storage: MagicMock
    ) -> None:
        # The backstop: the pre-check saw no hold (a concurrent hold() committed right after it).
        # The DELETE then violates fk_library_asset_holds_*; the purge that was already flushed is
        # rolled back with it and the answer is HELD - never a 500.
        user = await hold_world.user()
        image = await hold_world.upload(user)
        ref = upload_ref(image)
        await _favorite_and_tag(hold_world, user, ref)
        await _hold(hold_world, ref, user)

        async def no_holders(*_: object) -> tuple[()]:
            return ()

        with patch("src.api.services.library.holder_counts_for_ref", no_holders):
            outcome = await _delete(hold_world, content_proxy, user, ref)

        assert outcome is AssetDeleteOutcome.HELD
        assert await hold_world.upload_exists(image)
        assert await _metadata_and_tag_counts(hold_world, ref) == (1, 1)  # the purge was undone
        storage.delete.assert_not_awaited()

    async def test_an_unrelated_integrity_error_still_surfaces(
        self, hold_world: HoldWorld, content_proxy: ContentProxyService
    ) -> None:
        from sqlalchemy.exc import IntegrityError

        user = await hold_world.user()
        image = await hold_world.upload(user)
        with (
            patch.object(
                content_proxy,
                "delete_content",
                AsyncMock(side_effect=IntegrityError("stmt", {}, Exception("some other"))),
            ),
            pytest.raises(IntegrityError),
        ):
            await _delete(hold_world, content_proxy, user, upload_ref(image))


class TestBulkDelete:
    async def test_held_and_unheld_items_get_their_own_result_and_the_batch_continues(
        self, hold_world: HoldWorld, content_proxy: ContentProxyService
    ) -> None:
        user = await hold_world.user()
        held = await hold_world.upload(user)
        free_a = await hold_world.upload(user)
        job = await hold_world.job(user)
        held_output = await hold_world.output(user, job)
        free_b = await hold_world.upload(user)
        await _hold(hold_world, upload_ref(held), user)
        await _hold(hold_world, output_ref(held_output), user)
        refs = [
            upload_ref(held),
            upload_ref(free_a),
            output_ref(held_output),
            upload_ref(free_b),
        ]

        async with hold_world.factory() as session:
            result = await LibraryService(session=session).bulk_apply(
                BulkDelete(asset_refs=[format_asset_ref(r.source, r.asset_id) for r in refs]),
                user.id,
                "vex",
                session=session,
                content_proxy=content_proxy,
            )

        assert [(r.success, r.error) for r in result.results] == [
            (False, "asset_held"),
            (True, None),
            (False, "asset_held"),
            (True, None),
        ]
        assert (result.succeeded, result.failed) == (2, 2)
        assert await hold_world.upload_exists(held)
        assert await hold_world.output_exists(held_output)
        assert not await hold_world.upload_exists(free_a)
        assert not await hold_world.upload_exists(free_b)

    async def test_other_failures_keep_error_none(
        self, hold_world: HoldWorld, content_proxy: ContentProxyService
    ) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, image.id)

        # The asset vanished between validation and delete: a failed item, with no reason code.
        with patch.object(
            content_proxy, "delete_content", AsyncMock(side_effect=ContentNotFoundError("gone"))
        ):
            async with hold_world.factory() as session:
                result = await LibraryService(session=session).bulk_apply(
                    BulkDelete(asset_refs=[ref]),
                    user.id,
                    "vex",
                    session=session,
                    content_proxy=content_proxy,
                )

        assert [(r.success, r.error) for r in result.results] == [(False, None)]

    async def test_an_unknown_ref_still_fails_the_whole_request_before_anything_runs(
        self, hold_world: HoldWorld, content_proxy: ContentProxyService
    ) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user)
        unknown = format_asset_ref(LibraryAssetSource.UPLOAD, uuid4())

        async with hold_world.factory() as session:
            with pytest.raises(LibraryBulkValidationError):
                await LibraryService(session=session).bulk_apply(
                    BulkDelete(
                        asset_refs=[format_asset_ref(LibraryAssetSource.UPLOAD, image.id), unknown]
                    ),
                    user.id,
                    "vex",
                    session=session,
                    content_proxy=content_proxy,
                )
        assert await hold_world.upload_exists(image)  # validate first, then mutate


class TestJobSoftDelete:
    async def _soft_delete(self, world: HoldWorld, job: GenerationJob, user: User):
        async with world.factory() as session:
            outcome = await JobRepository(session).soft_delete(job.id, user_id=user.id)
            await session.commit()
            return outcome

    async def _is_deleted(self, world: HoldWorld, job: GenerationJob) -> bool:
        async with world.factory() as session:
            return (
                await session.execute(
                    select(GenerationJob.is_deleted).where(GenerationJob.id == job.id)
                )
            ).scalar_one()

    async def test_a_job_with_a_held_output_is_not_deleted(self, hold_world: HoldWorld) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user)
        held = await hold_world.output(user, job)
        await hold_world.output(user, job, output_index=1)
        await _hold(hold_world, output_ref(held), user)

        assert await self._soft_delete(hold_world, job, user) is SoftDeleteOutcome.HELD
        assert not await self._is_deleted(hold_world, job)

    async def test_a_job_without_holds_is_deleted(self, hold_world: HoldWorld) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user)
        await hold_world.output(user, job)

        assert await self._soft_delete(hold_world, job, user) is SoftDeleteOutcome.DELETED
        assert await self._is_deleted(hold_world, job)

    async def test_a_hold_on_another_jobs_output_does_not_matter(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user)
        other_job = await hold_world.job(user)
        await hold_world.output(user, job)
        other_output = await hold_world.output(user, other_job)
        await _hold(hold_world, output_ref(other_output), user)

        assert await self._soft_delete(hold_world, job, user) is SoftDeleteOutcome.DELETED

    async def test_an_in_flight_job_still_answers_in_flight(self, hold_world: HoldWorld) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user, status=JobStatus.RUNNING)

        assert await self._soft_delete(hold_world, job, user) is SoftDeleteOutcome.IN_FLIGHT
        assert not await self._is_deleted(hold_world, job)

    async def test_a_missing_or_foreign_job_is_not_found(self, hold_world: HoldWorld) -> None:
        owner = await hold_world.user()
        other = await hold_world.user()
        job = await hold_world.job(owner)

        assert await self._soft_delete(hold_world, job, other) is SoftDeleteOutcome.NOT_FOUND
        async with hold_world.factory() as session:
            assert (
                await JobRepository(session).soft_delete(uuid4(), user_id=owner.id)
                is SoftDeleteOutcome.NOT_FOUND
            )

    async def test_the_job_can_be_deleted_after_the_last_hold_is_released(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user)
        output = await hold_world.output(user, job)
        async with hold_world.factory() as session:
            hold_id = await AssetHoldService(session, retention_days=RETENTION_DAYS).hold(
                output_ref(output),
                holder=AssetHolder(KIND, uuid4()),
                user_id=user.id,
                product_id="vex",
            )
            await session.commit()
        assert await self._soft_delete(hold_world, job, user) is SoftDeleteOutcome.HELD

        async with hold_world.factory() as session:
            await AssetHoldService(session, retention_days=RETENTION_DAYS).release([hold_id])
            await session.commit()

        assert await self._soft_delete(hold_world, job, user) is SoftDeleteOutcome.DELETED


# ---------------------------------------------------------------------------------------------
# On the wire: the typed 409 and the additive bulk ``error``
# ---------------------------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncTestClient[Litestar]]:
    app = build_app(factory, [LibraryController, UnifiedJobController])
    async with AsyncTestClient(app=app) as test_client:
        yield test_client


class TestOnTheWire:
    async def test_library_delete_answers_409_asset_held_with_the_holder_counts(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user)
        ref = upload_ref(image)
        await _hold(hold_world, ref, user, kind="studio.cast_reference")
        await _hold(hold_world, ref, user, kind="studio.cast_reference")
        await _hold(hold_world, ref, user, kind="a.other")

        response = await client.delete(
            f"/v1/library/assets/{format_asset_ref(ref.source, ref.asset_id)}",
            headers=bearer(user.id),
        )

        assert response.status_code == 409
        body = response.json()
        assert body["error"] == "asset_held"
        assert body["detail"] == {
            "holders": [
                {"kind": "a.other", "count": 1},
                {"kind": "studio.cast_reference", "count": 2},
            ]
        }
        assert await hold_world.upload_exists(image)

    async def test_library_delete_of_an_unheld_asset_is_204_and_of_a_missing_one_404(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user)

        done = await client.delete(f"/v1/library/assets/upload:{image.id}", headers=bearer(user.id))
        missing = await client.delete(
            f"/v1/library/assets/upload:{uuid4()}", headers=bearer(user.id)
        )

        assert done.status_code == 204
        assert missing.status_code == 404

    async def test_bulk_delete_reports_the_reason_per_item(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        held = await hold_world.upload(user)
        free = await hold_world.upload(user)
        await _hold(hold_world, upload_ref(held), user)

        response = await client.post(
            "/v1/library/assets/bulk",
            headers=bearer(user.id),
            json={"type": "delete", "asset_refs": [f"upload:{held.id}", f"upload:{free.id}"]},
        )

        assert response.status_code == 201
        assert response.json()["results"] == [
            {"asset_ref": f"upload:{held.id}", "success": False, "error": "asset_held"},
            {"asset_ref": f"upload:{free.id}", "success": True, "error": None},
        ]

    async def test_job_delete_answers_409_asset_held_with_counts_over_its_outputs(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user)
        first = await hold_world.output(user, job)
        second = await hold_world.output(user, job, output_index=1)
        await _hold(hold_world, output_ref(first), user)
        await _hold(hold_world, output_ref(second), user)

        response = await client.delete(f"/v1/jobs/{job.id}", headers=bearer(user.id))

        assert response.status_code == 409
        assert response.json()["error"] == "asset_held"
        assert response.json()["detail"] == {"holders": [{"kind": KIND, "count": 2}]}

    async def test_job_delete_of_an_unheld_job_is_204_and_an_in_flight_one_still_409_in_progress(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        done = await hold_world.job(user)
        running = await hold_world.job(user, status=JobStatus.RUNNING)

        ok = await client.delete(f"/v1/jobs/{done.id}", headers=bearer(user.id))
        busy = await client.delete(f"/v1/jobs/{running.id}", headers=bearer(user.id))

        assert ok.status_code == 204
        assert busy.status_code == 409
        assert busy.json()["error"] == "job_in_progress"
