"""Audio in the library (S2a W6 / W11.2): list filters, paging, peaks, and the generation refusals.

Regression under test: ``media_type=image`` used to be ``NOT content_type LIKE 'video/%'``, which
would list audio as images. Every kind is now a positive prefix match.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from litestar.testing import AsyncTestClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.routes.library import LibraryController
from src.api.schemas.unified_generation import SourceMediaReference
from src.api.services.generation.source_media import (
    SourceMediaKindNotSupportedError,
    SourceMediaResolver,
    SourceMediaValidationError,
)
from src.api.services.library import LibraryService
from src.api.services.library_holds import AssetHolder, AssetHoldService
from src.core.enums import LibrarySort, MediaKind
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref
from src.core.product_registry import VEX_CONFIG
from src.db.models.library import LibraryAudioProfile
from tests.library_holds_support import (
    AUDIO,
    RETENTION_DAYS,
    HoldWorld,
    bearer,
    build_app,
    make_audio_upload,
    make_job,
    make_output,
    make_upload,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from litestar import Litestar
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.db.models.user import User


@pytest.fixture
def factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def client(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncTestClient[Litestar]]:
    app = build_app(factory, [LibraryController])
    async with AsyncTestClient(app=app) as test_client:
        yield test_client


async def _list(
    world: HoldWorld,
    user: User,
    *,
    media_type: MediaKind | None = None,
    sort: LibrarySort = LibrarySort.NEWEST,
    limit: int = 50,
    cursor: str | None = None,
    expiring: bool | None = None,
) -> tuple[list[str], str | None, list[object]]:
    async with world.factory() as session:
        page = await LibraryService(session=session).list_assets(
            user.id,
            "vex",
            VEX_CONFIG,
            session=session,
            limit=limit,
            cursor=cursor,
            media_type=media_type,
            sort=sort,
            expiring=expiring,
        )
    return [item.asset_ref for item in page.items], page.next_cursor, list(page.items)


async def _mixed(world: HoldWorld, user: User) -> dict[str, str]:
    """One of each kind in both tables; returns ``{name: asset_ref}``."""
    now = datetime.now(UTC)
    async with world.factory() as session:
        image = await make_upload(session, user, expires_at=now + timedelta(days=5))
        video = await make_upload(
            session, user, content_type="video/mp4", expires_at=now + timedelta(days=4)
        )
        audio = await make_audio_upload(session, user, expires_at=now + timedelta(days=3))
        job = await make_job(session, user)
        out_image = await make_output(session, user, job, expires_at=now + timedelta(days=2))
        out_video = await make_output(
            session,
            user,
            job,
            content_type="video/mp4",
            output_index=1,
            expires_at=now + timedelta(days=1),
        )
        out_audio = await make_output(
            session,
            user,
            job,
            content_type="audio/mpeg",
            output_index=2,
            expires_at=now + timedelta(hours=12),
        )
        out_audio.duration_ms = 30_000
        out_audio.width = None
        out_audio.height = None
        await session.commit()
    return {
        "image": format_asset_ref(LibraryAssetSource.UPLOAD, image.id),
        "video": format_asset_ref(LibraryAssetSource.UPLOAD, video.id),
        "audio": format_asset_ref(LibraryAssetSource.UPLOAD, audio.id),
        "out_image": format_asset_ref(LibraryAssetSource.OUTPUT, out_image.id),
        "out_video": format_asset_ref(LibraryAssetSource.OUTPUT, out_video.id),
        "out_audio": format_asset_ref(LibraryAssetSource.OUTPUT, out_audio.id),
    }


class TestKindFilters:
    async def test_audio_is_listed_with_its_kind_and_no_thumbnail(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        refs = await _mixed(hold_world, user)

        _, _, items = await _list(hold_world, user, media_type=MediaKind.AUDIO)

        assert {item.asset_ref for item in items} == {refs["audio"], refs["out_audio"]}  # type: ignore[attr-defined]
        for item in items:
            assert item.media.media_type is MediaKind.AUDIO  # type: ignore[attr-defined]
            assert item.media.variants == []  # type: ignore[attr-defined]
            assert item.media.original.width is None  # type: ignore[attr-defined]
            assert item.media.original.height is None  # type: ignore[attr-defined]
            assert item.duration_ms is not None  # type: ignore[attr-defined]
            actions = {a.value for a in item.available_actions}  # type: ignore[attr-defined]
            # no generation action takes audio; outputs keep their settings / reproduce actions
            assert actions == {"favorite", "rename", "download", "delete"} | (
                {"view_settings", "reproduce"} if item.source.value == "output" else set()  # type: ignore[attr-defined]
            )

    async def test_audio_is_excluded_from_the_image_and_video_filters(
        self, hold_world: HoldWorld
    ) -> None:
        """The regression: the old image filter was ``NOT LIKE 'video/%'`` and listed audio."""
        user = await hold_world.user()
        refs = await _mixed(hold_world, user)

        images, _, image_items = await _list(hold_world, user, media_type=MediaKind.IMAGE)
        videos, _, _ = await _list(hold_world, user, media_type=MediaKind.VIDEO)

        assert set(images) == {refs["image"], refs["out_image"]}
        assert set(videos) == {refs["video"], refs["out_video"]}
        assert not {refs["audio"], refs["out_audio"]} & (set(images) | set(videos))
        assert {item.media.media_type for item in image_items} == {MediaKind.IMAGE}  # type: ignore[attr-defined]

    async def test_unfiltered_lists_every_kind_once(self, hold_world: HoldWorld) -> None:
        user = await hold_world.user()
        refs = await _mixed(hold_world, user)

        listed, _, _ = await _list(hold_world, user)

        assert sorted(listed) == sorted(refs.values())

    async def test_the_kinds_partition_the_library(self, hold_world: HoldWorld) -> None:
        user = await hold_world.user()
        await _mixed(hold_world, user)
        everything, _, _ = await _list(hold_world, user)

        by_kind = [await _list(hold_world, user, media_type=kind) for kind in MediaKind]

        flat = [ref for refs, _, _ in by_kind for ref in refs]
        assert sorted(flat) == sorted(everything)  # no asset in two kinds, none in no kind

    async def test_the_query_parameter_accepts_audio_over_http(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        refs = await _mixed(hold_world, user)

        response = await client.get("/v1/library/?media_type=audio", headers=bearer(user.id))

        assert response.status_code == 200
        body = response.json()
        assert {item["asset_ref"] for item in body["items"]} == {refs["audio"], refs["out_audio"]}
        assert {item["media"]["media_type"] for item in body["items"]} == {"audio"}

    async def test_an_unknown_kind_is_still_refused(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()

        response = await client.get("/v1/library/?media_type=document", headers=bearer(user.id))

        assert response.status_code == 400


class TestPagingOverMixedKinds:
    @pytest.mark.parametrize("sort", list(LibrarySort))
    async def test_keyset_paging_visits_every_asset_once_in_every_order(
        self, hold_world: HoldWorld, sort: LibrarySort
    ) -> None:
        user = await hold_world.user()
        refs = await _mixed(hold_world, user)
        seen: list[str] = []
        cursor: str | None = None

        for _ in range(10):
            page, cursor, _ = await _list(hold_world, user, sort=sort, limit=2, cursor=cursor)
            seen.extend(page)
            if cursor is None:
                break

        assert sorted(seen) == sorted(refs.values())
        assert len(seen) == len(set(seen))

    async def test_each_kind_pages_on_its_own(self, hold_world: HoldWorld) -> None:
        user = await hold_world.user()
        async with hold_world.factory() as session:
            for _ in range(5):
                await make_audio_upload(session, user)
            await make_upload(session, user)
            await session.commit()
        seen: list[str] = []
        cursor: str | None = None

        for _ in range(10):
            page, cursor, _ = await _list(
                hold_world, user, media_type=MediaKind.AUDIO, limit=2, cursor=cursor
            )
            seen.extend(page)
            if cursor is None:
                break

        assert len(seen) == 5
        assert len(set(seen)) == 5


class TestExpiringAndHeld:
    async def test_expiring_filters_audio_like_any_upload(self, hold_world: HoldWorld) -> None:
        user = await hold_world.user()
        now = datetime.now(UTC)
        async with hold_world.factory() as session:
            soon = await make_audio_upload(session, user, expires_at=now + timedelta(days=1))
            later = await make_audio_upload(session, user, expires_at=now + timedelta(days=30))
            await session.commit()

        expiring, _, _ = await _list(hold_world, user, media_type=MediaKind.AUDIO, expiring=True)
        lasting, _, _ = await _list(hold_world, user, media_type=MediaKind.AUDIO, expiring=False)

        assert expiring == [format_asset_ref(LibraryAssetSource.UPLOAD, soon.id)]
        assert lasting == [format_asset_ref(LibraryAssetSource.UPLOAD, later.id)]

    async def test_a_held_audio_asset_is_flagged_and_never_expires(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        async with hold_world.factory() as session:
            audio = await make_audio_upload(
                session, user, expires_at=datetime.now(UTC) - timedelta(days=2)
            )
            await session.commit()
            await AssetHoldService(session, retention_days=RETENTION_DAYS).hold(
                AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=audio.id),
                holder=AssetHolder(kind="studio.track_audio", id=uuid4()),
                user_id=user.id,
                product_id="vex",
                accept=AUDIO,
            )
            await session.commit()

        listed, _, items = await _list(hold_world, user, media_type=MediaKind.AUDIO)
        expiring, _, _ = await _list(hold_world, user, media_type=MediaKind.AUDIO, expiring=True)
        soonest, _, _ = await _list(
            hold_world, user, media_type=MediaKind.AUDIO, sort=LibrarySort.EXPIRING_SOON
        )

        assert len(listed) == 1
        assert items[0].held is True  # type: ignore[attr-defined]
        assert expiring == []  # a held asset never expires
        assert soonest == listed


class TestDetail:
    async def test_an_audio_upload_has_a_detail_without_dimensions_or_variants(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        audio = await hold_world.audio(user, duration_ms=187_000)

        async with hold_world.factory() as session:
            detail = await LibraryService(session=session).get_asset_detail(
                format_asset_ref(LibraryAssetSource.UPLOAD, audio.id),
                user.id,
                "vex",
                VEX_CONFIG,
                session=session,
            )

        assert detail is not None
        assert detail.media.media_type is MediaKind.AUDIO
        assert detail.duration_ms == 187_000
        assert detail.media.original.duration_ms == 187_000
        assert detail.media.original.content_type == "audio/flac"
        assert detail.media.original.width is None
        assert detail.media.variants == []
        assert detail.held is False


class TestPeaksEndpoint:
    async def test_it_returns_the_stored_peaks_as_signed_integers(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        stored = bytes([0, 1, 127, 128, 129, 255] * 341 + [0, 0])  # 2,048 bytes, signed on the wire
        audio = await hold_world.audio(user, peaks=stored)

        response = await client.get(
            f"/v1/library/assets/upload:{audio.id}/peaks", headers=bearer(user.id)
        )

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"level", "version", "buckets", "peaks"}
        assert body["level"] == "overview"  # additive (S3a)
        assert body["version"] == 1
        assert body["buckets"] == 1024
        assert len(body["peaks"]) == 2048
        assert all(-128 <= value <= 127 for value in body["peaks"])
        assert body["peaks"][:6] == [0, 1, 127, -128, -127, -1]
        assert [v & 0xFF for v in body["peaks"]] == list(stored)

    async def test_the_peaks_of_an_audio_output_are_served_too(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld
    ) -> None:
        from src.db.models.library import LibraryAudioProfile

        user = await hold_world.user()
        job = await hold_world.job(user)
        output = await hold_world.output(user, job, content_type="audio/mpeg")
        async with hold_world.factory() as session:
            session.add(
                LibraryAudioProfile(
                    id=uuid4(),
                    output_id=output.id,
                    codec="mp3",
                    sample_rate_hz=44100,
                    channels=2,
                    peaks=bytes(2048),
                    peaks_version=1,
                )
            )
            await session.commit()

        response = await client.get(
            f"/v1/library/assets/output:{output.id}/peaks", headers=bearer(user.id)
        )

        assert response.status_code == 200
        assert response.json()["peaks"] == [0] * 2048

    async def test_an_image_is_a_422_media_kind_not_supported(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user)
        video = await hold_world.upload(user, content_type="video/mp4")

        for asset in (image, video):
            response = await client.get(
                f"/v1/library/assets/upload:{asset.id}/peaks", headers=bearer(user.id)
            )
            assert response.status_code == 422
            assert response.json()["error"] == "media_kind_not_supported"

    @pytest.mark.parametrize(
        "case", ["foreign", "other_product", "unknown", "malformed", "thumbnail"]
    )
    async def test_it_is_an_asset_not_found_404_whatever_the_reason(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld, case: str
    ) -> None:
        user = await hold_world.user()
        other = await hold_world.user()
        other_product = await hold_world.user(product_id="synthara")
        refs = {
            "foreign": f"upload:{(await hold_world.audio(other)).id}",
            "other_product": f"upload:{(await hold_world.audio(other_product)).id}",
            "unknown": f"upload:{uuid4()}",
            "malformed": "not-a-ref",
            "thumbnail": f"upload:{await self._thumbnail(hold_world, user)}",
        }

        response = await client.get(
            f"/v1/library/assets/{refs[case]}/peaks", headers=bearer(user.id)
        )

        assert response.status_code == 404
        assert response.json()["error"] == "asset_not_found"

    @staticmethod
    async def _thumbnail(world: HoldWorld, user: User) -> object:
        parent = await world.upload(user)
        thumb = await world.upload(
            user, is_thumbnail=True, parent=parent, content_type="audio/flac"
        )
        return thumb.id

    async def test_an_output_of_an_unfinished_job_has_no_peaks(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld
    ) -> None:
        from src.core.enums import JobStatus

        user = await hold_world.user()
        job = await hold_world.job(user, status=JobStatus.RUNNING)
        output = await hold_world.output(user, job, content_type="audio/mpeg")

        response = await client.get(
            f"/v1/library/assets/output:{output.id}/peaks", headers=bearer(user.id)
        )

        assert response.status_code == 404

    async def test_it_needs_authentication(self, client: AsyncTestClient[Litestar]) -> None:
        response = await client.get(f"/v1/library/assets/upload:{uuid4()}/peaks")

        assert response.status_code in {401, 403}

    async def test_it_is_one_select(self, db_engine: AsyncEngine, hold_world: HoldWorld) -> None:
        user = await hold_world.user()
        audio = await hold_world.audio(user)
        statements: list[str] = []

        def on_execute(_conn: object, _cursor: object, statement: str, *_: object) -> None:
            statements.append(statement)

        event.listen(db_engine.sync_engine, "before_cursor_execute", on_execute)
        try:
            async with hold_world.factory() as session:
                peaks = await LibraryService(session=session).get_audio_peaks(
                    format_asset_ref(LibraryAssetSource.UPLOAD, audio.id),
                    user.id,
                    "vex",
                    session=session,
                )
        finally:
            event.remove(db_engine.sync_engine, "before_cursor_execute", on_execute)

        assert peaks is not None
        selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
        assert len(selects) == 1, selects
        assert "library_audio_profiles" in selects[0]


class TestAudioIsNeverAGenerationSource:
    async def test_the_resolver_refuses_audio_with_its_own_code(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        audio = await hold_world.audio(user)
        job = await hold_world.job(user)
        output = await hold_world.output(user, job, content_type="audio/mpeg")

        for ref in (
            format_asset_ref(LibraryAssetSource.UPLOAD, audio.id),
            format_asset_ref(LibraryAssetSource.OUTPUT, output.id),
        ):
            async with hold_world.factory() as session:
                with pytest.raises(SourceMediaKindNotSupportedError) as raised:
                    await SourceMediaResolver().resolve(
                        [SourceMediaReference(asset_ref=ref)],
                        user_id=user.id,
                        session=session,
                        product_id="vex",
                    )
            assert raised.value.code == "media_kind_not_supported"
            assert isinstance(raised.value, SourceMediaValidationError)

    async def test_images_and_videos_still_resolve(self, hold_world: HoldWorld) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user)
        video = await hold_world.upload(user, content_type="video/mp4")

        async with hold_world.factory() as session:
            resolved = await SourceMediaResolver().resolve(
                [
                    SourceMediaReference(
                        asset_ref=format_asset_ref(LibraryAssetSource.UPLOAD, image.id)
                    ),
                    SourceMediaReference(
                        asset_ref=format_asset_ref(LibraryAssetSource.UPLOAD, video.id)
                    ),
                ],
                user_id=user.id,
                session=session,
                product_id="vex",
            )

        assert [r.media_kind for r in resolved] == [MediaKind.IMAGE, MediaKind.VIDEO]

    async def test_audio_is_not_a_frame_lineage_source(self, hold_world: HoldWorld) -> None:
        """Frame capture needs a *video* source: an audio source is the single public error."""
        from src.api.services.frame_lineage import FrameLineage, FrameLineageReason
        from src.api.services.user_content import UserContentFrameLineageError, UserContentService

        user = await hold_world.user()
        audio = await hold_world.audio(user)
        async with hold_world.factory() as session:
            service = UserContentService(
                MagicMock(), session, product_id="vex", media_ingestor=MagicMock()
            )
            with pytest.raises(UserContentFrameLineageError) as raised:
                await service._validate_frame_lineage(
                    user.id,
                    FrameLineage(
                        source=AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=audio.id),
                        timestamp_ms=0,
                    ),
                )

        assert raised.value.reason in {
            FrameLineageReason.SOURCE_UNAVAILABLE,
            FrameLineageReason.SOURCE_NOT_VIDEO,
        }


class TestDeleteAndBulkDeleteOfAudio:
    """Audio is deleted like any upload, and refused like any held one (invariant 5)."""

    async def test_bulk_delete_refuses_a_held_audio_asset_and_keeps_its_metadata_and_tags(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld
    ) -> None:
        from sqlalchemy import func, select

        from src.db.models.library import LibraryAssetMetadata, LibraryAssetTag

        user = await hold_world.user()
        held = await hold_world.audio(user)
        free = await hold_world.audio(user)
        ref = AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=held.id)
        async with hold_world.factory() as session:
            session.add(
                LibraryAssetMetadata(
                    id=uuid4(),
                    user_id=user.id,
                    product_id="vex",
                    asset_type="upload",
                    asset_id=held.id,
                    is_favorite=True,
                    display_title="Mix",
                )
            )
            await AssetHoldService(session, retention_days=RETENTION_DAYS).hold(
                ref,
                holder=AssetHolder(kind="studio.track_audio", id=uuid4()),
                user_id=user.id,
                product_id="vex",
                accept=AUDIO,
            )
            await session.commit()

        response = await client.post(
            "/v1/library/assets/bulk",
            headers=bearer(user.id),
            json={"type": "delete", "asset_refs": [f"upload:{held.id}", f"upload:{free.id}"]},
        )

        assert response.status_code == 201, response.text
        results = {item["asset_ref"]: item for item in response.json()["results"]}
        assert results[f"upload:{held.id}"]["success"] is False
        assert results[f"upload:{held.id}"]["error"] == "asset_held"
        assert results[f"upload:{free.id}"]["success"] is True
        assert await hold_world.upload_exists(held)
        assert not await hold_world.upload_exists(free)
        async with hold_world.factory() as session:
            metadata = (
                await session.execute(
                    select(func.count())
                    .select_from(LibraryAssetMetadata)
                    .where(LibraryAssetMetadata.asset_id == held.id)
                )
            ).scalar_one()
            tags = (
                await session.execute(
                    select(func.count())
                    .select_from(LibraryAssetTag)
                    .where(LibraryAssetTag.asset_id == held.id)
                )
            ).scalar_one()
            profiles = (
                await session.execute(
                    select(func.count())
                    .select_from(LibraryAudioProfile)
                    .where(LibraryAudioProfile.upload_id.in_([held.id, free.id]))
                )
            ).scalar_one()
        assert metadata == 1  # a refused delete changes nothing
        assert tags == 0
        assert profiles == 1  # the deleted asset's profile went with it; the held one's stayed

    async def test_single_delete_of_a_held_audio_asset_is_409_asset_held(
        self, client: AsyncTestClient[Litestar], hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        audio = await hold_world.audio(user)
        async with hold_world.factory() as session:
            await AssetHoldService(session, retention_days=RETENTION_DAYS).hold(
                AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=audio.id),
                holder=AssetHolder(kind="studio.track_audio", id=uuid4()),
                user_id=user.id,
                product_id="vex",
                accept=AUDIO,
            )
            await session.commit()

        response = await client.delete(
            f"/v1/library/assets/upload:{audio.id}", headers=bearer(user.id)
        )

        assert response.status_code == 409
        assert response.json()["error"] == "asset_held"
        assert response.json()["detail"] == {
            "holders": [{"kind": "studio.track_audio", "count": 1}]
        }
        assert await hold_world.upload_exists(audio)
