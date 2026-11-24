"""Track audio versions against real PostgreSQL (S2a W7 / W11.4, W11.5, W11.7).

The service runs on the savepoint-wrapped ``db_session`` (one transaction, like a request).
Every test ends with the reconciliation check: one hold per audio version, one audio version
per ``studio.track_audio`` hold, numbers unique and behind the counter (invariant 4).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import msgspec
import pytest
import pytest_asyncio
from sqlalchemy import event, select, text, update
from sqlalchemy.exc import IntegrityError

from src.api.schemas.studio import TrackAudioPatch
from src.api.services.content_proxy import ContentProxyService
from src.api.services.library import LibraryService
from src.api.services.library_holds import AssetHoldService
from src.core.enums import AssetDeleteOutcome, JobStatus, MediaKind
from src.core.enums import StudioSectionKind as K
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref
from src.db.models.library import LibraryAssetHold, LibraryAudioProfile
from src.db.models.storage import UserImage
from src.db.models.studio import TRACK_AUDIO_HOLDER_KIND, StudioTrack, StudioTrackAudio
from src.studio.constraints import MAX_AUDIO_VERSIONS_PER_TRACK
from src.studio.errors import (
    AssetNotEligible,
    AssetNotFound,
    AudioVersionIsCurrent,
    AudioVersionLimitReached,
    AudioVersionNotFound,
    InvalidCursor,
    StudioDocumentInvalid,
    StudioNotFound,
    VersionHasAudio,
    VersionKindMismatch,
    VersionNotFound,
)
from tests.library_holds_support import (
    RETENTION_DAYS,
    make_audio_upload,
    make_job,
    make_output,
    make_upload,
)
from tests.studio_support import make_lyrics, make_plan, make_section
from tests.track_audio_support import assert_audio_reconciled, attach_body

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

    from tests.studio_world import StudioWorld


@pytest_asyncio.fixture
async def world(enabled_studio: StudioWorld) -> AsyncGenerator[StudioWorld]:
    """An enabled studio; the reconciliation check runs after the test (invariant 4)."""
    yield enabled_studio
    await assert_audio_reconciled(enabled_studio.session)


@pytest.fixture
def content_proxy() -> ContentProxyService:
    storage = MagicMock()
    storage.delete = AsyncMock()
    return ContentProxyService(storage=storage, settings=SimpleNamespace(content_url_ttl=60))  # type: ignore[arg-type]


async def _track(w: StudioWorld, title: str = "Track") -> UUID:
    return await w.new_track(title)


async def _audio(w: StudioWorld, **kwargs: object) -> UserImage:
    return await make_audio_upload(w.session, w.user, **kwargs)  # type: ignore[arg-type]


async def _attach(
    w: StudioWorld,
    track_id: UUID,
    asset: UserImage | None = None,
    **fields: object,
) -> UUID:
    asset = asset or await _audio(w)
    out = await w.audio.create(track_id, w.user_id, w.product_id, attach_body(asset, **fields))  # type: ignore[arg-type]
    return out.id


async def _numbers(w: StudioWorld, track_id: UUID) -> list[int]:
    page = await w.audio.list_page(track_id, w.user_id, w.product_id, limit=50, cursor=None)
    return [item.version_number for item in page.items]


async def _holds_of(session: AsyncSession, kind: str = TRACK_AUDIO_HOLDER_KIND) -> int:
    return (
        await session.execute(
            select(text("count(*)"))
            .select_from(LibraryAssetHold)
            .where(LibraryAssetHold.holder_kind == kind)
        )
    ).scalar_one()


async def _expires_at(session: AsyncSession, image: UserImage) -> datetime:
    return (
        await session.execute(
            select(UserImage.expires_at)
            .where(UserImage.id == image.id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


class TestCreate:
    async def test_it_pins_an_audio_upload_and_reports_it(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        asset = await _audio(world, duration_ms=187_500)

        out = await world.audio.create(
            track_id, world.user_id, world.product_id, attach_body(asset, label="  Mix   one ")
        )

        assert out.version_number == 1
        assert out.track_id == track_id
        assert out.label == "Mix one"
        assert out.origin.value == "upload"
        assert out.duration_ms == 187_500
        assert out.is_current is False
        assert out.plan_version_id is None
        assert out.lyrics_version_id is None
        assert out.asset.asset_ref == format_asset_ref(LibraryAssetSource.UPLOAD, asset.id)
        assert out.asset.media_type is MediaKind.AUDIO
        assert out.asset.original.content_type == "audio/flac"
        assert out.asset.original.width is None
        assert out.asset.original.height is None
        assert out.asset.original.duration_ms == 187_500
        assert out.asset.variants == []
        hold = (
            await world.session.execute(
                select(LibraryAssetHold).where(LibraryAssetHold.holder_id == out.id)
            )
        ).scalar_one()
        assert hold.holder_kind == TRACK_AUDIO_HOLDER_KIND
        assert hold.upload_id == asset.id

    async def test_an_audio_output_of_a_completed_job_can_be_attached_too(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        job = await make_job(world.session, world.user)
        output = await make_output(world.session, world.user, job, content_type="audio/mpeg")
        output.duration_ms = 61_000
        await world.session.flush()

        out = await world.audio.create(
            track_id, world.user_id, world.product_id, attach_body(output)
        )

        assert out.duration_ms == 61_000
        assert out.asset.asset_ref == format_asset_ref(LibraryAssetSource.OUTPUT, output.id)

    async def test_numbers_are_gap_free_and_never_reused_after_a_delete(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        first = await _attach(world, track_id)
        second = await _attach(world, track_id)
        third = await _attach(world, track_id)

        await world.audio.delete(second, world.user_id, world.product_id)
        fourth = await _attach(world, track_id)
        await world.audio.delete(fourth, world.user_id, world.product_id)
        fifth = await _attach(world, track_id)

        assert await _numbers(world, track_id) == [5, 3, 1]
        assert {first, third, fifth} == {
            item.id
            for item in (
                await world.audio.list_page(
                    track_id, world.user_id, world.product_id, limit=50, cursor=None
                )
            ).items
        }

    async def test_numbers_are_per_track(self, world: StudioWorld) -> None:
        one, two = await _track(world, "One"), await _track(world, "Two")

        await _attach(world, one)
        await _attach(world, one)
        await _attach(world, two)

        assert await _numbers(world, one) == [2, 1]
        assert await _numbers(world, two) == [1]

    async def test_a_refused_attach_burns_no_number(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        image = await make_upload(world.session, world.user)

        with pytest.raises(AssetNotEligible):
            await world.audio.create(track_id, world.user_id, world.product_id, attach_body(image))
        number = (
            await world.session.execute(
                select(StudioTrack.next_audio_version_number)
                .where(StudioTrack.id == track_id)
                .execution_options(populate_existing=True)
            )
        ).scalar_one()
        await _attach(world, track_id)

        assert number == 1
        assert await _numbers(world, track_id) == [1]

    async def test_one_asset_on_one_track_twice_is_two_versions_with_two_holds(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        asset = await _audio(world)

        await _attach(world, track_id, asset)
        await _attach(world, track_id, asset)

        assert await _numbers(world, track_id) == [2, 1]
        assert await _holds_of(world.session) == 2

    async def test_the_same_asset_on_two_tracks_has_two_holds_and_keeps_the_asset_held(
        self, world: StudioWorld
    ) -> None:
        one, two = await _track(world, "One"), await _track(world, "Two")
        asset = await _audio(world)
        first = await _attach(world, one, asset)
        await _attach(world, two, asset)

        await world.audio.delete(first, world.user_id, world.product_id)

        holds = AssetHoldService(world.session, retention_days=RETENTION_DAYS)
        ref = AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=asset.id)
        counts = await holds.holder_counts(ref)
        assert [(c.kind, c.count) for c in counts] == [(TRACK_AUDIO_HOLDER_KIND, 1)]
        assert await holds.held_refs([ref], user_id=world.user_id, product_id="vex") == {ref}

    async def test_a_label_is_optional(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        out = await world.audio.create(
            track_id, world.user_id, world.product_id, attach_body(await _audio(world))
        )
        assert out.label is None


class TestCap:
    async def test_the_twenty_first_version_is_refused_with_nothing_changed(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        for _ in range(MAX_AUDIO_VERSIONS_PER_TRACK):
            await _attach(world, track_id)
        extra = await _audio(world)

        with pytest.raises(AudioVersionLimitReached) as raised:
            await world.audio.create(track_id, world.user_id, world.product_id, attach_body(extra))

        assert raised.value.detail == {"limit": 20}
        assert len(await _numbers(world, track_id)) == 20
        assert await _holds_of(world.session) == 20  # the refused asset was never pinned

    async def test_deleting_one_makes_room_and_the_number_keeps_counting(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        ids = [await _attach(world, track_id) for _ in range(MAX_AUDIO_VERSIONS_PER_TRACK)]

        await world.audio.delete(ids[3], world.user_id, world.product_id)
        await _attach(world, track_id)

        numbers = await _numbers(world, track_id)
        assert len(numbers) == 20
        assert numbers[0] == 21
        assert 4 not in numbers

    async def test_the_cap_is_per_track(self, world: StudioWorld) -> None:
        one, two = await _track(world, "One"), await _track(world, "Two")
        for _ in range(MAX_AUDIO_VERSIONS_PER_TRACK):
            await _attach(world, one)

        await _attach(world, two)

        assert await _numbers(world, two) == [1]


class TestRefusedAssets:
    async def test_an_image_is_not_accepted_by_an_audio_holder(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        image = await make_upload(world.session, world.user)

        with pytest.raises(AssetNotEligible) as raised:
            await world.audio.create(track_id, world.user_id, world.product_id, attach_body(image))

        assert raised.value.detail == {"reason": "media_kind_not_accepted"}

    async def test_a_video_is_not_accepted_either(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        video = await make_upload(world.session, world.user, content_type="video/mp4")

        with pytest.raises(AssetNotEligible) as raised:
            await world.audio.create(track_id, world.user_id, world.product_id, attach_body(video))

        assert raised.value.detail == {"reason": "media_kind_not_accepted"}

    async def test_a_derivative_is_not_eligible(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        parent = await _audio(world)
        derivative = await make_upload(
            world.session, world.user, is_thumbnail=True, parent=parent, content_type="audio/flac"
        )

        with pytest.raises(AssetNotEligible) as raised:
            await world.audio.create(
                track_id, world.user_id, world.product_id, attach_body(derivative)
            )

        assert raised.value.detail == {"reason": "derivative"}

    async def test_an_output_of_an_unfinished_job_is_not_eligible(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        job = await make_job(world.session, world.user, status=JobStatus.RUNNING)
        output = await make_output(world.session, world.user, job, content_type="audio/mpeg")

        with pytest.raises(AssetNotEligible) as raised:
            await world.audio.create(track_id, world.user_id, world.product_id, attach_body(output))

        assert raised.value.detail == {"reason": "job_not_available"}

    async def test_someone_else_s_asset_is_not_found(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        from src.db.models.user import User

        other = User(
            id=uuid4(), email=f"o-{uuid4().hex}@example.com", password_hash="x", product_id="vex"
        )
        world.session.add(other)
        await world.session.flush()
        theirs = await make_audio_upload(world.session, other)

        with pytest.raises(AssetNotFound):
            await world.audio.create(track_id, world.user_id, world.product_id, attach_body(theirs))

    async def test_an_asset_of_another_product_is_not_found(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        foreign = await make_audio_upload(world.session, world.user, product_id="synthara")

        with pytest.raises(AssetNotFound):
            await world.audio.create(
                track_id, world.user_id, world.product_id, attach_body(foreign)
            )

    async def test_an_unknown_asset_is_not_found(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        ghost = AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=uuid4())

        with pytest.raises(AssetNotFound):
            await world.audio.create(track_id, world.user_id, world.product_id, attach_body(ghost))

    @pytest.mark.parametrize(
        "raw", ["", "nope", f"image:{uuid4()}", "upload:not-a-uuid", "upload:"]
    )
    async def test_a_malformed_asset_ref_is_a_document_issue(
        self, world: StudioWorld, raw: str
    ) -> None:
        track_id = await _track(world)

        with pytest.raises(StudioDocumentInvalid) as raised:
            await world.audio.create(
                track_id,
                world.user_id,
                world.product_id,
                msgspec.structs.replace(
                    attach_body(AssetRef(LibraryAssetSource.UPLOAD, uuid4())), asset=raw
                ),
            )

        assert [(i.path, i.code) for i in raised.value.issues] == [("asset", "invalid_asset_ref")]

    async def test_an_unknown_or_foreign_track_is_not_found(self, world: StudioWorld) -> None:
        asset = await _audio(world)

        with pytest.raises(StudioNotFound):
            await world.audio.create(uuid4(), world.user_id, world.product_id, attach_body(asset))
        with pytest.raises(StudioNotFound):
            await world.audio.create(
                await _track(world), uuid4(), world.product_id, attach_body(asset)
            )


class TestLabel:
    async def test_an_empty_label_after_normalization_is_refused(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        with pytest.raises(StudioDocumentInvalid) as raised:
            await world.audio.create(
                track_id,
                world.user_id,
                world.product_id,
                attach_body(await _audio(world), label="   \t "),
            )
        assert [(i.path, i.code) for i in raised.value.issues] == [
            ("label", "empty_after_normalization")
        ]

    async def test_a_label_that_grows_past_the_limit_when_normalized_is_refused(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        with pytest.raises(StudioDocumentInvalid) as raised:
            await world.audio.create(
                track_id,
                world.user_id,
                world.product_id,
                attach_body(await _audio(world), label="क़" * 31),  # 31 -> 62 code points
            )
        assert [(i.path, i.code) for i in raised.value.issues] == [
            ("label", "too_long_after_normalization")
        ]
        assert await _holds_of(world.session) == 0

    async def test_a_label_of_exactly_sixty_characters_is_kept(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        out = await world.audio.create(
            track_id,
            world.user_id,
            world.product_id,
            attach_body(await _audio(world), label="x" * 60),
        )
        assert out.label == "x" * 60


class TestRealizedVersionLinks:
    async def _plan_and_lyrics(self, w: StudioWorld, track_id: UUID) -> tuple[UUID, UUID]:
        verse = make_section(K.VERSE)
        plan, _ = await w.plan(track_id, make_plan(verse))
        lyrics, _ = await w.lyrics(track_id, make_lyrics((verse.id, ["la"])), plan.version.id)
        return plan.version.id, lyrics.version.id

    async def test_both_links_are_stored(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        plan_id, lyrics_id = await self._plan_and_lyrics(world, track_id)

        out = await world.audio.create(
            track_id,
            world.user_id,
            world.product_id,
            attach_body(await _audio(world), plan_version_id=plan_id, lyrics_version_id=lyrics_id),
        )

        assert (out.plan_version_id, out.lyrics_version_id) == (plan_id, lyrics_id)

    async def test_another_track_s_version_is_not_found(self, world: StudioWorld) -> None:
        mine, theirs = await _track(world, "Mine"), await _track(world, "Theirs")
        other_plan, _ = await self._plan_and_lyrics(world, theirs)

        with pytest.raises(VersionNotFound):
            await world.audio.create(
                mine,
                world.user_id,
                world.product_id,
                attach_body(await _audio(world), plan_version_id=other_plan),
            )
        assert await _holds_of(world.session) == 0

    async def test_an_unknown_version_is_not_found(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        with pytest.raises(VersionNotFound):
            await world.audio.create(
                track_id,
                world.user_id,
                world.product_id,
                attach_body(await _audio(world), lyrics_version_id=uuid4()),
            )

    async def test_a_lyrics_version_in_the_plan_slot_is_a_kind_mismatch(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        _, lyrics_id = await self._plan_and_lyrics(world, track_id)

        with pytest.raises(VersionKindMismatch) as raised:
            await world.audio.create(
                track_id,
                world.user_id,
                world.product_id,
                attach_body(await _audio(world), plan_version_id=lyrics_id),
            )

        assert raised.value.detail == {"field": "plan_version_id", "expected": "plan"}

    async def test_a_plan_version_in_the_lyrics_slot_is_a_kind_mismatch(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        plan_id, _ = await self._plan_and_lyrics(world, track_id)

        with pytest.raises(VersionKindMismatch) as raised:
            await world.audio.create(
                track_id,
                world.user_id,
                world.product_id,
                attach_body(await _audio(world), lyrics_version_id=plan_id),
            )

        assert raised.value.detail == {"field": "lyrics_version_id", "expected": "lyrics"}

    async def test_the_database_refuses_another_track_s_version_whatever_the_service_says(
        self, world: StudioWorld
    ) -> None:
        """The composite foreign key is the backstop of the service check."""
        mine, theirs = await _track(world, "Mine"), await _track(world, "Theirs")
        other_plan, _ = await self._plan_and_lyrics(world, theirs)
        audio_id = await _attach(world, mine)

        with pytest.raises(IntegrityError, match="fk_studio_track_audio_plan_version"):
            async with world.session.begin_nested():
                await world.session.execute(
                    update(StudioTrackAudio)
                    .where(StudioTrackAudio.id == audio_id)
                    .values(plan_version_id=other_plan)
                )


class TestPatch:
    async def test_the_label_is_set_normalized_and_cleared(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        audio_id = await _attach(world, track_id, label="old")

        set_ = await world.audio.patch(
            audio_id, world.user_id, world.product_id, TrackAudioPatch(label="  New   name ")
        )
        cleared = await world.audio.patch(
            audio_id, world.user_id, world.product_id, TrackAudioPatch(label=None)
        )

        assert set_.label == "New name"
        assert cleared.label is None

    async def test_an_omitted_field_keeps_its_value(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        plan, _ = await world.plan(track_id, make_plan(make_section(K.VERSE)))
        audio_id = await _attach(world, track_id, label="keep", plan_version_id=plan.version.id)

        out = await world.audio.patch(audio_id, world.user_id, world.product_id, TrackAudioPatch())

        assert (out.label, out.plan_version_id) == ("keep", plan.version.id)

    async def test_links_are_set_and_cleared(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        verse = make_section(K.VERSE)
        plan, _ = await world.plan(track_id, make_plan(verse))
        lyrics, _ = await world.lyrics(track_id, make_lyrics((verse.id, ["la"])), plan.version.id)
        audio_id = await _attach(world, track_id)

        linked = await world.audio.patch(
            audio_id,
            world.user_id,
            world.product_id,
            TrackAudioPatch(plan_version_id=plan.version.id, lyrics_version_id=lyrics.version.id),
        )
        cleared = await world.audio.patch(
            audio_id,
            world.user_id,
            world.product_id,
            TrackAudioPatch(plan_version_id=None, lyrics_version_id=None),
        )

        assert (linked.plan_version_id, linked.lyrics_version_id) == (
            plan.version.id,
            lyrics.version.id,
        )
        assert (cleared.plan_version_id, cleared.lyrics_version_id) == (None, None)

    async def test_a_link_is_checked_like_on_create(self, world: StudioWorld) -> None:
        mine, theirs = await _track(world, "Mine"), await _track(world, "Theirs")
        other_plan, _ = await world.plan(theirs, make_plan(make_section(K.VERSE)))
        own_plan, _ = await world.plan(mine, make_plan(make_section(K.VERSE)))
        audio_id = await _attach(world, mine)

        with pytest.raises(VersionNotFound):
            await world.audio.patch(
                audio_id,
                world.user_id,
                world.product_id,
                TrackAudioPatch(plan_version_id=other_plan.version.id),
            )
        with pytest.raises(VersionKindMismatch):
            await world.audio.patch(
                audio_id,
                world.user_id,
                world.product_id,
                TrackAudioPatch(lyrics_version_id=own_plan.version.id),
            )

    async def test_the_label_is_validated_after_normalization(self, world: StudioWorld) -> None:
        audio_id = await _attach(world, await _track(world))

        with pytest.raises(StudioDocumentInvalid):
            await world.audio.patch(
                audio_id, world.user_id, world.product_id, TrackAudioPatch(label="   ")
            )
        with pytest.raises(StudioDocumentInvalid):
            await world.audio.patch(
                audio_id, world.user_id, world.product_id, TrackAudioPatch(label="क़" * 31)
            )

    async def test_a_foreign_or_unknown_version_is_not_found(self, world: StudioWorld) -> None:
        with pytest.raises(AudioVersionNotFound):
            await world.audio.patch(uuid4(), world.user_id, world.product_id, TrackAudioPatch())
        audio_id = await _attach(world, await _track(world))
        with pytest.raises(AudioVersionNotFound):
            await world.audio.patch(audio_id, uuid4(), world.product_id, TrackAudioPatch())

    async def test_the_asset_is_immutable_there_is_no_way_to_change_it(
        self, world: StudioWorld
    ) -> None:
        """No ``asset`` field on the patch Struct (the wire refuses it); the hold never moves."""
        asset = await _audio(world)
        audio_id = await _attach(world, await _track(world), asset)
        before = (
            await world.session.execute(
                select(StudioTrackAudio.hold_id).where(StudioTrackAudio.id == audio_id)
            )
        ).scalar_one()

        await world.audio.patch(
            audio_id, world.user_id, world.product_id, TrackAudioPatch(label="renamed")
        )

        assert "asset" not in TrackAudioPatch.__struct_fields__
        after = (
            await world.session.execute(
                select(StudioTrackAudio.hold_id).where(StudioTrackAudio.id == audio_id)
            )
        ).scalar_one()
        assert after == before


class TestReads:
    async def test_the_list_is_newest_version_first_with_keyset_paging(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        for _ in range(5):
            await _attach(world, track_id)

        first = await world.audio.list_page(
            track_id, world.user_id, world.product_id, limit=2, cursor=None
        )
        second = await world.audio.list_page(
            track_id, world.user_id, world.product_id, limit=2, cursor=first.next_cursor
        )
        third = await world.audio.list_page(
            track_id, world.user_id, world.product_id, limit=2, cursor=second.next_cursor
        )

        assert [i.version_number for i in first.items] == [5, 4]
        assert [i.version_number for i in second.items] == [3, 2]
        assert [i.version_number for i in third.items] == [1]
        assert (first.has_more, second.has_more, third.has_more) == (True, True, False)
        assert third.next_cursor is None

    async def test_paging_survives_a_delete_between_pages(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        ids = [await _attach(world, track_id) for _ in range(4)]
        first = await world.audio.list_page(
            track_id, world.user_id, world.product_id, limit=2, cursor=None
        )

        await world.audio.delete(ids[2], world.user_id, world.product_id)  # number 3, already seen
        rest = await world.audio.list_page(
            track_id, world.user_id, world.product_id, limit=2, cursor=first.next_cursor
        )

        assert [i.version_number for i in rest.items] == [2, 1]

    @pytest.mark.parametrize("cursor", ["nope", "e30=", "eyJ2ZXJzaW9uX251bWJlciI6IDB9"])
    async def test_a_bad_cursor_is_refused(self, world: StudioWorld, cursor: str) -> None:
        track_id = await _track(world)
        with pytest.raises(InvalidCursor):
            await world.audio.list_page(
                track_id, world.user_id, world.product_id, limit=5, cursor=cursor
            )

    async def test_the_list_of_an_unknown_or_foreign_track_is_not_found(
        self, world: StudioWorld
    ) -> None:
        with pytest.raises(StudioNotFound):
            await world.audio.list_page(
                uuid4(), world.user_id, world.product_id, limit=5, cursor=None
            )
        with pytest.raises(StudioNotFound):
            await world.audio.list_page(
                await _track(world), uuid4(), world.product_id, limit=5, cursor=None
            )

    async def test_get_returns_one_and_hides_foreign_ones(self, world: StudioWorld) -> None:
        audio_id = await _attach(world, await _track(world), label="solo")

        out = await world.audio.get(audio_id, world.user_id, world.product_id)

        assert (out.id, out.label, out.version_number) == (audio_id, "solo", 1)
        with pytest.raises(AudioVersionNotFound):
            await world.audio.get(audio_id, uuid4(), world.product_id)
        with pytest.raises(AudioVersionNotFound):
            await world.audio.get(audio_id, world.user_id, "synthara")
        with pytest.raises(AudioVersionNotFound):
            await world.audio.get(uuid4(), world.user_id, world.product_id)

    async def test_the_assets_of_a_page_are_built_in_batched_queries_not_one_per_item(
        self, world: StudioWorld, db_engine: object
    ) -> None:
        track_id = await _track(world)
        for _ in range(3):
            await _attach(world, track_id)
        small = await self._statements(world, db_engine, track_id, 3)
        for _ in range(9):
            await _attach(world, track_id)

        large = await self._statements(world, db_engine, track_id, 12)

        assert large == small, "the statement count grew with the number of items"

    @staticmethod
    async def _statements(world: StudioWorld, engine: object, track_id: UUID, expected: int) -> int:
        count = 0

        def on_execute(*_: object) -> None:
            nonlocal count
            count += 1

        sync_engine = engine.sync_engine  # type: ignore[attr-defined]
        event.listen(sync_engine, "before_cursor_execute", on_execute)
        try:
            page = await world.audio.list_page(
                track_id, world.user_id, world.product_id, limit=50, cursor=None
            )
        finally:
            event.remove(sync_engine, "before_cursor_execute", on_execute)
        assert len(page.items) == expected
        return count


class TestCurrentPointer:
    async def test_it_is_set_reported_and_cleared(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        audio_id = await _attach(world, track_id)

        track = await world.versions.set_current(
            track_id,
            world.user_id,
            world.product_id,
            plan_version_id=msgspec.UNSET,
            lyrics_version_id=msgspec.UNSET,
            audio_version_id=audio_id,
        )
        detail = await world.tracks.get_detail(track_id, world.user_id, world.product_id)
        listed = await world.audio.get(audio_id, world.user_id, world.product_id)
        cleared = await world.versions.set_current(
            track_id,
            world.user_id,
            world.product_id,
            plan_version_id=msgspec.UNSET,
            lyrics_version_id=msgspec.UNSET,
            audio_version_id=None,
        )

        assert track.current_audio_version_id == audio_id
        assert detail.current_audio_version_id == audio_id
        assert detail.current_audio is not None
        assert detail.current_audio.id == audio_id
        assert detail.current_audio.is_current is True
        assert detail.current_audio.asset.media_type is MediaKind.AUDIO
        assert listed.is_current is True
        assert cleared.current_audio_version_id is None
        again = await world.tracks.get_detail(track_id, world.user_id, world.product_id)
        assert again.current_audio is None

    async def test_a_track_without_audio_has_no_current_audio(self, world: StudioWorld) -> None:
        detail = await world.tracks.get_detail(await _track(world), world.user_id, world.product_id)
        assert detail.current_audio is None
        assert detail.current_audio_version_id is None

    async def test_another_track_s_audio_is_not_found(self, world: StudioWorld) -> None:
        mine, theirs = await _track(world, "Mine"), await _track(world, "Theirs")
        other = await _attach(world, theirs)

        with pytest.raises(AudioVersionNotFound):
            await world.versions.set_current(
                mine,
                world.user_id,
                world.product_id,
                plan_version_id=msgspec.UNSET,
                lyrics_version_id=msgspec.UNSET,
                audio_version_id=other,
            )

    async def test_an_unknown_id_is_not_found(self, world: StudioWorld) -> None:
        with pytest.raises(AudioVersionNotFound):
            await world.versions.set_current(
                await _track(world),
                world.user_id,
                world.product_id,
                plan_version_id=msgspec.UNSET,
                lyrics_version_id=msgspec.UNSET,
                audio_version_id=uuid4(),
            )

    async def test_the_database_refuses_another_track_s_audio_as_the_pointer(
        self, world: StudioWorld
    ) -> None:
        mine, theirs = await _track(world, "Mine"), await _track(world, "Theirs")
        other = await _attach(world, theirs)

        with pytest.raises(IntegrityError, match="fk_studio_tracks_current_audio_version"):
            async with world.session.begin_nested():
                await world.session.execute(
                    update(StudioTrack)
                    .where(StudioTrack.id == mine)
                    .values(current_audio_version_id=other)
                )

    async def test_setting_the_audio_leaves_the_plan_and_lyrics_pair_alone(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        verse = make_section(K.VERSE)
        plan, _ = await world.plan(track_id, make_plan(verse))
        lyrics, _ = await world.lyrics(track_id, make_lyrics((verse.id, ["la"])), plan.version.id)
        audio_id = await _attach(world, track_id)

        track = await world.versions.set_current(
            track_id,
            world.user_id,
            world.product_id,
            plan_version_id=msgspec.UNSET,
            lyrics_version_id=msgspec.UNSET,
            audio_version_id=audio_id,
        )

        assert track.current_plan_version_id == plan.version.id
        assert track.current_lyrics_version_id == lyrics.version.id
        assert track.current_audio_version_id == audio_id


class TestDelete:
    async def test_the_current_audio_cannot_be_deleted(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        audio_id = await _attach(world, track_id)
        await world.versions.set_current(
            track_id,
            world.user_id,
            world.product_id,
            plan_version_id=msgspec.UNSET,
            lyrics_version_id=msgspec.UNSET,
            audio_version_id=audio_id,
        )

        with pytest.raises(AudioVersionIsCurrent):
            await world.audio.delete(audio_id, world.user_id, world.product_id)

        assert await _numbers(world, track_id) == [1]
        assert await _holds_of(world.session) == 1

    async def test_after_clearing_the_pointer_it_can_be_deleted(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        audio_id = await _attach(world, track_id)
        for target in (audio_id, None):
            await world.versions.set_current(
                track_id,
                world.user_id,
                world.product_id,
                plan_version_id=msgspec.UNSET,
                lyrics_version_id=msgspec.UNSET,
                audio_version_id=target,
            )

        await world.audio.delete(audio_id, world.user_id, world.product_id)

        assert await _numbers(world, track_id) == []

    async def test_deleting_another_version_releases_its_hold_and_the_last_release_rearms(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        asset = await _audio(world, expires_at=datetime.now(UTC) - timedelta(days=3))
        audio_id = await _attach(world, track_id, asset)
        assert await _holds_of(world.session) == 1

        await world.audio.delete(audio_id, world.user_id, world.product_id)

        assert await _holds_of(world.session) == 0
        # the retention clock restarts once, from now, when the last hold goes
        restarted = await _expires_at(world.session, asset)
        assert restarted > datetime.now(UTC) + timedelta(days=RETENTION_DAYS - 1)

    async def test_with_a_second_holder_the_asset_stays_held_and_is_not_rearmed(
        self, world: StudioWorld
    ) -> None:
        one, two = await _track(world, "One"), await _track(world, "Two")
        past = datetime.now(UTC) - timedelta(days=3)
        asset = await _audio(world, expires_at=past)
        first = await _attach(world, one, asset)
        await _attach(world, two, asset)

        await world.audio.delete(first, world.user_id, world.product_id)

        assert await _holds_of(world.session) == 1
        assert await _expires_at(world.session, asset) == past

    async def test_a_foreign_or_unknown_version_is_not_found(self, world: StudioWorld) -> None:
        audio_id = await _attach(world, await _track(world))
        with pytest.raises(AudioVersionNotFound):
            await world.audio.delete(uuid4(), world.user_id, world.product_id)
        with pytest.raises(AudioVersionNotFound):
            await world.audio.delete(audio_id, uuid4(), world.product_id)
        assert await _holds_of(world.session) == 1

    async def test_an_attached_asset_cannot_be_deleted_from_the_library_and_is_left_intact(
        self, world: StudioWorld, content_proxy: ContentProxyService
    ) -> None:
        asset = await _audio(world)
        await _attach(world, await _track(world), asset)
        ref = format_asset_ref(LibraryAssetSource.UPLOAD, asset.id)

        outcome = await LibraryService(world.session).delete_asset(
            ref, world.user_id, "vex", session=world.session, content_proxy=content_proxy
        )

        assert outcome is AssetDeleteOutcome.HELD
        profile = (
            await world.session.execute(
                select(LibraryAudioProfile).where(LibraryAudioProfile.upload_id == asset.id)
            )
        ).scalar_one()
        assert len(profile.peaks) == 2048

    async def test_once_the_version_is_gone_the_asset_can_be_deleted_again(
        self, world: StudioWorld, content_proxy: ContentProxyService
    ) -> None:
        asset = await _audio(world)
        audio_id = await _attach(world, await _track(world), asset)
        await world.audio.delete(audio_id, world.user_id, world.product_id)

        outcome = await LibraryService(world.session).delete_asset(
            format_asset_ref(LibraryAssetSource.UPLOAD, asset.id),
            world.user_id,
            "vex",
            session=world.session,
            content_proxy=content_proxy,
        )

        assert outcome is AssetDeleteOutcome.DELETED
        assert (
            await world.session.execute(
                select(LibraryAudioProfile.id).where(LibraryAudioProfile.upload_id == asset.id)
            )
        ).scalar_one_or_none() is None  # the profile went with the asset (cascade)


class TestTrackDelete:
    async def test_it_releases_every_audio_hold_and_leaves_no_orphan(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        other = await _track(world, "Other")
        assets = [
            await _audio(world, expires_at=datetime.now(UTC) - timedelta(days=1)) for _ in range(3)
        ]
        for asset in assets:
            await _attach(world, track_id, asset)
        kept = await _attach(world, other)
        current = await _attach(world, track_id)
        await world.versions.set_current(
            track_id,
            world.user_id,
            world.product_id,
            plan_version_id=msgspec.UNSET,
            lyrics_version_id=msgspec.UNSET,
            audio_version_id=current,
        )
        assert await _holds_of(world.session) == 5

        await world.tracks.delete(track_id, world.user_id, world.product_id)

        assert await _holds_of(world.session) == 1  # only the other track's
        remaining = (await world.session.execute(select(StudioTrackAudio.id))).scalars().all()
        assert list(remaining) == [kept]
        for asset in assets:
            assert await _expires_at(world.session, asset) > datetime.now(UTC) + timedelta(days=1)
        # the reconciliation after the test proves no orphan hold remains

    async def test_the_released_assets_can_be_deleted_afterwards(
        self, world: StudioWorld, content_proxy: ContentProxyService
    ) -> None:
        track_id = await _track(world)
        asset = await _audio(world)
        await _attach(world, track_id, asset)

        await world.tracks.delete(track_id, world.user_id, world.product_id)

        outcome = await LibraryService(world.session).delete_asset(
            format_asset_ref(LibraryAssetSource.UPLOAD, asset.id),
            world.user_id,
            "vex",
            session=world.session,
            content_proxy=content_proxy,
        )
        assert outcome is AssetDeleteOutcome.DELETED

    async def test_a_track_without_audio_deletes_as_before(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        await world.tracks.delete(track_id, world.user_id, world.product_id)
        with pytest.raises(StudioNotFound):
            await world.tracks.get_detail(track_id, world.user_id, world.product_id)


class TestDocumentVersionDelete:
    async def _two_versions(self, w: StudioWorld, track_id: UUID) -> tuple[UUID, UUID, UUID]:
        verse = make_section(K.VERSE)
        first, _ = await w.plan(track_id, make_plan(verse, title="One"))
        second, _ = await w.plan(
            track_id, make_plan(verse, title="Two"), parent=None, make_current=True
        )
        lyrics, _ = await w.lyrics(
            track_id, make_lyrics((verse.id, ["la"])), second.version.id, make_current=True
        )
        return first.version.id, second.version.id, lyrics.version.id

    async def test_a_plan_version_that_audio_realizes_cannot_be_deleted(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        old_plan, _, _ = await self._two_versions(world, track_id)
        await _attach(world, track_id, plan_version_id=old_plan)
        await _attach(world, track_id, plan_version_id=old_plan)

        with pytest.raises(VersionHasAudio) as raised:
            await world.versions.delete(old_plan, world.user_id, world.product_id)

        assert raised.value.detail == {"count": 2}

    async def test_a_lyrics_version_that_audio_realizes_cannot_be_deleted(
        self, world: StudioWorld
    ) -> None:
        track_id = await _track(world)
        _, plan_id, lyrics_id = await self._two_versions(world, track_id)
        verse = make_section(K.VERSE)
        await world.versions.set_current(
            track_id,
            world.user_id,
            world.product_id,
            plan_version_id=msgspec.UNSET,
            lyrics_version_id=None,
        )
        await _attach(world, track_id, lyrics_version_id=lyrics_id)
        del verse, plan_id

        with pytest.raises(VersionHasAudio) as raised:
            await world.versions.delete(lyrics_id, world.user_id, world.product_id)

        assert raised.value.detail == {"count": 1}

    async def test_once_the_link_is_cleared_the_version_can_go(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        old_plan, _, _ = await self._two_versions(world, track_id)
        audio_id = await _attach(world, track_id, plan_version_id=old_plan)

        await world.audio.patch(
            audio_id, world.user_id, world.product_id, TrackAudioPatch(plan_version_id=None)
        )
        await world.versions.delete(old_plan, world.user_id, world.product_id)

        assert await world.version_count(track_id) == 2

    async def test_deleting_the_audio_also_unblocks_the_version(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        old_plan, _, _ = await self._two_versions(world, track_id)
        audio_id = await _attach(world, track_id, plan_version_id=old_plan)

        await world.audio.delete(audio_id, world.user_id, world.product_id)
        await world.versions.delete(old_plan, world.user_id, world.product_id)

    async def test_audio_links_on_another_version_do_not_block(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        old_plan, new_plan, _ = await self._two_versions(world, track_id)
        await _attach(world, track_id, plan_version_id=new_plan)

        await world.versions.delete(old_plan, world.user_id, world.product_id)


class TestDatabaseBackstops:
    """What the schema refuses on its own, whatever the service does."""

    async def test_two_profiles_for_one_upload_are_refused(self, world: StudioWorld) -> None:
        asset = await _audio(world)
        with pytest.raises(IntegrityError, match="uq_library_audio_profiles_upload"):
            async with world.session.begin_nested():
                world.session.add(
                    LibraryAudioProfile(
                        id=uuid4(),
                        upload_id=asset.id,
                        codec="flac",
                        sample_rate_hz=44100,
                        channels=2,
                        bit_rate=None,
                        peaks=bytes(2048),
                        peaks_version=1,
                    )
                )
                await world.session.flush()

    @pytest.mark.parametrize(
        ("overrides", "constraint"),
        [
            ({"peaks": bytes(2047)}, "ck_library_audio_profiles_peaks_size"),
            ({"peaks": bytes(2049)}, "ck_library_audio_profiles_peaks_size"),
            ({"channels": 0}, "ck_library_audio_profiles_channels"),
            ({"channels": 9}, "ck_library_audio_profiles_channels"),
            ({"upload_id": None}, "ck_library_audio_profiles_one_asset"),
        ],
    )
    async def test_the_profile_checks(
        self, world: StudioWorld, overrides: dict[str, object], constraint: str
    ) -> None:
        asset = await _audio(world, with_profile=False)
        values: dict[str, object] = {
            "id": uuid4(),
            "upload_id": asset.id,
            "codec": "flac",
            "sample_rate_hz": 44100,
            "channels": 2,
            "bit_rate": None,
            "peaks": bytes(2048),
            "peaks_version": 1,
        }
        values.update(overrides)

        with pytest.raises(IntegrityError, match=constraint):
            async with world.session.begin_nested():
                world.session.add(LibraryAudioProfile(**values))
                await world.session.flush()

    async def test_a_profile_cannot_name_both_arms(self, world: StudioWorld) -> None:
        asset = await _audio(world, with_profile=False)
        job = await make_job(world.session, world.user)
        output = await make_output(world.session, world.user, job, content_type="audio/mpeg")
        with pytest.raises(IntegrityError, match="ck_library_audio_profiles_one_asset"):
            async with world.session.begin_nested():
                world.session.add(
                    LibraryAudioProfile(
                        id=uuid4(),
                        upload_id=asset.id,
                        output_id=output.id,
                        codec="mp3",
                        sample_rate_hz=44100,
                        channels=2,
                        peaks=bytes(2048),
                        peaks_version=1,
                    )
                )
                await world.session.flush()

    async def test_the_origin_duration_and_number_checks(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        audio_id = await _attach(world, track_id)
        for column, value, constraint in (
            ("origin", "generated", "ck_studio_track_audio_origin"),
            ("duration_ms", 0, "ck_studio_track_audio_duration"),
            ("version_number", 0, "ck_studio_track_audio_number"),
        ):
            with pytest.raises(IntegrityError, match=constraint):
                async with world.session.begin_nested():
                    await world.session.execute(
                        update(StudioTrackAudio)
                        .where(StudioTrackAudio.id == audio_id)
                        .values({column: value})
                    )

    async def test_a_hold_belongs_to_one_version_at_most(self, world: StudioWorld) -> None:
        track_id = await _track(world)
        first = await _attach(world, track_id)
        second = await _attach(world, track_id)
        hold = (
            await world.session.execute(
                select(StudioTrackAudio.hold_id).where(StudioTrackAudio.id == first)
            )
        ).scalar_one()

        with pytest.raises(IntegrityError, match="studio_track_audio_hold_id_key"):
            async with world.session.begin_nested():
                await world.session.execute(
                    update(StudioTrackAudio)
                    .where(StudioTrackAudio.id == second)
                    .values(hold_id=hold)
                )

    async def test_a_held_asset_cannot_be_deleted_by_sql_either(self, world: StudioWorld) -> None:
        asset = await _audio(world)
        await _attach(world, await _track(world), asset)

        with pytest.raises(IntegrityError, match="fk_library_asset_holds_upload_id"):
            async with world.session.begin_nested():
                await world.session.execute(
                    text("DELETE FROM user_images WHERE id = :id").bindparams(id=asset.id)
                )
                await world.session.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
