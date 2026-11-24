"""``AssetHoldService`` and the ``library_asset_holds`` table (S1-P3, W2 and the invariants of P3-10).

These tests share one rolled-back transaction (``db_session``); everything that needs a second
connection (locks, the sweep, races) lives in ``test_library_holds_races.py``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from src.api.services.library_holds import (
    AssetHolder,
    AssetHoldService,
    AssetIneligibleReason,
    AssetNotEligibleError,
    AssetNotFoundError,
    HolderCount,
)
from src.core.enums import JobStatus, MediaKind
from src.core.library_ref import AssetRef, LibraryAssetSource
from src.db.integrity import violated_constraint
from src.db.models.library import (
    HOLD_OUTPUT_FK,
    HOLD_UPLOAD_FK,
    LibraryAssetHold,
)
from src.db.models.storage import GenerationOutput, UserImage
from src.db.repositories.library_hold import LockedAsset
from tests.library_holds_support import (
    AUDIO,
    IMAGES,
    future,
    make_audio_upload,
    make_job,
    make_output,
    make_upload,
    output_ref,
    past,
    upload_ref,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.user import User
    from tests.library_holds_support import HoldWorld

RETENTION_DAYS = 7
KIND = "test.holder"


@pytest.fixture
def holds(db_session: AsyncSession) -> AssetHoldService:
    return AssetHoldService(db_session, retention_days=RETENTION_DAYS)


@pytest.fixture
async def user(make_user) -> User:  # type: ignore[no-untyped-def]
    return await make_user(email=f"holds-{uuid4().hex}@example.com")


def holder(kind: str = KIND) -> AssetHolder:
    return AssetHolder(kind=kind, id=uuid4())


async def _upload_expiry(session: AsyncSession, image: UserImage):
    return (
        await session.execute(
            select(UserImage.expires_at)
            .where(UserImage.id == image.id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def _output_expiry(session: AsyncSession, output: GenerationOutput):
    return (
        await session.execute(
            select(GenerationOutput.expires_at)
            .where(GenerationOutput.id == output.id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


class TestRoundTrip:
    async def test_upload_hold_then_release(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        image = await make_upload(db_session, user)
        ref = upload_ref(image)

        hold_id = await holds.hold(
            ref, holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
        )

        assert await holds.holder_counts(ref) == (HolderCount(kind=KIND, count=1),)
        assert await holds.held_refs([ref], user_id=user.id, product_id="vex") == {ref}
        assert await holds.hold_refs([hold_id]) == {hold_id: ref}

        await holds.release([hold_id])

        assert await holds.holder_counts(ref) == ()
        assert await holds.held_refs([ref], user_id=user.id, product_id="vex") == frozenset()
        assert await holds.hold_refs([hold_id]) == {}

    async def test_output_hold_then_release(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        job = await make_job(db_session, user)
        output = await make_output(db_session, user, job)
        ref = output_ref(output)

        hold_id = await holds.hold(
            ref, holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
        )

        assert await holds.hold_refs([hold_id]) == {hold_id: ref}
        assert await holds.held_refs([ref], user_id=user.id, product_id="vex") == {ref}
        await holds.release([hold_id])
        assert await holds.held_refs([ref], user_id=user.id, product_id="vex") == frozenset()

    async def test_held_refs_answers_for_a_mixed_batch_in_two_statements(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        held_upload = await make_upload(db_session, user)
        free_upload = await make_upload(db_session, user)
        job = await make_job(db_session, user)
        held_output = await make_output(db_session, user, job)
        free_output = await make_output(db_session, user, job, output_index=1)
        await holds.hold(
            upload_ref(held_upload),
            holder=holder(),
            user_id=user.id,
            product_id="vex",
            accept=IMAGES,
        )
        await holds.hold(
            output_ref(held_output),
            holder=holder(),
            user_id=user.id,
            product_id="vex",
            accept=IMAGES,
        )

        refs = [
            upload_ref(held_upload),
            upload_ref(free_upload),
            output_ref(held_output),
            output_ref(free_output),
        ]
        held = await holds.held_refs(refs, user_id=user.id, product_id="vex")

        assert held == {upload_ref(held_upload), output_ref(held_output)}
        assert await holds.held_refs([], user_id=user.id, product_id="vex") == frozenset()

    async def test_holder_counts_group_by_kind_ordered_by_kind(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        image = await make_upload(db_session, user)
        ref = upload_ref(image)
        for kind in ("z.last", "a.first", "a.first"):
            await holds.hold(
                ref, holder=holder(kind), user_id=user.id, product_id="vex", accept=IMAGES
            )

        assert await holds.holder_counts(ref) == (
            HolderCount(kind="a.first", count=2),
            HolderCount(kind="z.last", count=1),
        )

    async def test_the_same_holder_cannot_hold_the_same_asset_twice(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        image = await make_upload(db_session, user)
        who = holder()
        await holds.hold(
            upload_ref(image), holder=who, user_id=user.id, product_id="vex", accept=IMAGES
        )

        # A programming error of the caller: raised, not swallowed (and not a 404).
        with pytest.raises(IntegrityError) as raised:
            await holds.hold(
                upload_ref(image), holder=who, user_id=user.id, product_id="vex", accept=IMAGES
            )

        assert "uq_library_asset_holds_upload_holder" in str(raised.value)
        # The savepoint kept the session usable.
        assert await holds.holder_counts(upload_ref(image)) == (HolderCount(KIND, 1),)


class TestRearm:
    async def test_only_the_last_release_restarts_the_clock(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        image = await make_upload(db_session, user, expires_at=past(hours=3))
        expired_at = await _upload_expiry(db_session, image)
        first = await holds.hold(
            upload_ref(image),
            holder=holder("a.one"),
            user_id=user.id,
            product_id="vex",
            accept=IMAGES,
        )
        second = await holds.hold(
            upload_ref(image),
            holder=holder("b.two"),
            user_id=user.id,
            product_id="vex",
            accept=IMAGES,
        )

        await holds.release([first])
        assert await _upload_expiry(db_session, image) == expired_at  # one holder is left

        await holds.release([second])
        db_now = (await db_session.execute(select(func.now()))).scalar_one()
        assert await _upload_expiry(db_session, image) >= db_now + timedelta(days=RETENTION_DAYS)

    async def test_an_output_is_rearmed_the_same_way(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        job = await make_job(db_session, user)
        output = await make_output(db_session, user, job, expires_at=past(hours=3))
        hold_id = await holds.hold(
            output_ref(output), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
        )

        await holds.release([hold_id])

        db_now = (await db_session.execute(select(func.now()))).scalar_one()
        assert await _output_expiry(db_session, output) >= db_now + timedelta(days=RETENTION_DAYS)

    async def test_an_asset_that_expires_later_than_the_new_horizon_keeps_its_date(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        # greatest(): releasing a hold never makes an asset expire *sooner*.
        image = await make_upload(db_session, user, expires_at=future(days=60))
        before = await _upload_expiry(db_session, image)
        hold_id = await holds.hold(
            upload_ref(image), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
        )

        await holds.release([hold_id])

        assert await _upload_expiry(db_session, image) == before

    async def test_releasing_several_holds_of_several_assets_rearms_each_unheld_asset(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        a = await make_upload(db_session, user, expires_at=past())
        b = await make_upload(db_session, user, expires_at=past())
        kept = await make_upload(db_session, user, expires_at=past())
        hold_a = await holds.hold(
            upload_ref(a), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
        )
        hold_b = await holds.hold(
            upload_ref(b), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
        )
        # ``b`` has a second holder that stays.
        await holds.hold(
            upload_ref(b),
            holder=holder("other.kind"),
            user_id=user.id,
            product_id="vex",
            accept=IMAGES,
        )
        await holds.hold(
            upload_ref(kept), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
        )

        await holds.release([hold_a, hold_b])

        db_now = (await db_session.execute(select(func.now()))).scalar_one()
        assert await _upload_expiry(db_session, a) >= db_now + timedelta(days=RETENTION_DAYS)
        assert await _upload_expiry(db_session, b) < db_now  # still held by the other holder
        assert await _upload_expiry(db_session, kept) < db_now  # never released

    async def test_release_ignores_unknown_ids_and_an_empty_list(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        image = await make_upload(db_session, user, expires_at=past())
        before = await _upload_expiry(db_session, image)

        await holds.release([])
        await holds.release([uuid4(), uuid4()])

        assert await _upload_expiry(db_session, image) == before  # nothing was re-armed
        # Releasing the same hold twice is idempotent.
        hold_id = await holds.hold(
            upload_ref(image), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
        )
        await holds.release([hold_id])
        await holds.release([hold_id])
        assert await holds.holder_counts(upload_ref(image)) == ()


class TestEligibility:
    async def test_an_upload_thumbnail_is_a_derivative(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        parent = await make_upload(db_session, user)
        thumb = await make_upload(db_session, user, is_thumbnail=True, parent=parent)

        with pytest.raises(AssetNotEligibleError) as raised:
            await holds.hold(
                upload_ref(thumb), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
            )

        assert raised.value.reason is AssetIneligibleReason.DERIVATIVE

    async def test_an_output_thumbnail_is_a_derivative(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        job = await make_job(db_session, user)
        parent = await make_output(db_session, user, job)
        thumb = await make_output(db_session, user, job, is_thumbnail=True, parent=parent)

        with pytest.raises(AssetNotEligibleError) as raised:
            await holds.hold(
                output_ref(thumb), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
            )

        assert raised.value.reason is AssetIneligibleReason.DERIVATIVE

    async def test_a_child_row_that_is_not_flagged_a_thumbnail_is_still_a_derivative(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        parent = await make_upload(db_session, user)
        child = await make_upload(db_session, user, parent=parent)  # parent set, flag not

        with pytest.raises(AssetNotEligibleError) as raised:
            await holds.hold(
                upload_ref(child), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
            )

        assert raised.value.reason is AssetIneligibleReason.DERIVATIVE

    async def test_a_video_upload_is_not_accepted_by_an_image_holder(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        video = await make_upload(db_session, user, content_type="video/mp4")

        with pytest.raises(AssetNotEligibleError) as raised:
            await holds.hold(
                upload_ref(video), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
            )

        assert raised.value.reason is AssetIneligibleReason.MEDIA_KIND_NOT_ACCEPTED

    async def test_a_video_output_is_not_accepted_by_an_image_holder(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        job = await make_job(db_session, user)
        video = await make_output(db_session, user, job, content_type="video/mp4")

        with pytest.raises(AssetNotEligibleError) as raised:
            await holds.hold(
                output_ref(video), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
            )

        assert raised.value.reason is AssetIneligibleReason.MEDIA_KIND_NOT_ACCEPTED

    @pytest.mark.parametrize(
        "status", [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FAILED]
    )
    async def test_an_output_of_an_unfinished_job_is_not_available(
        self, status: JobStatus, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        job = await make_job(db_session, user, status=status)
        output = await make_output(db_session, user, job)

        with pytest.raises(AssetNotEligibleError) as raised:
            await holds.hold(
                output_ref(output),
                holder=holder(),
                user_id=user.id,
                product_id="vex",
                accept=IMAGES,
            )

        assert raised.value.reason is AssetIneligibleReason.JOB_NOT_AVAILABLE

    async def test_an_output_of_a_soft_deleted_job_is_not_available(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        job = await make_job(db_session, user, is_deleted=True)
        output = await make_output(db_session, user, job)

        with pytest.raises(AssetNotEligibleError) as raised:
            await holds.hold(
                output_ref(output),
                holder=holder(),
                user_id=user.id,
                product_id="vex",
                accept=IMAGES,
            )

        assert raised.value.reason is AssetIneligibleReason.JOB_NOT_AVAILABLE

    async def test_an_ineligible_asset_gets_no_hold(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        video = await make_upload(db_session, user, content_type="video/mp4")
        with pytest.raises(AssetNotEligibleError):
            await holds.hold(
                upload_ref(video), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
            )

        assert (
            await db_session.execute(select(func.count()).select_from(LibraryAssetHold))
        ).scalar_one() == 0


class TestAcceptedKinds:
    """Eligibility is per holder (S2a): the holder declares the media kinds it pins."""

    async def test_accept_is_required_there_is_no_default(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        image = await make_upload(db_session, user)

        with pytest.raises(TypeError, match="accept"):
            await holds.hold(  # type: ignore[call-arg]
                upload_ref(image), holder=holder(), user_id=user.id, product_id="vex"
            )

    @pytest.mark.parametrize("output", [False, True], ids=["upload", "output"])
    async def test_an_image_is_refused_by_an_audio_holder(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User, output: bool
    ) -> None:
        if output:
            asset = await make_output(db_session, user, await make_job(db_session, user))
            ref = output_ref(asset)
        else:
            ref = upload_ref(await make_upload(db_session, user))

        with pytest.raises(AssetNotEligibleError) as raised:
            await holds.hold(ref, holder=holder(), user_id=user.id, product_id="vex", accept=AUDIO)

        assert raised.value.reason is AssetIneligibleReason.MEDIA_KIND_NOT_ACCEPTED
        assert raised.value.reason.value == "media_kind_not_accepted"
        assert await holds.holder_counts(ref) == ()

    @pytest.mark.parametrize("output", [False, True], ids=["upload", "output"])
    async def test_audio_is_refused_by_an_image_holder(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User, output: bool
    ) -> None:
        if output:
            asset = await make_output(
                db_session, user, await make_job(db_session, user), content_type="audio/mpeg"
            )
            ref = output_ref(asset)
        else:
            ref = upload_ref(await make_audio_upload(db_session, user))

        with pytest.raises(AssetNotEligibleError) as raised:
            await holds.hold(ref, holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES)

        assert raised.value.reason is AssetIneligibleReason.MEDIA_KIND_NOT_ACCEPTED

    @pytest.mark.parametrize("output", [False, True], ids=["upload", "output"])
    async def test_audio_is_held_by_an_audio_holder(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User, output: bool
    ) -> None:
        if output:
            asset = await make_output(
                db_session, user, await make_job(db_session, user), content_type="audio/flac"
            )
            ref = output_ref(asset)
        else:
            ref = upload_ref(await make_audio_upload(db_session, user))

        hold_id = await holds.hold(
            ref,
            holder=holder("studio.track_audio"),
            user_id=user.id,
            product_id="vex",
            accept=AUDIO,
        )

        assert await holds.hold_refs([hold_id]) == {hold_id: ref}

    async def test_a_holder_may_accept_several_kinds(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        both = frozenset({MediaKind.IMAGE, MediaKind.AUDIO})
        image = upload_ref(await make_upload(db_session, user))
        audio = upload_ref(await make_audio_upload(db_session, user))
        video = upload_ref(await make_upload(db_session, user, content_type="video/mp4"))

        for ref in (image, audio):
            await holds.hold(ref, holder=holder(), user_id=user.id, product_id="vex", accept=both)
        with pytest.raises(AssetNotEligibleError):
            await holds.hold(video, holder=holder(), user_id=user.id, product_id="vex", accept=both)

    async def test_an_empty_accept_set_holds_nothing(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        image = upload_ref(await make_upload(db_session, user))

        with pytest.raises(AssetNotEligibleError) as raised:
            await holds.hold(
                image, holder=holder(), user_id=user.id, product_id="vex", accept=frozenset()
            )

        assert raised.value.reason is AssetIneligibleReason.MEDIA_KIND_NOT_ACCEPTED

    async def test_a_content_type_of_no_library_kind_is_never_accepted(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        odd = upload_ref(await make_upload(db_session, user, content_type="application/pdf"))

        with pytest.raises(AssetNotEligibleError) as raised:
            await holds.hold(
                odd,
                holder=holder(),
                user_id=user.id,
                product_id="vex",
                accept=frozenset(MediaKind),
            )

        assert raised.value.reason is AssetIneligibleReason.MEDIA_KIND_NOT_ACCEPTED

    async def test_the_derivative_check_still_comes_first(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        parent = await make_audio_upload(db_session, user)
        thumb = await make_upload(
            db_session, user, is_thumbnail=True, parent=parent, content_type="audio/flac"
        )

        with pytest.raises(AssetNotEligibleError) as raised:
            await holds.hold(
                upload_ref(thumb), holder=holder(), user_id=user.id, product_id="vex", accept=AUDIO
            )

        assert raised.value.reason is AssetIneligibleReason.DERIVATIVE


class TestOwnership:
    async def test_another_users_asset_is_not_found(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User, make_user
    ) -> None:  # type: ignore[no-untyped-def]
        other = await make_user(email=f"other-{uuid4().hex}@example.com")
        image = await make_upload(db_session, other)
        job = await make_job(db_session, other)
        output = await make_output(db_session, other, job)

        for ref in (upload_ref(image), output_ref(output)):
            with pytest.raises(AssetNotFoundError):
                await holds.hold(
                    ref, holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
                )

    async def test_an_asset_of_another_product_is_not_found(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        image = await make_upload(db_session, user)
        job = await make_job(db_session, user)
        output = await make_output(db_session, user, job)

        for ref in (upload_ref(image), output_ref(output)):
            with pytest.raises(AssetNotFoundError):
                await holds.hold(
                    ref, holder=holder(), user_id=user.id, product_id="synthara", accept=IMAGES
                )

    async def test_a_missing_asset_is_the_same_error(
        self, holds: AssetHoldService, user: User
    ) -> None:
        for source in LibraryAssetSource:
            with pytest.raises(AssetNotFoundError):
                await holds.hold(
                    AssetRef(source=source, asset_id=uuid4()),
                    holder=holder(),
                    user_id=user.id,
                    product_id="vex",
                    accept=IMAGES,
                )

    async def test_held_refs_only_answers_for_the_owner(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User, make_user
    ) -> None:  # type: ignore[no-untyped-def]
        other = await make_user(email=f"other-{uuid4().hex}@example.com")
        image = await make_upload(db_session, user)
        await holds.hold(
            upload_ref(image), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
        )

        assert (
            await holds.held_refs([upload_ref(image)], user_id=other.id, product_id="vex")
            == frozenset()
        )
        assert (
            await holds.held_refs([upload_ref(image)], user_id=user.id, product_id="synthara")
            == frozenset()
        )


class TestInsertErrors:
    async def test_a_foreign_key_violation_on_insert_is_asset_not_found(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        # The asset is locked in real life, so the FK cannot fail; this proves that if it ever
        # does (the asset vanished), the answer is the typed 404 and the session stays usable.
        holds._repo.lock_upload = AsyncMock(  # type: ignore[method-assign]
            return_value=LockedAsset(is_derivative=False, content_type="image/png")
        )
        with pytest.raises(AssetNotFoundError):
            await holds.hold(
                AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=uuid4()),
                holder=holder(),
                user_id=user.id,
                product_id="vex",
                accept=IMAGES,
            )

        image = await make_upload(db_session, user)
        holds._repo.lock_output = AsyncMock(  # type: ignore[method-assign]
            return_value=LockedAsset(
                is_derivative=False,
                content_type="image/png",
                job_id=(await make_job(db_session, user)).id,
            )
        )
        with pytest.raises(AssetNotFoundError):
            await holds.hold(
                AssetRef(source=LibraryAssetSource.OUTPUT, asset_id=uuid4()),
                holder=holder(),
                user_id=user.id,
                product_id="vex",
                accept=IMAGES,
            )
        # still usable: the savepoint rolled back, not the whole transaction
        await holds.hold(
            upload_ref(image), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
        )


class TestDatabaseGuarantees:
    async def test_a_held_asset_cannot_be_deleted(
        self, holds: AssetHoldService, db_session: AsyncSession, user: User
    ) -> None:
        image = await make_upload(db_session, user)
        job = await make_job(db_session, user)
        output = await make_output(db_session, user, job)
        await holds.hold(
            upload_ref(image), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
        )
        await holds.hold(
            output_ref(output), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
        )

        for statement, constraint in (
            (delete(UserImage).where(UserImage.id == image.id), HOLD_UPLOAD_FK),
            (delete(GenerationOutput).where(GenerationOutput.id == output.id), HOLD_OUTPUT_FK),
        ):
            with pytest.raises(IntegrityError) as raised:
                async with db_session.begin_nested():
                    await db_session.execute(statement)
            assert violated_constraint(raised.value, frozenset({constraint})) == constraint

    async def test_a_hold_names_exactly_one_asset(
        self, db_session: AsyncSession, user: User
    ) -> None:
        image = await make_upload(db_session, user)
        job = await make_job(db_session, user)
        output = await make_output(db_session, user, job)
        for upload_id, output_id in ((None, None), (image.id, output.id)):
            with pytest.raises(IntegrityError) as raised:
                async with db_session.begin_nested():
                    db_session.add(
                        LibraryAssetHold(
                            user_id=user.id,
                            product_id="vex",
                            upload_id=upload_id,
                            output_id=output_id,
                            holder_kind=KIND,
                            holder_id=uuid4(),
                        )
                    )
                    await db_session.flush()
            assert "ck_library_asset_holds_one_asset" in str(raised.value)

    async def test_the_by_asset_lookups_are_served_by_the_partial_unique_indexes(
        self, db_session: AsyncSession
    ) -> None:
        # EXPLAIN with sequential scans disabled: the planner must have an index for the lookup
        # (the partial uniques double as the EXISTS index, so no extra index exists).
        from sqlalchemy import text

        await db_session.execute(text("SET LOCAL enable_seqscan = off"))
        for column, index in (
            ("upload_id", "uq_library_asset_holds_upload_holder"),
            ("output_id", "uq_library_asset_holds_output_holder"),
        ):
            plan = "\n".join(
                row[0]
                for row in await db_session.execute(
                    text(
                        f"EXPLAIN SELECT 1 FROM library_asset_holds WHERE {column} = :id"  # noqa: S608
                    ),
                    {"id": uuid4()},
                )
            )
            assert index in plan, plan


class TestUserHardDelete:
    async def test_deleting_a_user_removes_holds_and_assets_in_one_statement(
        self, hold_world: HoldWorld
    ) -> None:
        # ``NO ACTION`` (not ``RESTRICT``) is what lets this cascade succeed.
        user = await hold_world.user()
        image = await hold_world.upload(user)
        job = await hold_world.job(user)
        output = await hold_world.output(user, job)
        async with hold_world.factory() as session:
            service = AssetHoldService(session, retention_days=RETENTION_DAYS)
            await service.hold(
                upload_ref(image), holder=holder(), user_id=user.id, product_id="vex", accept=IMAGES
            )
            await service.hold(
                output_ref(output),
                holder=holder(),
                user_id=user.id,
                product_id="vex",
                accept=IMAGES,
            )
            await session.commit()
        assert await hold_world.hold_count() == 2

        await hold_world.cleanup()

        assert not await hold_world.upload_exists(image)
        assert not await hold_world.output_exists(output)
        async with hold_world.factory() as session:
            remaining = (
                await session.execute(
                    select(func.count())
                    .select_from(LibraryAssetHold)
                    .where(LibraryAssetHold.user_id == user.id)
                )
            ).scalar_one()
        assert remaining == 0
