"""The studio cast service against PostgreSQL (S1-P3, W6 and the invariants of P3-10).

One rolled-back transaction per test (``db_session``). Every test ends with the **reconciliation
check** (invariant 3): every reference has exactly one hold, every cast hold has its reference,
positions are dense. The two-session races and the HTTP surface live in their own modules.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import msgspec
import pytest
import pytest_asyncio
from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError

from src.api.schemas.studio import (
    ArtistCreate,
    ArtistPatch,
    CastLookIn,
    CastLookPatch,
    CastMemberCreate,
    CastMemberPatch,
    CastReferenceIn,
    CastReferenceOrder,
    CastReferencePatch,
)
from src.api.services.library_holds import AssetHolder, AssetHoldService, HolderCount
from src.core.library_ref import AssetRef, LibraryAssetSource, parse_asset_ref
from src.db.integrity import violated_constraint
from src.db.models.storage import GenerationOutput, UserImage
from src.db.models.studio_cast import (
    CAST_REFERENCE_HOLDER_KIND,
    LOOK_LINK_LOOK_FK,
    LOOK_LINK_REFERENCE_FK,
    StudioCastLookReference,
    StudioCastReference,
)
from src.studio.constraints import (
    MAX_CAST_LOOKS,
    MAX_CAST_MEMBERS_PER_OWNER,
    MAX_CAST_REFERENCES,
)
from src.studio.errors import (
    ArtistNotFound,
    AssetNotEligible,
    AssetNotFound,
    CastLabelTaken,
    CastLimitReached,
    CastLookNameTaken,
    CastLookNotFound,
    CastMemberInUse,
    CastMemberNotFound,
    CastNameTaken,
    CastReferenceNotFound,
    InvalidCursor,
    InvalidReferenceOrder,
    StudioDocumentInvalid,
)
from src.studio.services.artist import ArtistService
from tests.cast_support import (
    assert_reconciled,
    cast_service,
    count,
    create_body,
    ref_in,
    wire,
)
from tests.library_holds_support import (
    RETENTION_DAYS,
    future,
    make_job,
    make_output,
    make_upload,
    past,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.user import User
    from src.studio.services.cast import CastService

PRODUCT = "vex"
EXPANDING = "क़"  # NFC expands it to two code points (U+0915 U+093C)


@pytest_asyncio.fixture(autouse=True)
async def reconciled(db_session: AsyncSession) -> AsyncGenerator[None]:
    """Invariant 3, after **every** test of this module."""
    yield
    await assert_reconciled(db_session)


@pytest.fixture
async def user(make_user) -> User:  # type: ignore[no-untyped-def]
    return await make_user(email=f"cast-{uuid4().hex}@example.com")


@pytest.fixture
def cast(db_session: AsyncSession) -> CastService:
    return cast_service(db_session)


async def images(session: AsyncSession, user: User, n: int) -> list[UserImage]:
    return [await make_upload(session, user, expires_at=past()) for _ in range(n)]


async def member_with(
    cast: CastService, session: AsyncSession, user: User, n: int = 3, *, name: str = "Mira"
):
    """A member with ``n`` references labelled ``ref-0`` … over fresh (expired) uploads."""
    assets = await images(session, user, n)
    member = await cast.create(
        user.id,
        PRODUCT,
        create_body(name, references=[ref_in(f"ref-{i}", asset) for i, asset in enumerate(assets)]),
    )
    return member, assets


async def expiry(session: AsyncSession, image: UserImage):
    return (
        await session.execute(
            select(UserImage.expires_at)
            .where(UserImage.id == image.id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def holders(session: AsyncSession, asset: UserImage) -> tuple[HolderCount, ...]:
    return await AssetHoldService(session, retention_days=RETENTION_DAYS).holder_counts(
        AssetRef(source=LibraryAssetSource.UPLOAD, asset_id=asset.id)
    )


class TestMembers:
    async def test_create_returns_the_member_with_its_ordered_references_and_media(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        a, b = await images(db_session, user, 2)

        out = await cast.create(
            user.id,
            PRODUCT,
            create_body(
                "Mira",
                description="Lead singer",
                appearance="short silver hair",
                references=[ref_in("front", a), ref_in("side", b)],
            ),
        )

        assert (out.name, out.description, out.appearance) == (
            "Mira",
            "Lead singer",
            "short silver hair",
        )
        assert [(r.label, r.position, r.asset) for r in out.references] == [
            ("front", 0, wire(a)),
            ("side", 1, wire(b)),
        ]
        assert out.references[0].media.asset_ref == wire(a)
        assert out.references[0].media.original.url == f"/v1/content/uploads/{a.id}"
        assert out.looks == []
        assert await holders(db_session, a) == (HolderCount(CAST_REFERENCE_HOLDER_KIND, 1),)

    async def test_create_pins_both_uploads_and_outputs(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        upload = await make_upload(db_session, user)
        job = await make_job(db_session, user)
        output = await make_output(db_session, user, job)

        out = await cast.create(
            user.id,
            PRODUCT,
            create_body("Mixed", references=[ref_in("up", upload), ref_in("gen", output)]),
        )

        assert [r.asset for r in out.references] == [wire(upload), wire(output)]
        assert out.references[1].media.original.url == f"/v1/content/outputs/{output.id}"

    async def test_get_returns_the_same_member(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        out, _ = await member_with(cast, db_session, user, 2)

        assert msgspec.to_builtins(await cast.get(out.id, user.id, PRODUCT)) == (
            msgspec.to_builtins(out)
        )

    async def test_names_are_normalized(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        asset = await make_upload(db_session, user)
        out = await cast.create(
            user.id,
            PRODUCT,
            create_body(
                "  Mira \t  Vale  ",
                description="  notes  ",
                references=[ref_in("  front   view ", asset)],
            ),
        )

        assert out.name == "Mira Vale"
        assert out.description == "notes"
        assert out.references[0].label == "front view"

    async def test_blank_names_and_labels_are_issues_not_empty_strings(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        asset = await make_upload(db_session, user)

        with pytest.raises(StudioDocumentInvalid) as raised:
            await cast.create(
                user.id, PRODUCT, create_body("   ", references=[ref_in("\t", asset)])
            )

        assert {(i.path, i.code) for i in raised.value.issues} == {
            ("name", "empty_after_normalization"),
            ("references[0].label", "empty_after_normalization"),
        }

    async def test_a_malformed_asset_ref_is_an_issue_at_its_path(
        self, cast: CastService, user: User
    ) -> None:
        with pytest.raises(StudioDocumentInvalid) as raised:
            await cast.create(
                user.id,
                PRODUCT,
                create_body("Mira", references=[CastReferenceIn(label="x", asset="not-a-ref")]),
            )

        assert [(i.path, i.code) for i in raised.value.issues] == [
            ("references[0].asset", "invalid_asset_ref")
        ]

    @pytest.mark.parametrize(
        ("field", "count_on_wire", "path"),
        [
            ("name", 60, "name"),  # limit 100: 60 code points on the wire, 120 after NFC
            ("appearance", 161, "appearance"),  # limit 320
            ("description", 1001, "description"),  # limit 2000
            ("label", 21, "references[0].label"),  # limit 40
        ],
    )
    async def test_nfc_expansion_past_a_limit_is_reported_after_normalization(
        self,
        field: str,
        count_on_wire: int,
        path: str,
        cast: CastService,
        db_session: AsyncSession,
        user: User,
    ) -> None:
        asset = await make_upload(db_session, user)
        text = EXPANDING * count_on_wire
        references = [ref_in("ok", asset)]
        if field == "label":
            references = [CastReferenceIn(label=text, asset=wire(asset))]
        built = create_body(
            "Mira",
            references=references,
            description=text if field == "description" else None,
            appearance=text if field == "appearance" else None,
        )
        if field == "name":
            built = create_body("x", references=references)
            built = msgspec.structs.replace(built, name=text)

        # The wire accepts it: it is within its limit as sent ...
        body = msgspec.json.decode(msgspec.json.encode(built), type=CastMemberCreate)
        # ... and the service refuses it once it is normalized.
        with pytest.raises(StudioDocumentInvalid) as raised:
            await cast.create(user.id, PRODUCT, body)

        assert [(i.path, i.code) for i in raised.value.issues] == [
            (path, "too_long_after_normalization")
        ]

    async def test_patch_is_tri_state(self, cast: CastService, user: User) -> None:
        created = await cast.create(
            user.id, PRODUCT, create_body("Mira", description="d", appearance="a")
        )

        untouched = await cast.patch(created.id, user.id, PRODUCT, CastMemberPatch())
        renamed = await cast.patch(created.id, user.id, PRODUCT, CastMemberPatch(name="  Nova "))
        cleared = await cast.patch(
            created.id, user.id, PRODUCT, CastMemberPatch(description=None, appearance=None)
        )

        assert (untouched.name, untouched.description, untouched.appearance) == ("Mira", "d", "a")
        assert renamed.name == "Nova"
        assert (cleared.description, cleared.appearance) == (None, None)
        assert cleared.name == "Nova"

    async def test_patch_rechecks_lengths_after_normalization(
        self, cast: CastService, user: User
    ) -> None:
        created = await cast.create(user.id, PRODUCT, create_body("Mira"))

        with pytest.raises(StudioDocumentInvalid) as raised:
            await cast.patch(
                created.id,
                user.id,
                PRODUCT,
                CastMemberPatch(appearance=EXPANDING * 161),
            )

        assert [(i.path, i.code) for i in raised.value.issues] == [
            ("appearance", "too_long_after_normalization")
        ]

    async def test_names_are_unique_per_owner_case_insensitively(
        self, cast: CastService, user: User, make_user
    ) -> None:  # type: ignore[no-untyped-def]
        await cast.create(user.id, PRODUCT, create_body("Mira"))

        with pytest.raises(CastNameTaken):
            await cast.create(user.id, PRODUCT, create_body("  mIRA "))
        other = await cast.create(user.id, PRODUCT, create_body("Nova"))
        with pytest.raises(CastNameTaken):
            await cast.patch(other.id, user.id, PRODUCT, CastMemberPatch(name="MIRA"))
        # another owner, another product: the same name is free
        stranger = await make_user(email=f"stranger-{uuid4().hex}@example.com")
        await cast.create(stranger.id, PRODUCT, create_body("Mira"))
        await cast.create(user.id, "synthara", create_body("Mira"))
        # a rename to its own name in another case is the same row, not a conflict
        await cast.patch(other.id, user.id, PRODUCT, CastMemberPatch(name="nova"))

    async def test_a_failed_create_leaves_nothing_behind(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        good = await make_upload(db_session, user)
        video = await make_upload(db_session, user, content_type="video/mp4")

        with pytest.raises(AssetNotEligible) as raised:
            await cast.create(
                user.id,
                PRODUCT,
                create_body("Mira", references=[ref_in("ok", good), ref_in("bad", video)]),
            )

        assert raised.value.detail == {"reason": "not_an_image"}
        assert await count(db_session, "studio_cast_members") == 0
        assert await count(db_session, "studio_cast_references") == 0
        assert await count(db_session, "library_asset_holds") == 0  # the first hold went too
        # and the session is still healthy
        await cast.create(user.id, PRODUCT, create_body("Mira"))

    async def test_duplicate_labels_in_one_create_are_a_conflict_that_leaves_nothing(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        a, b = await images(db_session, user, 2)

        with pytest.raises(CastLabelTaken):
            await cast.create(
                user.id,
                PRODUCT,
                create_body("Mira", references=[ref_in("Front", a), ref_in("front", b)]),
            )

        assert await count(db_session, "studio_cast_members") == 0
        assert await count(db_session, "library_asset_holds") == 0

    async def test_the_member_cap_is_a_soft_cap(self, cast: CastService, user: User) -> None:
        for index in range(MAX_CAST_MEMBERS_PER_OWNER):
            await cast.create(user.id, PRODUCT, create_body(f"Member {index}"))

        with pytest.raises(CastLimitReached) as raised:
            await cast.create(user.id, PRODUCT, create_body("One too many"))

        assert raised.value.detail == {"kind": "members", "limit": MAX_CAST_MEMBERS_PER_OWNER}
        # the cap is per owner and product
        await cast.create(user.id, "synthara", create_body("Elsewhere"))

    async def test_delete_releases_every_hold_and_rearms_the_assets(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, assets = await member_with(cast, db_session, user, 3)
        for asset in assets:
            assert await holders(db_session, asset) != ()

        await cast.delete(member.id, user.id, PRODUCT)

        db_now = (await db_session.execute(select(func.now()))).scalar_one()
        for asset in assets:
            assert await holders(db_session, asset) == ()
            assert await expiry(db_session, asset) >= db_now + timedelta(days=RETENTION_DAYS)
        with pytest.raises(CastMemberNotFound):
            await cast.get(member.id, user.id, PRODUCT)
        assert await count(db_session, "studio_cast_references") == 0

    async def test_delete_of_a_performer_is_refused_with_the_blockers(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, assets = await member_with(cast, db_session, user, 1)
        artists = ArtistService(db_session)
        first = await artists.create(
            user.id, PRODUCT, ArtistCreate(name="A1", performer_cast_member_id=member.id)
        )
        await artists.create(
            user.id, PRODUCT, ArtistCreate(name="A2", performer_cast_member_id=member.id)
        )

        with pytest.raises(CastMemberInUse) as raised:
            await cast.delete(member.id, user.id, PRODUCT)

        assert raised.value.detail == {"blockers": [{"code": "artists", "count": 2}]}
        assert await holders(db_session, assets[0]) != ()  # nothing was released
        # unlink both, then it works
        await artists.patch(first.id, user.id, PRODUCT, ArtistPatch(performer_cast_member_id=None))
        with pytest.raises(CastMemberInUse):
            await cast.delete(member.id, user.id, PRODUCT)
        for artist in (await artists.list_page(user.id, PRODUCT, limit=10, cursor=None)).items:
            await artists.patch(
                artist.id, user.id, PRODUCT, ArtistPatch(performer_cast_member_id=None)
            )
        await cast.delete(member.id, user.id, PRODUCT)


class TestList:
    async def test_newest_first_with_a_keyset_cursor_and_summaries(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        first, _ = await member_with(cast, db_session, user, 2, name="First")
        await cast.create_look(
            first.id,
            user.id,
            PRODUCT,
            CastLookIn(name="Stage", reference_ids=[first.references[0].id]),
        )
        second = await cast.create(user.id, PRODUCT, create_body("Second", appearance="tall"))
        third = await cast.create(user.id, PRODUCT, create_body("Third"))

        page_one = await cast.list_page(user.id, PRODUCT, limit=2, cursor=None)
        assert [m.name for m in page_one.items] == ["Third", "Second"]
        assert page_one.has_more
        page_two = await cast.list_page(user.id, PRODUCT, limit=2, cursor=page_one.next_cursor)

        assert [m.name for m in page_two.items] == ["First"]
        assert not page_two.has_more
        summary = page_two.items[0]
        assert (summary.reference_count, summary.look_count) == (2, 1)
        assert summary.cover is not None
        assert summary.cover.asset_ref == first.references[0].asset
        bare = next(m for m in page_one.items if m.id == second.id)
        assert (bare.reference_count, bare.look_count, bare.cover, bare.appearance) == (
            0,
            0,
            None,
            "tall",
        )
        assert third.id == page_one.items[0].id

    async def test_the_cover_follows_the_reference_at_position_zero(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, 3)
        reordered = [member.references[2].id, member.references[0].id, member.references[1].id]

        await cast.reorder_references(
            member.id, user.id, PRODUCT, CastReferenceOrder(reference_ids=reordered)
        )

        page = await cast.list_page(user.id, PRODUCT, limit=10, cursor=None)
        assert page.items[0].cover is not None
        assert page.items[0].cover.asset_ref == member.references[2].asset

    async def test_a_bad_cursor_is_invalid_cursor(self, cast: CastService, user: User) -> None:
        with pytest.raises(InvalidCursor):
            await cast.list_page(user.id, PRODUCT, limit=10, cursor="!!not-a-cursor!!")

    async def test_lists_only_the_callers_members_in_the_callers_product(
        self, cast: CastService, user: User, make_user
    ) -> None:  # type: ignore[no-untyped-def]
        stranger = await make_user(email=f"stranger-{uuid4().hex}@example.com")
        await cast.create(user.id, PRODUCT, create_body("Mine"))
        await cast.create(stranger.id, PRODUCT, create_body("Theirs"))
        await cast.create(user.id, "synthara", create_body("Elsewhere"))

        page = await cast.list_page(user.id, PRODUCT, limit=10, cursor=None)

        assert [m.name for m in page.items] == ["Mine"]

    async def test_the_holding_filter_returns_exactly_the_members_referencing_the_asset(
        self, cast: CastService, db_session: AsyncSession, user: User, make_user
    ) -> None:  # type: ignore[no-untyped-def]
        x, y, z = await images(db_session, user, 3)
        job = await make_job(db_session, user)
        output = await make_output(db_session, user, job)
        a = await cast.create(user.id, PRODUCT, create_body("A", references=[ref_in("x", x)]))
        b = await cast.create(user.id, PRODUCT, create_body("B", references=[ref_in("y", y)]))
        c = await cast.create(
            user.id,
            PRODUCT,
            create_body("C", references=[ref_in("x", x), ref_in("o", output)]),
        )
        # a hold of ANOTHER context on z must not make anyone match
        await AssetHoldService(db_session, retention_days=RETENTION_DAYS).hold(
            parse_asset_ref(wire(z)),
            holder=AssetHolder(kind="other.context", id=uuid4()),
            user_id=user.id,
            product_id=PRODUCT,
        )

        async def names(asset: UserImage | GenerationOutput) -> set[str]:
            page = await cast.list_page(
                user.id, PRODUCT, limit=50, cursor=None, holding=parse_asset_ref(wire(asset))
            )
            return {m.name for m in page.items}

        assert await names(x) == {"A", "C"}
        assert await names(y) == {"B"}
        assert await names(output) == {"C"}
        assert await names(z) == set()
        assert {a.id, b.id, c.id}  # the three exist
        # another owner never sees these members, even by asset
        stranger = await make_user(email=f"stranger-{uuid4().hex}@example.com")
        page = await cast.list_page(
            stranger.id, PRODUCT, limit=50, cursor=None, holding=parse_asset_ref(wire(x))
        )
        assert page.items == []


class TestReferences:
    async def test_add_appends_at_the_end_and_pins_the_asset(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, 2)
        (extra,) = await images(db_session, user, 1)

        reference = await cast.add_reference(
            member.id, user.id, PRODUCT, ref_in("  Extra  ", extra)
        )

        assert (reference.label, reference.position, reference.asset) == ("Extra", 2, wire(extra))
        assert await holders(db_session, extra) == (HolderCount(CAST_REFERENCE_HOLDER_KIND, 1),)
        assert [r.position for r in (await cast.get(member.id, user.id, PRODUCT)).references] == [
            0,
            1,
            2,
        ]

    async def test_the_same_asset_can_back_two_references_each_with_its_own_hold(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        (asset,) = await images(db_session, user, 1)
        member = await cast.create(
            user.id,
            PRODUCT,
            create_body("Mira", references=[ref_in("one", asset), ref_in("two", asset)]),
        )

        assert await holders(db_session, asset) == (HolderCount(CAST_REFERENCE_HOLDER_KIND, 2),)
        await cast.delete_reference(member.id, member.references[0].id, user.id, PRODUCT)
        assert await holders(db_session, asset) == (HolderCount(CAST_REFERENCE_HOLDER_KIND, 1),)

    async def test_the_reference_cap_is_twelve(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, MAX_CAST_REFERENCES)
        (extra,) = await images(db_session, user, 1)

        with pytest.raises(CastLimitReached) as raised:
            await cast.add_reference(member.id, user.id, PRODUCT, ref_in("13th", extra))

        assert raised.value.detail == {"kind": "references", "limit": MAX_CAST_REFERENCES}
        assert await holders(db_session, extra) == ()  # nothing was pinned

    async def test_labels_are_unique_per_member_case_insensitively_and_free_across_members(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, assets = await member_with(cast, db_session, user, 2)
        other = await cast.create(user.id, PRODUCT, create_body("Nova"))
        (extra,) = await images(db_session, user, 1)

        with pytest.raises(CastLabelTaken):
            await cast.add_reference(member.id, user.id, PRODUCT, ref_in("REF-0", extra))
        assert await holders(db_session, extra) == ()  # the savepoint took the hold with it
        await cast.add_reference(
            other.id, user.id, PRODUCT, ref_in("ref-0", extra)
        )  # elsewhere: ok
        with pytest.raises(CastLabelTaken):
            await cast.change_reference(
                member.id,
                member.references[1].id,
                user.id,
                PRODUCT,
                CastReferencePatch(label="Ref-0"),
            )
        assert assets  # the two original references are untouched

    async def test_relabel_only_changes_the_label(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, assets = await member_with(cast, db_session, user, 2)
        target = member.references[1]
        before = await holders(db_session, assets[1])

        changed = await cast.change_reference(
            member.id, target.id, user.id, PRODUCT, CastReferencePatch(label="  Profile ")
        )

        assert (changed.label, changed.asset, changed.position) == ("Profile", target.asset, 1)
        assert await holders(db_session, assets[1]) == before

    async def test_swapping_the_asset_moves_the_hold_and_rearms_the_old_asset(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, assets = await member_with(cast, db_session, user, 1)
        (new_asset,) = await images(db_session, user, 1)
        reference = member.references[0]

        changed = await cast.change_reference(
            member.id, reference.id, user.id, PRODUCT, CastReferencePatch(asset=wire(new_asset))
        )

        assert changed.asset == wire(new_asset)
        assert changed.id == reference.id
        assert await holders(db_session, assets[0]) == ()
        assert await holders(db_session, new_asset) == (HolderCount(CAST_REFERENCE_HOLDER_KIND, 1),)
        db_now = (await db_session.execute(select(func.now()))).scalar_one()
        assert await expiry(db_session, assets[0]) >= db_now + timedelta(days=RETENTION_DAYS)

    async def test_pointing_a_reference_at_the_same_asset_changes_nothing(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, assets = await member_with(cast, db_session, user, 1)
        reference = member.references[0]
        hold_before = (await db_session.execute(select(StudioCastReference.hold_id))).scalar_one()

        same = await cast.change_reference(
            member.id, reference.id, user.id, PRODUCT, CastReferencePatch(asset=wire(assets[0]))
        )

        assert msgspec.to_builtins(same) == msgspec.to_builtins(reference)
        hold_after = (await db_session.execute(select(StudioCastReference.hold_id))).scalar_one()
        assert hold_after == hold_before
        assert await expiry(db_session, assets[0]) < future(days=1)  # still held: not re-armed

    async def test_a_failed_swap_keeps_the_old_hold(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, assets = await member_with(cast, db_session, user, 1)
        video = await make_upload(db_session, user, content_type="video/mp4")

        with pytest.raises(AssetNotEligible):
            await cast.change_reference(
                member.id,
                member.references[0].id,
                user.id,
                PRODUCT,
                CastReferencePatch(asset=wire(video)),
            )
        with pytest.raises(AssetNotFound):
            await cast.change_reference(
                member.id,
                member.references[0].id,
                user.id,
                PRODUCT,
                CastReferencePatch(asset=f"upload:{uuid4()}"),
            )

        assert (await cast.get(member.id, user.id, PRODUCT)).references[0].asset == wire(assets[0])
        assert await holders(db_session, assets[0]) != ()

    async def test_a_swap_that_hits_a_label_conflict_rolls_the_new_hold_back(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, assets = await member_with(cast, db_session, user, 2)
        (new_asset,) = await images(db_session, user, 1)

        with pytest.raises(CastLabelTaken):
            await cast.change_reference(
                member.id,
                member.references[1].id,
                user.id,
                PRODUCT,
                CastReferencePatch(label="ref-0", asset=wire(new_asset)),
            )

        assert await holders(db_session, new_asset) == ()
        assert await holders(db_session, assets[1]) != ()

    async def test_delete_recompacts_positions_releases_the_hold_and_leaves_the_looks(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, assets = await member_with(cast, db_session, user, 4)
        refs = [r.id for r in member.references]
        look = await cast.create_look(
            member.id, user.id, PRODUCT, CastLookIn(name="Stage", reference_ids=[refs[1], refs[2]])
        )

        await cast.delete_reference(member.id, refs[1], user.id, PRODUCT)

        after = await cast.get(member.id, user.id, PRODUCT)
        assert [(r.id, r.position) for r in after.references] == [
            (refs[0], 0),
            (refs[2], 1),
            (refs[3], 2),
        ]
        assert after.looks[0].id == look.id
        assert after.looks[0].reference_ids == [refs[2]]  # it left the look
        assert await holders(db_session, assets[1]) == ()
        db_now = (await db_session.execute(select(func.now()))).scalar_one()
        assert await expiry(db_session, assets[1]) >= db_now + timedelta(days=RETENTION_DAYS)

    async def test_unknown_and_foreign_references_are_not_found(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, 1)
        other = await cast.create(user.id, PRODUCT, create_body("Nova"))

        with pytest.raises(CastReferenceNotFound):
            await cast.delete_reference(member.id, uuid4(), user.id, PRODUCT)
        # a reference of ANOTHER member of the same owner is just as unknown here
        with pytest.raises(CastReferenceNotFound):
            await cast.delete_reference(other.id, member.references[0].id, user.id, PRODUCT)
        with pytest.raises(CastReferenceNotFound):
            await cast.change_reference(
                other.id, member.references[0].id, user.id, PRODUCT, CastReferencePatch(label="x")
            )

    async def test_reorder_accepts_exactly_a_permutation(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, 3)
        ids = [r.id for r in member.references]

        out = await cast.reorder_references(
            member.id,
            user.id,
            PRODUCT,
            CastReferenceOrder(reference_ids=[ids[2], ids[0], ids[1]]),
        )

        assert [(r.id, r.position) for r in out.references] == [
            (ids[2], 0),
            (ids[0], 1),
            (ids[1], 2),
        ]

    @pytest.mark.parametrize(
        "bad",
        ["missing", "extra", "duplicate", "foreign", "empty"],
    )
    async def test_reorder_rejects_anything_else(
        self, bad: str, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, 3)
        other, _ = await member_with(cast, db_session, user, 1, name="Nova")
        ids = [r.id for r in member.references]
        order = {
            "missing": ids[:2],
            "extra": [*ids, uuid4()],
            "duplicate": [ids[0], ids[0], ids[1]],
            "foreign": [ids[0], ids[1], other.references[0].id],
            "empty": [],
        }[bad]

        with pytest.raises(InvalidReferenceOrder):
            await cast.reorder_references(
                member.id, user.id, PRODUCT, CastReferenceOrder(reference_ids=order)
            )

        assert [r.id for r in (await cast.get(member.id, user.id, PRODUCT)).references] == ids


class TestLooks:
    async def test_create_rename_and_replace_a_look(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, 3)
        ids = [r.id for r in member.references]

        look = await cast.create_look(
            member.id,
            user.id,
            PRODUCT,
            CastLookIn(name="  Stage   look ", reference_ids=[ids[2], ids[0]]),
        )
        assert look.name == "Stage look"
        assert look.reference_ids == [
            ids[0],
            ids[2],
        ]  # in reference POSITION order, not input order

        renamed = await cast.change_look(
            member.id, look.id, user.id, PRODUCT, CastLookPatch(name="Street")
        )
        replaced = await cast.change_look(
            member.id, look.id, user.id, PRODUCT, CastLookPatch(reference_ids=[ids[1]])
        )

        assert (renamed.name, renamed.reference_ids) == ("Street", [ids[0], ids[2]])
        assert (replaced.name, replaced.reference_ids) == ("Street", [ids[1]])
        detail = await cast.get(member.id, user.id, PRODUCT)
        assert [(lk.name, lk.reference_ids) for lk in detail.looks] == [("Street", [ids[1]])]

    async def test_looks_come_back_ordered_by_name(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, 1)
        only = [member.references[0].id]
        for name in ("zebra", "Apple", "mango"):
            await cast.create_look(
                member.id, user.id, PRODUCT, CastLookIn(name=name, reference_ids=only)
            )

        detail = await cast.get(member.id, user.id, PRODUCT)

        assert [lk.name for lk in detail.looks] == ["Apple", "mango", "zebra"]

    async def test_a_look_needs_valid_references(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, 2)
        other, _ = await member_with(cast, db_session, user, 1, name="Nova")
        ids = [r.id for r in member.references]
        foreign = other.references[0].id

        async def issues(reference_ids: list[UUID]) -> set[tuple[str, str]]:
            with pytest.raises(StudioDocumentInvalid) as raised:
                await cast.create_look(
                    member.id,
                    user.id,
                    PRODUCT,
                    CastLookIn(name="Look", reference_ids=reference_ids),
                )
            return {(i.path, i.code) for i in raised.value.issues}

        assert await issues([]) == {("reference_ids", "empty")}
        assert await issues([ids[0], ids[0]]) == {("reference_ids[1]", "duplicate_reference")}
        assert await issues([ids[0], foreign, uuid4()]) == {
            ("reference_ids[1]", "reference_not_found"),
            ("reference_ids[2]", "reference_not_found"),
        }
        assert await count(db_session, "studio_cast_looks") == 0

    async def test_replacing_with_an_empty_set_is_refused_too(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, 1)
        look = await cast.create_look(
            member.id,
            user.id,
            PRODUCT,
            CastLookIn(name="Look", reference_ids=[member.references[0].id]),
        )

        with pytest.raises(StudioDocumentInvalid) as raised:
            await cast.change_look(
                member.id, look.id, user.id, PRODUCT, CastLookPatch(reference_ids=[])
            )

        assert [(i.path, i.code) for i in raised.value.issues] == [("reference_ids", "empty")]

    async def test_a_look_emptied_by_reference_deletions_stays_and_is_returned_empty(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, 2)
        look = await cast.create_look(
            member.id,
            user.id,
            PRODUCT,
            CastLookIn(name="Look", reference_ids=[member.references[0].id]),
        )

        await cast.delete_reference(member.id, member.references[0].id, user.id, PRODUCT)

        detail = await cast.get(member.id, user.id, PRODUCT)
        assert [(lk.id, lk.reference_ids) for lk in detail.looks] == [(look.id, [])]

    async def test_look_names_are_unique_per_member_case_insensitively(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, 1)
        other, _ = await member_with(cast, db_session, user, 1, name="Nova")
        only = [member.references[0].id]
        first = await cast.create_look(
            member.id, user.id, PRODUCT, CastLookIn(name="Stage", reference_ids=only)
        )
        second = await cast.create_look(
            member.id, user.id, PRODUCT, CastLookIn(name="Street", reference_ids=only)
        )

        with pytest.raises(CastLookNameTaken):
            await cast.create_look(
                member.id, user.id, PRODUCT, CastLookIn(name=" STAGE ", reference_ids=only)
            )
        with pytest.raises(CastLookNameTaken):
            await cast.change_look(
                member.id, second.id, user.id, PRODUCT, CastLookPatch(name="stage")
            )
        # a different member may reuse it, and a look keeps its own name in another case
        await cast.create_look(
            other.id,
            user.id,
            PRODUCT,
            CastLookIn(name="Stage", reference_ids=[other.references[0].id]),
        )
        await cast.change_look(member.id, first.id, user.id, PRODUCT, CastLookPatch(name="STAGE"))

    async def test_the_look_cap_is_twelve(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, 1)
        only = [member.references[0].id]
        for index in range(MAX_CAST_LOOKS):
            await cast.create_look(
                member.id, user.id, PRODUCT, CastLookIn(name=f"Look {index}", reference_ids=only)
            )

        with pytest.raises(CastLimitReached) as raised:
            await cast.create_look(
                member.id, user.id, PRODUCT, CastLookIn(name="13th", reference_ids=only)
            )

        assert raised.value.detail == {"kind": "looks", "limit": MAX_CAST_LOOKS}

    async def test_delete_look_keeps_the_references(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        member, _ = await member_with(cast, db_session, user, 2)
        look = await cast.create_look(
            member.id,
            user.id,
            PRODUCT,
            CastLookIn(name="Look", reference_ids=[r.id for r in member.references]),
        )

        await cast.delete_look(member.id, look.id, user.id, PRODUCT)

        detail = await cast.get(member.id, user.id, PRODUCT)
        assert detail.looks == []
        assert len(detail.references) == 2
        with pytest.raises(CastLookNotFound):
            await cast.delete_look(member.id, look.id, user.id, PRODUCT)
        with pytest.raises(CastLookNotFound):
            await cast.change_look(member.id, uuid4(), user.id, PRODUCT, CastLookPatch(name="x"))


class TestLookLinkIntegrity:
    """A look only links its own member's references: a database rule behind the service check.

    ``CastService`` reports a foreign reference as a ``422`` with paths (see
    ``test_a_look_needs_valid_references``); these tests go around it with a direct Core insert,
    because the composite foreign keys are what make the rule hold for every writer.
    """

    LINK_FKS = frozenset({LOOK_LINK_LOOK_FK, LOOK_LINK_REFERENCE_FK})

    async def _link(
        self, session: AsyncSession, look_id: UUID, reference_id: UUID, member_id: UUID
    ) -> None:
        async with session.begin_nested():
            await session.execute(
                insert(StudioCastLookReference).values(
                    look_id=look_id, reference_id=reference_id, cast_member_id=member_id
                )
            )

    async def test_a_link_across_members_is_refused_by_the_database(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        mira, _ = await member_with(cast, db_session, user, 2)
        nova, _ = await member_with(cast, db_session, user, 1, name="Nova")
        look = await cast.create_look(
            mira.id,
            user.id,
            PRODUCT,
            CastLookIn(name="Look", reference_ids=[mira.references[0].id]),
        )
        nova_reference = nova.references[0].id

        # stamped with the look's member: the reference is not that member's ...
        with pytest.raises(IntegrityError) as stamped_with_look_member:
            await self._link(db_session, look.id, nova_reference, mira.id)
        # ... stamped with the reference's member: the look is not that member's
        with pytest.raises(IntegrityError) as stamped_with_reference_member:
            await self._link(db_session, look.id, nova_reference, nova.id)

        assert violated_constraint(stamped_with_look_member.value, self.LINK_FKS) == (
            LOOK_LINK_REFERENCE_FK
        )
        assert violated_constraint(stamped_with_reference_member.value, self.LINK_FKS) == (
            LOOK_LINK_LOOK_FK
        )
        # nothing was written: the look still links exactly its own reference
        detail = await cast.get(mira.id, user.id, PRODUCT)
        assert [lk.reference_ids for lk in detail.looks] == [[mira.references[0].id]]

    async def test_a_link_within_one_member_is_accepted(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        mira, _ = await member_with(cast, db_session, user, 2)
        look = await cast.create_look(
            mira.id,
            user.id,
            PRODUCT,
            CastLookIn(name="Look", reference_ids=[mira.references[0].id]),
        )

        await self._link(db_session, look.id, mira.references[1].id, mira.id)

        detail = await cast.get(mira.id, user.id, PRODUCT)
        assert [lk.reference_ids for lk in detail.looks] == [[r.id for r in mira.references]]

    async def test_a_link_with_a_member_that_matches_neither_side_is_refused(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        mira, _ = await member_with(cast, db_session, user, 2)
        look = await cast.create_look(
            mira.id,
            user.id,
            PRODUCT,
            CastLookIn(name="Look", reference_ids=[mira.references[0].id]),
        )

        with pytest.raises(IntegrityError) as raised:
            await self._link(db_session, look.id, mira.references[1].id, uuid4())

        assert violated_constraint(raised.value, self.LINK_FKS) in self.LINK_FKS

    async def test_the_cascades_still_remove_the_links(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        mira, _ = await member_with(cast, db_session, user, 2)
        look = await cast.create_look(
            mira.id,
            user.id,
            PRODUCT,
            CastLookIn(name="Look", reference_ids=[r.id for r in mira.references]),
        )

        await cast.delete_reference(mira.id, mira.references[0].id, user.id, PRODUCT)
        assert await count(db_session, "studio_cast_look_references", look_id=look.id) == 1

        await cast.delete_look(mira.id, look.id, user.id, PRODUCT)
        assert await count(db_session, "studio_cast_look_references") == 0


class TestOwnership:
    """A foreign, cross-product or unknown id is the same 404 on every operation."""

    async def test_every_member_operation_is_not_found_for_someone_else(
        self, cast: CastService, db_session: AsyncSession, user: User, make_user
    ) -> None:  # type: ignore[no-untyped-def]
        member, assets = await member_with(cast, db_session, user, 1)
        look = await cast.create_look(
            member.id,
            user.id,
            PRODUCT,
            CastLookIn(name="Look", reference_ids=[member.references[0].id]),
        )
        stranger = await make_user(email=f"stranger-{uuid4().hex}@example.com")
        reference_id = member.references[0].id
        (asset,) = await images(db_session, stranger, 1)

        for who, product in ((stranger.id, PRODUCT), (user.id, "synthara"), (uuid4(), PRODUCT)):
            operations = [
                cast.get(member.id, who, product),
                cast.patch(member.id, who, product, CastMemberPatch(name="x")),
                cast.delete(member.id, who, product),
                cast.add_reference(member.id, who, product, ref_in("x", asset)),
                cast.change_reference(
                    member.id, reference_id, who, product, CastReferencePatch(label="x")
                ),
                cast.delete_reference(member.id, reference_id, who, product),
                cast.reorder_references(
                    member.id, who, product, CastReferenceOrder(reference_ids=[reference_id])
                ),
                cast.create_look(
                    member.id, who, product, CastLookIn(name="y", reference_ids=[reference_id])
                ),
                cast.change_look(member.id, look.id, who, product, CastLookPatch(name="y")),
                cast.delete_look(member.id, look.id, who, product),
            ]
            for operation in operations:
                with pytest.raises(CastMemberNotFound):
                    await operation

        # nothing changed
        detail = await cast.get(member.id, user.id, PRODUCT)
        assert detail.name == "Mira"
        assert [lk.name for lk in detail.looks] == ["Look"]
        assert await holders(db_session, assets[0]) != ()

    async def test_another_users_or_products_asset_is_asset_not_found(
        self, cast: CastService, db_session: AsyncSession, user: User, make_user
    ) -> None:  # type: ignore[no-untyped-def]
        stranger = await make_user(email=f"stranger-{uuid4().hex}@example.com")
        (theirs,) = await images(db_session, stranger, 1)
        (mine,) = await images(db_session, user, 1)
        member = await cast.create(user.id, PRODUCT, create_body("Mira"))

        with pytest.raises(AssetNotFound):
            await cast.add_reference(member.id, user.id, PRODUCT, ref_in("x", theirs))
        with pytest.raises(AssetNotFound):  # my asset, but filed under the other product
            await cast.create(
                user.id, "synthara", create_body("Elsewhere", references=[ref_in("x", mine)])
            )
        with pytest.raises(AssetNotFound):
            await cast.create(
                user.id,
                PRODUCT,
                create_body(
                    "Other", references=[CastReferenceIn(label="x", asset=f"output:{uuid4()}")]
                ),
            )

    async def test_derivatives_unfinished_jobs_and_videos_are_not_eligible(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        from src.core.enums import JobStatus

        parent = await make_upload(db_session, user)
        thumb = await make_upload(db_session, user, is_thumbnail=True, parent=parent)
        video = await make_upload(db_session, user, content_type="video/mp4")
        running = await make_job(db_session, user, status=JobStatus.RUNNING)
        unfinished = await make_output(db_session, user, running)
        deleted_job = await make_job(db_session, user, is_deleted=True)
        hidden = await make_output(db_session, user, deleted_job)
        member = await cast.create(user.id, PRODUCT, create_body("Mira"))

        reasons = {}
        for name, asset in (
            ("thumb", thumb),
            ("video", video),
            ("unfinished", unfinished),
            ("hidden", hidden),
        ):
            with pytest.raises(AssetNotEligible) as raised:
                await cast.add_reference(member.id, user.id, PRODUCT, ref_in(name, asset))
            assert raised.value.status_code == 422
            reasons[name] = raised.value.detail

        assert reasons == {
            "thumb": {"reason": "derivative"},
            "video": {"reason": "not_an_image"},
            "unfinished": {"reason": "job_not_available"},
            "hidden": {"reason": "job_not_available"},
        }


class TestArtistPerformerLink:
    async def test_an_artist_links_changes_and_clears_its_performer(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        artists = ArtistService(db_session)
        mira = await cast.create(user.id, PRODUCT, create_body("Mira"))
        nova = await cast.create(user.id, PRODUCT, create_body("Nova"))
        artist = await artists.create(user.id, PRODUCT, ArtistCreate(name="Band"))
        assert artist.performer_cast_member_id is None

        linked = await artists.patch(
            artist.id, user.id, PRODUCT, ArtistPatch(performer_cast_member_id=mira.id)
        )
        untouched = await artists.patch(artist.id, user.id, PRODUCT, ArtistPatch(persona="p"))
        moved = await artists.patch(
            artist.id, user.id, PRODUCT, ArtistPatch(performer_cast_member_id=nova.id)
        )
        cleared = await artists.patch(
            artist.id, user.id, PRODUCT, ArtistPatch(performer_cast_member_id=None)
        )

        assert linked.performer_cast_member_id == mira.id
        assert untouched.performer_cast_member_id == mira.id
        assert moved.performer_cast_member_id == nova.id
        assert cleared.performer_cast_member_id is None
        assert (await artists.get(artist.id, user.id, PRODUCT)).performer_cast_member_id is None

    async def test_an_unknown_or_foreign_member_is_cast_member_not_found(
        self, cast: CastService, db_session: AsyncSession, user: User, make_user
    ) -> None:  # type: ignore[no-untyped-def]
        artists = ArtistService(db_session)
        stranger = await make_user(email=f"stranger-{uuid4().hex}@example.com")
        theirs = await cast.create(stranger.id, PRODUCT, create_body("Theirs"))
        elsewhere = await cast.create(user.id, "synthara", create_body("Elsewhere"))
        artist = await artists.create(user.id, PRODUCT, ArtistCreate(name="Band"))

        for member_id in (uuid4(), theirs.id, elsewhere.id):
            with pytest.raises(CastMemberNotFound):
                await artists.patch(
                    artist.id, user.id, PRODUCT, ArtistPatch(performer_cast_member_id=member_id)
                )
            with pytest.raises(CastMemberNotFound):
                await artists.create(
                    user.id,
                    PRODUCT,
                    ArtistCreate(name="Other", performer_cast_member_id=member_id),
                )
        assert (await artists.get(artist.id, user.id, PRODUCT)).performer_cast_member_id is None

    async def test_a_member_deleted_after_the_existence_check_is_the_same_404(
        self, db_session: AsyncSession, user: User
    ) -> None:
        # The foreign key is the backstop: pretend the check passed for a member that is gone.
        artists = ArtistService(db_session)
        artist = await artists.create(user.id, PRODUCT, ArtistCreate(name="Band"))

        async def pass_the_check(*_: object) -> None:
            return None

        artists._require_cast_member = pass_the_check  # type: ignore[method-assign]
        with pytest.raises(CastMemberNotFound):
            await artists.patch(
                artist.id, user.id, PRODUCT, ArtistPatch(performer_cast_member_id=uuid4())
            )
        with pytest.raises(CastMemberNotFound):
            await artists.create(
                user.id, PRODUCT, ArtistCreate(name="Other", performer_cast_member_id=uuid4())
            )
        # an unrelated integrity failure is still a name conflict
        with pytest.raises(Exception, match="artist_name_conflict"):
            await artists.create(user.id, PRODUCT, ArtistCreate(name="band"))

    async def test_a_missing_artist_is_artist_not_found(
        self, db_session: AsyncSession, user: User
    ) -> None:
        with pytest.raises(ArtistNotFound):
            await ArtistService(db_session).patch(
                uuid4(), user.id, PRODUCT, ArtistPatch(performer_cast_member_id=None)
            )


class TestLogging:
    """Invariant 7: logs carry ids, counts and field names - never names, labels or text."""

    async def test_no_cast_log_line_contains_user_text(
        self, cast: CastService, db_session: AsyncSession, user: User
    ) -> None:
        from structlog.testing import capture_logs

        a, b, c = await images(db_session, user, 3)
        secrets = ("Zebulon-Secret", "label-secret", "look-secret", "desc-secret", "look-appear")

        with capture_logs() as logs:
            member = await cast.create(
                user.id,
                PRODUCT,
                create_body(
                    secrets[0],
                    description=secrets[3],
                    appearance=secrets[4],
                    references=[ref_in(secrets[1], a), ref_in("second-secret", b)],
                ),
            )
            await cast.patch(member.id, user.id, PRODUCT, CastMemberPatch(name="Renamed-Secret"))
            added = await cast.add_reference(member.id, user.id, PRODUCT, ref_in("third-secret", c))
            await cast.change_reference(
                member.id,
                added.id,
                user.id,
                PRODUCT,
                CastReferencePatch(label="relabel-secret"),
            )
            look = await cast.create_look(
                member.id,
                user.id,
                PRODUCT,
                CastLookIn(name=secrets[2], reference_ids=[member.references[0].id]),
            )
            await cast.change_look(
                member.id, look.id, user.id, PRODUCT, CastLookPatch(name="look-renamed-secret")
            )
            await cast.delete_look(member.id, look.id, user.id, PRODUCT)
            await cast.delete_reference(member.id, added.id, user.id, PRODUCT)
            await cast.delete(member.id, user.id, PRODUCT)

        events = [log["event"] for log in logs]
        assert events == [
            "studio.cast_member_created",
            "studio.cast_member_updated",
            "studio.cast_reference_added",
            "studio.cast_reference_changed",
            "studio.cast_look_saved",
            "studio.cast_look_saved",
            "studio.cast_look_deleted",
            "studio.cast_reference_removed",
            "studio.cast_member_deleted",
        ]
        rendered = " ".join(str(log) for log in logs).lower()
        for text in (*secrets, "renamed-secret", "second-secret", "third-secret", "relabel-secret"):
            assert text.lower() not in rendered, text
        for log in logs:
            for key, value in log.items():
                assert key in {
                    "event", "log_level", "cast_member_id", "reference_id", "look_id", "user_id",
                    "reference_count", "fields", "asset_changed",
                }, key  # fmt: skip
                if key == "fields":
                    assert set(value) <= {"name", "description", "appearance"}


class TestWireLimits:
    def test_the_constants_are_the_wire_limits_and_nothing_else_copies_them(self) -> None:
        from src.api.schemas.studio import (
            CastLookIn as LookIn,
        )
        from src.api.schemas.studio import (
            CastMemberCreate as Create,
        )
        from src.studio.constraints import (
            CAST_APPEARANCE_MAX,
            CAST_DESCRIPTION_MAX,
            CAST_LABEL_MAX,
            CAST_LOOK_NAME_MAX,
            CAST_NAME_MAX,
        )

        def limit(struct: type[msgspec.Struct], field: str) -> int | None:
            info = msgspec.inspect.type_info(struct)
            assert isinstance(info, msgspec.inspect.StructType)
            [spec] = [f for f in info.fields if f.name == field]
            type_ = spec.type
            if isinstance(type_, msgspec.inspect.UnionType):
                [type_] = [t for t in type_.types if isinstance(t, msgspec.inspect.StrType)]
            return getattr(type_, "max_length", None)

        assert limit(Create, "name") == CAST_NAME_MAX == 100
        assert limit(Create, "description") == CAST_DESCRIPTION_MAX == 2000
        assert limit(Create, "appearance") == CAST_APPEARANCE_MAX == 320
        assert limit(LookIn, "name") == CAST_LOOK_NAME_MAX == 60
        assert limit(CastReferenceIn, "label") == CAST_LABEL_MAX == 40
        assert (MAX_CAST_MEMBERS_PER_OWNER, MAX_CAST_REFERENCES, MAX_CAST_LOOKS) == (50, 12, 12)
