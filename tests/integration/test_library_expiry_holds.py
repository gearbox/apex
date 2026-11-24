"""The library's ``expiring`` filter and ``expiring_soon`` sort read the *effective* expiry (S1-P3 R1).

A held asset never expires while it is held: the retention sweep skips it, so its stored
``expires_at`` routinely lies in the past. The filter, the sort and that sort's cursor therefore
key off ``CASE WHEN EXISTS (hold) THEN PINNED_EXPIRY ELSE expires_at END`` (the row's
``expiry_key``) - never the raw column - in **both** branches of the ``UNION ALL``. The wire's
``expires_at`` keeps the stored value.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import select

from src.api.schemas.pagination import decode_library_cursor
from src.api.services.library import LibraryService
from src.api.services.library_holds import AssetHolder, AssetHoldService
from src.core.enums import LibrarySort
from src.core.library_limits import PINNED_EXPIRY
from src.core.library_ref import LibraryAssetSource, format_asset_ref, parse_asset_ref
from src.core.product_registry import VEX_CONFIG
from src.db.models.storage import GenerationOutput, UserImage
from tests.library_holds_support import RETENTION_DAYS, HoldWorld, future, past

if TYPE_CHECKING:
    from uuid import UUID

    from src.db.models.user import User

UPLOAD = LibraryAssetSource.UPLOAD
OUTPUT = LibraryAssetSource.OUTPUT
WINDOW = timedelta(days=7)  # the ``expiring`` window (P6)


class _Seed:
    """Committed assets of one user, addressed by their ``asset_ref`` string."""

    def __init__(self, world: HoldWorld, user: User) -> None:
        self.world = world
        self.user = user
        self._hold_ids: dict[str, list[UUID]] = {}

    async def asset(
        self,
        source: LibraryAssetSource,
        *,
        expires_at: datetime,
        held: bool = False,
    ) -> str:
        if source == UPLOAD:
            image = await self.world.upload(self.user, expires_at=expires_at)
            asset_ref = format_asset_ref(UPLOAD, image.id)
        else:
            job = await self.world.job(self.user)  # one job per output: no index to collide on
            output = await self.world.output(self.user, job, expires_at=expires_at)
            asset_ref = format_asset_ref(OUTPUT, output.id)
        if held:
            await self.hold(asset_ref)
        return asset_ref

    async def hold(self, asset_ref: str) -> None:
        """Add one more holder of ``asset_ref``."""
        async with self.world.factory() as session:
            hold_id = await AssetHoldService(session, retention_days=RETENTION_DAYS).hold(
                parse_asset_ref(asset_ref),
                holder=AssetHolder(kind="studio.cast_reference", id=uuid4()),
                user_id=self.user.id,
                product_id="vex",
            )
            await session.commit()
        self._hold_ids.setdefault(asset_ref, []).append(hold_id)

    async def release(self, asset_ref: str, *, retention_days: int) -> None:
        """Release the most recent holder of ``asset_ref`` (the last one re-arms the expiry)."""
        async with self.world.factory() as session:
            await AssetHoldService(session, retention_days=retention_days).release(
                [self._hold_ids[asset_ref].pop()]
            )
            await session.commit()

    async def stored_expiry(self, asset_ref: str) -> datetime:
        ref = parse_asset_ref(asset_ref)
        model = UserImage if ref.source == UPLOAD else GenerationOutput
        async with self.world.factory() as session:
            return (
                await session.execute(select(model.expires_at).where(model.id == ref.asset_id))
            ).scalar_one()

    async def listing(
        self,
        *,
        expiring: bool | None = None,
        sort: LibrarySort = LibrarySort.NEWEST,
        source: LibraryAssetSource | None = None,
    ) -> list[str]:
        async with self.world.factory() as session:
            page = await LibraryService(session=session).list_assets(
                self.user.id,
                "vex",
                VEX_CONFIG,
                session=session,
                limit=50,
                expiring=expiring,
                sort=sort,
                source=source,
            )
        return [item.asset_ref for item in page.items]


@pytest.fixture
async def seed(hold_world: HoldWorld) -> _Seed:
    return _Seed(hold_world, await hold_world.user())


class TestHeldAssetNeverExpires:
    """The reproduction: a pinned upload/output whose stored ``expires_at`` is 48 h in the past."""

    @pytest.fixture
    async def library(self, seed: _Seed) -> dict[str, str]:
        return {
            "soon_upload": await seed.asset(UPLOAD, expires_at=future(days=5)),
            "soon_output": await seed.asset(OUTPUT, expires_at=future(days=6)),
            "far_upload": await seed.asset(UPLOAD, expires_at=future(days=20)),
            "far_output": await seed.asset(OUTPUT, expires_at=future(days=30)),
            "pinned_upload": await seed.asset(UPLOAD, expires_at=past(hours=48), held=True),
            "pinned_output": await seed.asset(OUTPUT, expires_at=past(hours=49), held=True),
        }

    @pytest.mark.parametrize("source", [None, UPLOAD, OUTPUT], ids=["both", "uploads", "outputs"])
    async def test_expiring_true_excludes_the_held_asset_and_false_includes_it(
        self, seed: _Seed, library: dict[str, str], source: LibraryAssetSource | None
    ) -> None:
        def only(*names: str) -> set[str]:
            return {
                ref
                for name, ref in library.items()
                if name in names and (source is None or ref.startswith(f"{source.value}:"))
            }

        assert set(await seed.listing(expiring=True, source=source)) == only(
            "soon_upload", "soon_output"
        )
        assert set(await seed.listing(expiring=False, source=source)) == only(
            "far_upload", "far_output", "pinned_upload", "pinned_output"
        )
        assert set(await seed.listing(source=source)) == only(*library)

    @pytest.mark.parametrize("source", [None, UPLOAD, OUTPUT], ids=["both", "uploads", "outputs"])
    async def test_expiring_soon_puts_every_held_asset_after_every_unheld_one(
        self, seed: _Seed, library: dict[str, str], source: LibraryAssetSource | None
    ) -> None:
        expected = [
            library[name]
            for name in (
                "soon_upload",
                "soon_output",
                "far_upload",
                "far_output",
                "pinned_upload",  # held assets tie on PINNED_EXPIRY: source rank, then id
                "pinned_output",
            )
            if source is None or library[name].startswith(f"{source.value}:")
        ]

        assert await seed.listing(sort=LibrarySort.EXPIRING_SOON, source=source) == expected

    async def test_the_wire_keeps_the_stored_expires_at_and_held(
        self, seed: _Seed, library: dict[str, str]
    ) -> None:
        async with seed.world.factory() as session:
            page = await LibraryService(session=session).list_assets(
                seed.user.id, "vex", VEX_CONFIG, session=session, limit=50
            )

        by_ref = {item.asset_ref: item for item in page.items}
        pinned = by_ref[library["pinned_upload"]]
        assert pinned.held is True
        assert pinned.expires_at < datetime.now(UTC)  # the stored value, not PINNED_EXPIRY
        assert pinned.expires_at != PINNED_EXPIRY
        assert by_ref[library["soon_upload"]].held is False


class TestReleasedAssetIsOrdinaryAgain:
    """After the last hold goes the asset has its re-armed expiry and is judged by it."""

    @pytest.mark.parametrize("source", [UPLOAD, OUTPUT], ids=["upload", "output"])
    @pytest.mark.parametrize(
        ("retention_days", "inside_window"),
        [(3, True), (10, False)],
        ids=["retention-below-the-window", "retention-above-the-window"],
    )
    async def test_the_three_requests_use_the_rearmed_expires_at(
        self,
        seed: _Seed,
        source: LibraryAssetSource,
        retention_days: int,
        inside_window: bool,
    ) -> None:
        before = datetime.now(UTC)
        released = await seed.asset(source, expires_at=past(hours=48), held=True)
        near = await seed.asset(source, expires_at=future(days=4))
        far = await seed.asset(source, expires_at=future(days=20))

        await seed.release(released, retention_days=retention_days)

        # the retention clock restarted: expires_at >= now + RETENTION_DAYS
        rearmed = await seed.stored_expiry(released)
        assert rearmed >= before + timedelta(days=retention_days)
        assert (rearmed <= datetime.now(UTC) + WINDOW) is inside_window  # the case under test

        # the sort places it by that real value among the unheld assets ...
        by_expiry = sorted(
            [(await seed.stored_expiry(ref), ref) for ref in (released, near, far)],
        )
        assert await seed.listing(sort=LibrarySort.EXPIRING_SOON) == [ref for _, ref in by_expiry]
        # ... and the filter agrees with it, in both directions
        expiring = await seed.listing(expiring=True)
        not_expiring = await seed.listing(expiring=False)
        assert (released in expiring) is inside_window
        assert (released in not_expiring) is not inside_window
        assert near in expiring
        assert far in not_expiring

    async def test_a_second_holder_keeps_the_asset_pinned_until_the_last_goes(
        self, seed: _Seed
    ) -> None:
        pinned = await seed.asset(UPLOAD, expires_at=past(hours=48), held=True)
        ordinary = await seed.asset(UPLOAD, expires_at=future(days=6))
        await seed.hold(pinned)  # a second holder of the same asset

        await seed.release(pinned, retention_days=3)

        # one hold remains: the asset is still held, so it still never expires
        assert await seed.listing(expiring=True) == [ordinary]
        assert await seed.listing(sort=LibrarySort.EXPIRING_SOON) == [ordinary, pinned]

        await seed.release(pinned, retention_days=3)

        # the last hold went: re-armed to now + 3 days, an ordinary expiring asset again
        assert set(await seed.listing(expiring=True)) == {ordinary, pinned}
        assert await seed.listing(sort=LibrarySort.EXPIRING_SOON) == [pinned, ordinary]


class TestPagination:
    async def test_limit_one_walk_visits_every_asset_once_held_last_and_round_trips_pinned_expiry(
        self, seed: _Seed
    ) -> None:
        unheld_output = await seed.asset(OUTPUT, expires_at=future(days=3))
        unheld_upload = await seed.asset(UPLOAD, expires_at=future(days=5))
        held_upload = await seed.asset(UPLOAD, expires_at=past(hours=48), held=True)
        held_output = await seed.asset(OUTPUT, expires_at=past(hours=72), held=True)

        seen: list[str] = []
        cursors: list[str] = []
        cursor: str | None = None
        for _ in range(10):
            async with seed.world.factory() as session:
                page = await LibraryService(session=session).list_assets(
                    seed.user.id,
                    "vex",
                    VEX_CONFIG,
                    session=session,
                    limit=1,
                    cursor=cursor,
                    sort=LibrarySort.EXPIRING_SOON,
                )
            seen.extend(item.asset_ref for item in page.items)
            if not page.has_more:
                assert page.next_cursor is None
                break
            assert page.next_cursor is not None
            cursor = page.next_cursor
            cursors.append(cursor)
        else:
            pytest.fail("the walk did not end")

        # every asset once, unheld by effective expiry, then held (source rank, id)
        assert seen == [unheld_output, unheld_upload, held_upload, held_output]
        # the cursor issued after a held asset carries PINNED_EXPIRY (9999-12-31), not its
        # stored expires_at (which lies in the past), and decodes back to exactly that value
        after_held_upload = decode_library_cursor(cursors[2], expected_sort="expiring_soon")
        assert after_held_upload[0] == PINNED_EXPIRY
        assert after_held_upload[1:] == ("upload", parse_asset_ref(held_upload).asset_id)
        # the cursor after an unheld asset still carries its real expiry
        after_unheld = decode_library_cursor(cursors[0], expected_sort="expiring_soon")
        assert after_unheld[0] < datetime.now(UTC) + WINDOW
