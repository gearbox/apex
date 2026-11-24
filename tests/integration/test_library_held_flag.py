"""The library ``held`` flag (S1-P3, W5, P3-4): list, detail and group payloads, in one statement.

``held`` is an ``EXISTS`` subquery in **both** branches of the ``UNION ALL`` - not a second query
per page and not one per item. ``expires_at`` keeps its meaning (the UI hides the countdown while
``held`` is true).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import msgspec
from sqlalchemy import event

from src.api.schemas.library import LibraryOutputItem, LineageNode
from src.api.services.library import LibraryService
from src.api.services.library_holds import AssetHolder, AssetHoldService
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref
from src.core.product_registry import VEX_CONFIG
from src.db.repositories.library import LibraryRepository
from tests.library_holds_support import IMAGES, RETENTION_DAYS, HoldWorld, output_ref, upload_ref

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.db.models.user import User


class _Statements:
    """Counts the statements a block sends to the database."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine.sync_engine
        self.count = 0

    def _on_execute(self, *_: object) -> None:
        self.count += 1

    def __enter__(self) -> _Statements:
        event.listen(self._engine, "before_cursor_execute", self._on_execute)
        return self

    def __exit__(self, *_: object) -> None:
        event.remove(self._engine, "before_cursor_execute", self._on_execute)


def counting(engine: AsyncEngine) -> AbstractContextManager[_Statements]:
    return _Statements(engine)


async def _hold(world: HoldWorld, ref: AssetRef, user: User) -> UUID:
    async with world.factory() as session:
        hold_id = await AssetHoldService(session, retention_days=RETENTION_DAYS).hold(
            ref,
            holder=AssetHolder(kind="studio.cast_reference", id=uuid4()),
            user_id=user.id,
            product_id="vex",
            accept=IMAGES,
        )
        await session.commit()
        return hold_id


async def _library(user: User, session: AsyncSession) -> list[tuple[str, bool]]:
    page = await LibraryService(session=session).list_assets(
        user.id, "vex", VEX_CONFIG, session=session, limit=50
    )
    return [(item.asset_ref, item.held) for item in page.items]


class TestListFlag:
    async def test_held_is_true_exactly_for_held_assets_of_both_sources(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        held_upload = await hold_world.upload(user)
        free_upload = await hold_world.upload(user)
        job = await hold_world.job(user)
        held_output = await hold_world.output(user, job)
        free_output = await hold_world.output(user, job, output_index=1)
        await _hold(hold_world, upload_ref(held_upload), user)
        await _hold(hold_world, output_ref(held_output), user)
        # a second holder changes nothing (still one flag, no duplicate row)
        await _hold(hold_world, upload_ref(held_upload), user)

        async with hold_world.factory() as session:
            flags = dict(await _library(user, session))

        assert flags == {
            format_asset_ref(LibraryAssetSource.UPLOAD, held_upload.id): True,
            format_asset_ref(LibraryAssetSource.UPLOAD, free_upload.id): False,
            format_asset_ref(LibraryAssetSource.OUTPUT, held_output.id): True,
            format_asset_ref(LibraryAssetSource.OUTPUT, free_output.id): False,
        }

    async def test_a_released_asset_is_not_held_any_more(self, hold_world: HoldWorld) -> None:
        user = await hold_world.user()
        image = await hold_world.upload(user)
        hold_id = await _hold(hold_world, upload_ref(image), user)
        async with hold_world.factory() as session:
            assert dict(await _library(user, session)) == {f"upload:{image.id}": True}

        async with hold_world.factory() as session:
            await AssetHoldService(session, retention_days=RETENTION_DAYS).release([hold_id])
            await session.commit()

        async with hold_world.factory() as session:
            assert dict(await _library(user, session)) == {f"upload:{image.id}": False}

    async def test_another_users_hold_on_nothing_of_mine_does_not_show(
        self, hold_world: HoldWorld
    ) -> None:
        mine = await hold_world.user()
        theirs = await hold_world.user()
        my_image = await hold_world.upload(mine)
        their_image = await hold_world.upload(theirs)
        await _hold(hold_world, upload_ref(their_image), theirs)

        async with hold_world.factory() as session:
            assert dict(await _library(mine, session)) == {f"upload:{my_image.id}": False}

    async def test_the_flag_is_computed_in_the_page_query_one_statement_for_the_repository(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user)
        for index in range(4):
            image = await hold_world.upload(user)
            output = await hold_world.output(user, job, output_index=index)
            if index % 2 == 0:
                await _hold(hold_world, upload_ref(image), user)
                await _hold(hold_world, output_ref(output), user)

        async with hold_world.factory() as session:
            with counting(db_engine) as statements:
                rows = await LibraryRepository(session).list_assets(user.id, "vex", limit=50)

        assert len(rows) == 8
        assert sum(row.held for row in rows) == 4
        assert statements.count == 1  # not a second query, not one per item

    async def test_the_whole_page_costs_the_same_for_two_items_and_for_twelve(
        self, hold_world: HoldWorld, db_engine: AsyncEngine
    ) -> None:
        small = await hold_world.user()
        large = await hold_world.user()
        for _ in range(2):
            held = await hold_world.upload(small)
            await _hold(hold_world, upload_ref(held), small)
        for _ in range(12):
            held = await hold_world.upload(large)
            await _hold(hold_world, upload_ref(held), large)

        async def statements_for(user: User) -> tuple[int, int]:
            async with hold_world.factory() as session:
                with counting(db_engine) as statements:
                    flags = await _library(user, session)
            return statements.count, len(flags)

        small_cost, small_items = await statements_for(small)
        large_cost, large_items = await statements_for(large)

        assert (small_items, large_items) == (2, 12)
        assert small_cost == large_cost  # no N+1 for held


class TestDetailFlag:
    async def test_the_detail_of_an_upload_and_of_an_output_carries_the_flag(
        self, hold_world: HoldWorld
    ) -> None:
        user = await hold_world.user()
        held_upload = await hold_world.upload(user)
        free_upload = await hold_world.upload(user)
        job = await hold_world.job(user)
        held_output = await hold_world.output(user, job)
        free_output = await hold_world.output(user, job, output_index=1)
        await _hold(hold_world, upload_ref(held_upload), user)
        await _hold(hold_world, output_ref(held_output), user)

        async with hold_world.factory() as session:
            service = LibraryService(session=session)
            flags = {}
            for ref in (held_upload, free_upload, held_output, free_output):
                asset_ref = (
                    f"upload:{ref.id}" if ref in (held_upload, free_upload) else f"output:{ref.id}"
                )
                detail = await service.get_asset_detail(
                    asset_ref, user.id, "vex", VEX_CONFIG, session=session
                )
                assert detail is not None
                flags[asset_ref] = detail.held
                assert detail.expires_at is not None  # unchanged: expires_at keeps its meaning

        assert flags == {
            f"upload:{held_upload.id}": True,
            f"upload:{free_upload.id}": False,
            f"output:{held_output.id}": True,
            f"output:{free_output.id}": False,
        }


class TestGroupFlag:
    async def test_the_outputs_of_a_group_carry_the_flag(self, hold_world: HoldWorld) -> None:
        user = await hold_world.user()
        job = await hold_world.job(user)
        held = await hold_world.output(user, job, output_index=0)
        free = await hold_world.output(user, job, output_index=1)
        await _hold(hold_world, output_ref(held), user)

        async with hold_world.factory() as session:
            group = await LibraryService(session=session).get_group_detail(
                job.id, user.id, "vex", session=session
            )

        assert group is not None
        assert [(o.id, o.held) for o in group.outputs] == [(held.id, True), (free.id, False)]


def test_only_payloads_that_carry_expires_at_carry_held() -> None:
    """The rule of W5: lineage nodes show no countdown, so they have no flag to hide it with."""

    def fields(struct: type[msgspec.Struct]) -> set[str]:
        return set(struct.__struct_fields__)

    assert "expires_at" not in fields(LineageNode)
    assert "held" not in fields(LineageNode)
    assert {"expires_at", "held"} <= fields(LibraryOutputItem)


def test_held_is_a_required_field_so_no_builder_can_forget_it() -> None:
    from src.api.schemas.library import LibraryAssetDetail, LibraryAssetItem

    for struct in (LibraryAssetItem, LibraryAssetDetail, LibraryOutputItem):
        info = msgspec.inspect.type_info(struct)
        assert isinstance(info, msgspec.inspect.StructType)
        [held] = [f for f in info.fields if f.name == "held"]
        assert held.required, struct.__name__
