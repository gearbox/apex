"""Unit tests for UserImageRepository with mocked SQLAlchemy sessions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.db.repositories.user_image import UserImageRepository

pytestmark = pytest.mark.unit


def _make_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.delete = AsyncMock()
    return session


def _scalars_result(items: list) -> MagicMock:
    r = MagicMock()
    r.scalars.return_value.all.return_value = items
    return r


def _scalar_one_result(item: object) -> MagicMock:
    r = MagicMock()
    r.scalar_one_or_none.return_value = item
    return r


def _one_result(row: tuple) -> MagicMock:
    r = MagicMock()
    r.one.return_value = row
    return r


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestUserImageRepositoryCreate:
    async def test_create_adds_to_session_and_flushes(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        img_id = uuid4()
        user_id = uuid4()

        img = await repo.create(
            id=img_id,
            user_id=user_id,
            storage_key=f"users/{user_id}/uploads/{img_id}.png",
            original_filename="photo.png",
            content_type="image/png",
            size_bytes=1024,
            format="png",
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
        )

        session.add.assert_called_once()
        session.flush.assert_awaited_once()
        assert img.id == img_id
        assert img.user_id == user_id

    async def test_create_thumbnail_sets_parent_fields(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        parent_id = uuid4()
        img = await repo.create(
            id=uuid4(),
            user_id=uuid4(),
            storage_key="users/u/uploads/thumb.webp",
            original_filename="thumb_sm_photo.png",
            content_type="image/webp",
            size_bytes=256,
            format="webp",
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
            is_thumbnail=True,
            parent_image_id=parent_id,
            thumbnail_max_edge=150,
            width=100,
            height=75,
        )

        assert img.is_thumbnail is True
        assert img.parent_image_id == parent_id
        assert img.thumbnail_max_edge == 150
        assert img.width == 100
        assert img.height == 75


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestUserImageRepositoryGet:
    async def test_get_with_user_id_calls_execute(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        img = MagicMock()
        session.execute = AsyncMock(return_value=_scalar_one_result(img))

        result = await repo.get(uuid4(), user_id=uuid4())

        assert result is img
        session.execute.assert_awaited_once()

    async def test_get_without_user_id_uses_session_get(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        img = MagicMock()
        session.get = AsyncMock(return_value=img)

        result = await repo.get(uuid4())

        assert result is img
        session.get.assert_awaited_once()

    async def test_get_returns_none_when_not_found(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        session.execute = AsyncMock(return_value=_scalar_one_result(None))

        result = await repo.get(uuid4(), user_id=uuid4())

        assert result is None


# ---------------------------------------------------------------------------
# get_by_key
# ---------------------------------------------------------------------------


class TestUserImageRepositoryGetByKey:
    async def test_returns_image_for_matching_key(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        img = MagicMock()
        session.execute = AsyncMock(return_value=_scalar_one_result(img))

        result = await repo.get_by_key("users/u/uploads/id.png")

        assert result is img

    async def test_returns_none_when_key_not_found(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        session.execute = AsyncMock(return_value=_scalar_one_result(None))

        result = await repo.get_by_key("nonexistent")

        assert result is None


# ---------------------------------------------------------------------------
# list_by_user
# ---------------------------------------------------------------------------


class TestUserImageRepositoryListByUser:
    async def test_returns_images_for_user(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        img = MagicMock()
        session.execute = AsyncMock(return_value=_scalars_result([img]))

        results = await repo.list_by_user(uuid4(), limit=10)

        assert list(results) == [img]

    async def test_with_cursor_executes_query(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        session.execute = AsyncMock(return_value=_scalars_result([]))

        results = await repo.list_by_user(
            uuid4(),
            limit=10,
            cursor_ts=datetime.now(UTC),
            cursor_id=uuid4(),
        )

        assert not list(results)
        session.execute.assert_awaited_once()

    async def test_returns_empty_list_when_no_images(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        session.execute = AsyncMock(return_value=_scalars_result([]))

        results = await repo.list_by_user(uuid4())

        assert not list(results)


# ---------------------------------------------------------------------------
# list_derivatives
# ---------------------------------------------------------------------------


class TestUserImageRepositoryListDerivatives:
    async def test_returns_thumbnail_rows(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        thumb = MagicMock()
        session.execute = AsyncMock(return_value=_scalars_result([thumb]))

        results = await repo.list_derivatives(uuid4())

        assert list(results) == [thumb]

    async def test_returns_empty_when_no_derivatives(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        session.execute = AsyncMock(return_value=_scalars_result([]))

        results = await repo.list_derivatives(uuid4())

        assert not list(results)


# ---------------------------------------------------------------------------
# batch_derivatives
# ---------------------------------------------------------------------------


class TestUserImageRepositoryBatchDerivatives:
    async def test_empty_parent_ids_returns_empty_dict_without_query(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        result = await repo.batch_derivatives([])

        assert result == {}
        session.execute.assert_not_awaited()

    async def test_groups_thumbnails_by_parent_id(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        parent_id = uuid4()
        thumb = MagicMock()
        thumb.parent_image_id = parent_id

        session.execute = AsyncMock(return_value=_scalars_result([thumb]))

        result = await repo.batch_derivatives([parent_id])

        assert parent_id in result
        assert thumb in result[parent_id]

    async def test_skips_rows_with_null_parent_id(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        thumb = MagicMock()
        thumb.parent_image_id = None

        session.execute = AsyncMock(return_value=_scalars_result([thumb]))

        result = await repo.batch_derivatives([uuid4()])

        assert result == {}

    async def test_multiple_parents_grouped_correctly(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        parent_a = uuid4()
        parent_b = uuid4()
        thumb_a = MagicMock()
        thumb_a.parent_image_id = parent_a
        thumb_b = MagicMock()
        thumb_b.parent_image_id = parent_b

        session.execute = AsyncMock(return_value=_scalars_result([thumb_a, thumb_b]))

        result = await repo.batch_derivatives([parent_a, parent_b])

        assert len(result[parent_a]) == 1
        assert len(result[parent_b]) == 1


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestUserImageRepositoryDelete:
    async def test_delete_returns_true_and_removes_record(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        img = MagicMock()
        session.get = AsyncMock(return_value=img)

        result = await repo.delete(uuid4())

        assert result is True
        session.delete.assert_awaited_once_with(img)
        session.flush.assert_awaited_once()

    async def test_delete_with_user_id_uses_execute(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        img = MagicMock()
        session.execute = AsyncMock(return_value=_scalar_one_result(img))

        result = await repo.delete(uuid4(), user_id=uuid4())

        assert result is True

    async def test_delete_returns_false_when_not_found(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        session.get = AsyncMock(return_value=None)

        result = await repo.delete(uuid4())

        assert result is False
        session.delete.assert_not_awaited()


# ---------------------------------------------------------------------------
# get_expired
# ---------------------------------------------------------------------------


class TestUserImageRepositoryGetExpired:
    async def test_returns_expired_images(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        expired = MagicMock()
        session.execute = AsyncMock(return_value=_scalars_result([expired]))

        results = await repo.get_expired()

        assert list(results) == [expired]

    async def test_accepts_explicit_before_date(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        session.execute = AsyncMock(return_value=_scalars_result([]))

        before = datetime.now(UTC) - timedelta(hours=2)
        results = await repo.get_expired(before=before)

        assert not list(results)
        session.execute.assert_awaited_once()

    async def test_returns_empty_when_nothing_expired(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        session.execute = AsyncMock(return_value=_scalars_result([]))

        results = await repo.get_expired()

        assert not list(results)


# ---------------------------------------------------------------------------
# count_and_sum_by_user
# ---------------------------------------------------------------------------


class TestUserImageRepositoryCountAndSum:
    async def test_returns_correct_count_and_bytes(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        session.execute = AsyncMock(return_value=_one_result((5, 10000)))

        count, total = await repo.count_and_sum_by_user(uuid4())

        assert count == 5
        assert total == 10000

    async def test_returns_zeros_for_user_with_no_uploads(self) -> None:
        session = _make_session()
        repo = UserImageRepository(session)

        session.execute = AsyncMock(return_value=_one_result((0, 0)))

        count, total = await repo.count_and_sum_by_user(uuid4())

        assert count == 0
        assert total == 0
