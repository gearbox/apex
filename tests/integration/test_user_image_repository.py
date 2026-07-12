"""Integration tests for UserImageRepository against a real PostgreSQL database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

if TYPE_CHECKING:
    from src.db.repositories.user_image import UserImageRepository


async def test_create_user_image(user_image_repo: UserImageRepository, make_user) -> None:
    """create persists and returns a UserImage."""
    user = await make_user(email=f"imgcreate-{uuid4().hex[:6]}@example.com")
    img_id = uuid4()
    image = await user_image_repo.create(
        id=img_id,
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/{img_id}.png",
        original_filename="photo.png",
        content_type="image/png",
        size_bytes=2048,
        format="png",
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    assert image.id == img_id
    assert image.user_id == user.id
    assert image.size_bytes == 2048


async def test_create_user_image_duplicate_key_raises(
    user_image_repo: UserImageRepository, make_user
) -> None:
    """Creating two UserImages with the same storage_key raises IntegrityError."""
    user = await make_user(email=f"imgdup-{uuid4().hex[:6]}@example.com")
    key = f"users/{user.id}/uploads/dup.png"
    await user_image_repo.create(
        id=uuid4(),
        user_id=user.id,
        storage_key=key,
        original_filename="dup.png",
        content_type="image/png",
        size_bytes=100,
        format="png",
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    with pytest.raises(IntegrityError):
        await user_image_repo.create(
            id=uuid4(),
            user_id=user.id,
            storage_key=key,
            original_filename="dup.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
        )


async def test_get_user_image_found(user_image_repo: UserImageRepository, make_user_image) -> None:
    """get returns the image by PK."""
    image = await make_user_image()
    found = await user_image_repo.get(image.id)
    assert found is not None
    assert found.id == image.id


async def test_get_user_image_not_found(user_image_repo: UserImageRepository) -> None:
    """get returns None for an unknown UUID."""
    assert await user_image_repo.get(uuid4()) is None


async def test_get_user_image_ownership_enforced(
    user_image_repo: UserImageRepository, make_user_image, make_user
) -> None:
    """get with wrong user_id returns None."""
    image = await make_user_image()
    other_user = await make_user(email=f"other-{uuid4().hex[:6]}@example.com")
    found = await user_image_repo.get(image.id, user_id=other_user.id)
    assert found is None


async def test_get_by_key(user_image_repo: UserImageRepository, make_user) -> None:
    """get_by_key returns image by storage key."""
    user = await make_user(email=f"imgkey-{uuid4().hex[:6]}@example.com")
    img_id = uuid4()
    key = f"users/{user.id}/uploads/{img_id}.png"
    await user_image_repo.create(
        id=img_id,
        user_id=user.id,
        storage_key=key,
        original_filename="x.png",
        content_type="image/png",
        size_bytes=512,
        format="png",
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    found = await user_image_repo.get_by_key(key)
    assert found is not None
    assert found.storage_key == key


async def test_list_by_user_paginated(
    user_image_repo: UserImageRepository, make_user, make_user_image
) -> None:
    """list_by_user returns paginated results using limit+1 fetch pattern."""
    user = await make_user(email=f"imglist-{uuid4().hex[:6]}@example.com")
    for i in range(5):
        await make_user_image(user=user, storage_key=f"users/{user.id}/uploads/{i}.png")
    images = await user_image_repo.list_by_user(user.id, limit=3)
    assert len(images) == 4  # 3+1 since 5 > 3, has_more=True


async def test_list_by_user_empty_for_new_user(
    user_image_repo: UserImageRepository, make_user
) -> None:
    """list_by_user returns empty list for a user with no uploads."""
    user = await make_user(email=f"imglistoff-{uuid4().hex[:6]}@example.com")
    images = await user_image_repo.list_by_user(user.id)
    assert not list(images)


async def test_delete_user_image(user_image_repo: UserImageRepository, make_user_image) -> None:
    """delete removes the row and returns True."""
    image = await make_user_image()
    result = await user_image_repo.delete(image.id)
    assert result is True
    assert await user_image_repo.get(image.id) is None


async def test_delete_user_image_not_found_returns_false(
    user_image_repo: UserImageRepository,
) -> None:
    """delete returns False for an unknown image."""
    assert await user_image_repo.delete(uuid4()) is False


async def test_get_expired(user_image_repo: UserImageRepository, make_user) -> None:
    """get_expired returns images past their expires_at."""
    user = await make_user(email=f"expiredimg-{uuid4().hex[:6]}@example.com")
    past = datetime.now(UTC) - timedelta(hours=1)
    img_id = uuid4()
    await user_image_repo.create(
        id=img_id,
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/{img_id}.png",
        original_filename="old.png",
        content_type="image/png",
        size_bytes=100,
        format="png",
        expires_at=past,
        product_id="vex",
    )
    expired = await user_image_repo.get_expired()
    assert any(img.id == img_id for img in expired)


async def test_count_and_sum_by_user_empty(user_image_repo: UserImageRepository, make_user) -> None:
    """count_and_sum_by_user returns zeros for a user with no uploads."""
    user = await make_user(email=f"statsempty-{uuid4().hex[:6]}@example.com")
    count, total_bytes = await user_image_repo.count_and_sum_by_user(user.id)
    assert count == 0
    assert total_bytes == 0


async def test_count_and_sum_by_user_with_data(
    user_image_repo: UserImageRepository, make_user, make_user_image
) -> None:
    """count_and_sum_by_user returns correct count and byte total."""
    user = await make_user(email=f"statsdata-{uuid4().hex[:6]}@example.com")
    await make_user_image(user=user, size_bytes=1000)
    await make_user_image(user=user, size_bytes=2000)

    count, total_bytes = await user_image_repo.count_and_sum_by_user(user.id)
    assert count == 2
    assert total_bytes == 3000


async def test_touch_expiry_updates_full_row_and_thumbnails(
    user_image_repo: UserImageRepository, make_user
) -> None:
    """touch_expiry bumps the full upload and all its derivative thumbnails."""
    user = await make_user(email=f"touchexpiry-{uuid4().hex[:6]}@example.com")
    old_expiry = datetime.now(UTC) + timedelta(days=1)
    parent_id = uuid4()
    await user_image_repo.create(
        id=parent_id,
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/{parent_id}.png",
        original_filename="parent.png",
        content_type="image/png",
        size_bytes=1024,
        format="png",
        expires_at=old_expiry,
        product_id="vex",
    )
    thumb_id = uuid4()
    await user_image_repo.create(
        id=thumb_id,
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/{thumb_id}.webp",
        original_filename="parent.png",
        content_type="image/webp",
        size_bytes=256,
        format="webp",
        expires_at=old_expiry,
        product_id="vex",
        is_thumbnail=True,
        parent_image_id=parent_id,
        thumbnail_max_edge=150,
    )

    new_expiry = datetime.now(UTC) + timedelta(days=7)
    result = await user_image_repo.touch_expiry(parent_id, user_id=user.id, expires_at=new_expiry)

    assert result is True
    parent = await user_image_repo.get(parent_id)
    thumb = await user_image_repo.get(thumb_id)
    assert parent is not None
    assert thumb is not None
    assert parent.expires_at == new_expiry
    assert thumb.expires_at == new_expiry


async def test_touch_expiry_wrong_user_returns_false(
    user_image_repo: UserImageRepository, make_user_image, make_user
) -> None:
    """touch_expiry scoped to another user's id leaves the row untouched."""
    old_expiry = datetime.now(UTC) + timedelta(days=1)
    image = await make_user_image(expires_at=old_expiry)
    other_user = await make_user(email=f"touchother-{uuid4().hex[:6]}@example.com")

    result = await user_image_repo.touch_expiry(
        image.id, user_id=other_user.id, expires_at=datetime.now(UTC) + timedelta(days=7)
    )

    assert result is False
    reloaded = await user_image_repo.get(image.id)
    assert reloaded is not None
    assert reloaded.expires_at == old_expiry


async def test_touch_expiry_missing_returns_false(
    user_image_repo: UserImageRepository, make_user
) -> None:
    """touch_expiry on a non-existent image id returns False."""
    user = await make_user(email=f"touchmissing-{uuid4().hex[:6]}@example.com")

    result = await user_image_repo.touch_expiry(
        uuid4(), user_id=user.id, expires_at=datetime.now(UTC) + timedelta(days=7)
    )

    assert result is False


async def test_touch_expiry_thumbnail_id_returns_false(
    user_image_repo: UserImageRepository, make_user
) -> None:
    """Passing a thumbnail row's own id is rejected — full uploads only."""
    user = await make_user(email=f"touchthumb-{uuid4().hex[:6]}@example.com")
    old_expiry = datetime.now(UTC) + timedelta(days=1)
    parent_id = uuid4()
    await user_image_repo.create(
        id=parent_id,
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/{parent_id}.png",
        original_filename="parent.png",
        content_type="image/png",
        size_bytes=1024,
        format="png",
        expires_at=old_expiry,
        product_id="vex",
    )
    thumb_id = uuid4()
    await user_image_repo.create(
        id=thumb_id,
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/{thumb_id}.webp",
        original_filename="parent.png",
        content_type="image/webp",
        size_bytes=256,
        format="webp",
        expires_at=old_expiry,
        product_id="vex",
        is_thumbnail=True,
        parent_image_id=parent_id,
        thumbnail_max_edge=150,
    )

    result = await user_image_repo.touch_expiry(
        thumb_id, user_id=user.id, expires_at=datetime.now(UTC) + timedelta(days=7)
    )

    assert result is False
    parent = await user_image_repo.get(parent_id)
    thumb = await user_image_repo.get(thumb_id)
    assert parent is not None
    assert thumb is not None
    assert parent.expires_at == old_expiry
    assert thumb.expires_at == old_expiry


async def test_expired_images_count_accuracy(
    user_image_repo: UserImageRepository, make_user
) -> None:
    """Expired image query returns exactly N expired items regardless of non-expired ones."""
    user = await make_user(email=f"expcount-{uuid4().hex[:6]}@example.com")
    past = datetime.now(UTC) - timedelta(hours=2)
    future = datetime.now(UTC) + timedelta(days=7)

    expired_ids = set()
    for i in range(3):
        img_id = uuid4()
        await user_image_repo.create(
            id=img_id,
            user_id=user.id,
            storage_key=f"users/{user.id}/uploads/exp{i}.png",
            original_filename=f"exp{i}.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            expires_at=past,
            product_id="vex",
        )
        expired_ids.add(img_id)

    for i in range(2):
        img_id = uuid4()
        await user_image_repo.create(
            id=img_id,
            user_id=user.id,
            storage_key=f"users/{user.id}/uploads/fresh{i}.png",
            original_filename=f"fresh{i}.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            expires_at=future,
            product_id="vex",
        )

    expired = await user_image_repo.get_expired()
    expired_from_this_user = {img.id for img in expired if img.user_id == user.id}
    assert expired_from_this_user == expired_ids
