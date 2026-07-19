"""Integration tests for library_asset_metadata lazy upsert + purge behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.services.content_retention import ContentRetentionService
from src.core.library_ref import LibraryAssetSource
from src.db.models.library import LibraryAssetMetadata
from src.db.models.storage import UserImage
from src.db.models.user import User
from src.db.repositories.library import LibraryRepository

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from sqlalchemy.ext.asyncio import AsyncEngine


@pytest_asyncio.fixture
async def library_repo(db_session: AsyncSession) -> LibraryRepository:
    return LibraryRepository(db_session)


async def _create_user_committed(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    product_id: str = "vex",
) -> User:
    """Seed a User via a real committing session bound to the shared engine.

    Needed whenever the assertion crosses connections (e.g. the retention
    sweeper's own session_factory, or a second concurrent session) — the
    SAVEPOINT-scoped ``db_session``/``make_user`` fixtures never commit
    their outer transaction, so writes made through them are invisible
    outside that one connection.
    """
    async with session_factory() as session:
        user = User(
            id=uuid4(),
            email=f"libmeta-{uuid4().hex[:8]}@example.com",
            password_hash="hashed",
            product_id=product_id,
        )
        session.add(user)
        await session.commit()
        return user


async def _create_upload_committed(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user: User,
    expires_at: datetime,
) -> UserImage:
    async with session_factory() as session:
        img_id = uuid4()
        image = UserImage(
            id=img_id,
            user_id=user.id,
            product_id=user.product_id,
            storage_key=f"users/{user.id}/uploads/{img_id}.png",
            original_filename="photo.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            expires_at=expires_at,
        )
        session.add(image)
        await session.commit()
        return image


class TestLazyUpsertIdempotency:
    async def test_first_call_creates_row(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        user = await make_user(email=f"lazy-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)

        row = await library_repo.upsert_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, image.id, is_favorite=True
        )
        assert row.is_favorite is True
        assert row.asset_id == image.id

    async def test_repeated_calls_converge_to_one_row(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"conv-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)

        await library_repo.upsert_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, image.id, is_favorite=True
        )
        await library_repo.upsert_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, image.id, is_favorite=False
        )
        row = await library_repo.upsert_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, image.id, display_title="Sunset"
        )

        assert row.is_favorite is False
        assert row.display_title == "Sunset"

        result = await db_session.execute(
            select(LibraryAssetMetadata).where(
                LibraryAssetMetadata.user_id == user.id,
                LibraryAssetMetadata.asset_id == image.id,
            )
        )
        rows = result.scalars().all()
        assert len(rows) == 1

    async def test_unset_leaves_field_unchanged(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        user = await make_user(email=f"unset-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)

        await library_repo.upsert_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, image.id, display_title="Original"
        )
        row = await library_repo.upsert_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, image.id, is_favorite=True
        )
        assert row.display_title == "Original"
        assert row.is_favorite is True


class TestConcurrentUpsert:
    async def test_two_sessions_converge_to_one_row(
        self,
        db_engine: AsyncEngine,
    ) -> None:
        """Two independent connections (NullPool) racing upsert_metadata for
        the same asset must converge on a single row via ON CONFLICT DO
        UPDATE, never a duplicate-key error or two rows."""
        session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
        user = await _create_user_committed(session_factory)
        image = await _create_upload_committed(
            session_factory, user=user, expires_at=datetime.now(UTC) + timedelta(days=7)
        )

        session1 = AsyncSession(bind=db_engine, expire_on_commit=False)
        session2 = AsyncSession(bind=db_engine, expire_on_commit=False)
        try:
            await LibraryRepository(session1).upsert_metadata(
                user.id, "vex", LibraryAssetSource.UPLOAD, image.id, is_favorite=True
            )
            await session1.commit()

            row2 = await LibraryRepository(session2).upsert_metadata(
                user.id, "vex", LibraryAssetSource.UPLOAD, image.id, display_title="From session2"
            )
            await session2.commit()

            assert row2.is_favorite is True
            assert row2.display_title == "From session2"

            async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify_session:
                result = await verify_session.execute(
                    select(LibraryAssetMetadata).where(
                        LibraryAssetMetadata.user_id == user.id,
                        LibraryAssetMetadata.asset_id == image.id,
                    )
                )
                assert len(result.scalars().all()) == 1
        finally:
            await session1.close()
            await session2.close()


class TestDeletePurgesMetadata:
    async def test_delete_metadata_for_assets_removes_row(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        user = await make_user(email=f"delmeta-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        await library_repo.upsert_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, image.id, is_favorite=True
        )

        count = await library_repo.delete_metadata_for_assets([("upload", image.id)])
        assert count == 1

        remaining = await library_repo.get_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, image.id
        )
        assert remaining is None

    async def test_delete_metadata_for_assets_empty_pairs_noop(
        self,
        library_repo: LibraryRepository,
    ) -> None:
        assert await library_repo.delete_metadata_for_assets([]) == 0


class TestRetentionSweepPurgesMetadata:
    async def test_sweep_purges_metadata_for_expired_upload(
        self,
        db_engine: AsyncEngine,
    ) -> None:
        """ContentRetentionService.sweep() must purge library_asset_metadata
        rows for uploads it deletes as expired — the sweeper uses its own
        session_factory, so seed via a real commit on the shared engine."""
        session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
        user = await _create_user_committed(session_factory)
        expired_image = await _create_upload_committed(
            session_factory, user=user, expires_at=datetime.now(UTC) - timedelta(days=1)
        )
        async with session_factory() as session:
            await LibraryRepository(session).upsert_metadata(
                user.id, "vex", LibraryAssetSource.UPLOAD, expired_image.id, is_favorite=True
            )
            await session.commit()

        mock_storage = MagicMock()
        mock_storage.delete_many = AsyncMock(return_value=1)

        service = ContentRetentionService(
            session_factory=session_factory,
            storage=mock_storage,
            batch_size=100,
            max_batches_per_run=5,
        )
        await service.sweep()

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as verify_session:
            remaining = await LibraryRepository(verify_session).get_metadata(
                user.id, "vex", LibraryAssetSource.UPLOAD, expired_image.id
            )
            assert remaining is None
