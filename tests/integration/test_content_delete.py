"""Integration tests for DELETE /v1/content/{content_id}.

Tests StorageRepository deletion methods and FK SET NULL behaviour
directly against a real PostgreSQL database (no R2 involved).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.repositories.storage import StorageRepository


class TestDeleteContentIntegration:
    """Integration tests for content deletion via StorageRepository."""

    async def test_delete_output_removes_db_record(
        self,
        storage_repo: StorageRepository,
        make_user: object,
        make_job: object,
    ) -> None:
        """Deleting an output removes the DB row."""
        user = await make_user(email=f"delout-{uuid4().hex[:6]}@example.com")  # type: ignore[operator]
        job = await make_job(user=user)  # type: ignore[operator]
        out_id = uuid4()
        await storage_repo.create_output(
            id=out_id,
            user_id=user.id,
            job_id=job.id,
            storage_key=f"users/{user.id}/outputs/{job.id}/{out_id}.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            output_index=0,
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
        )

        assert await storage_repo.delete_output(out_id, user_id=user.id) is True
        assert await storage_repo.get_output(out_id) is None

    async def test_delete_output_wrong_user_returns_false(
        self,
        storage_repo: StorageRepository,
        make_user: object,
        make_job: object,
    ) -> None:
        """Cannot delete another user's output."""
        user = await make_user(email=f"delout-own-{uuid4().hex[:6]}@example.com")  # type: ignore[operator]
        other = await make_user(email=f"delout-other-{uuid4().hex[:6]}@example.com")  # type: ignore[operator]
        job = await make_job(user=user)  # type: ignore[operator]
        out_id = uuid4()
        await storage_repo.create_output(
            id=out_id,
            user_id=user.id,
            job_id=job.id,
            storage_key=f"users/{user.id}/outputs/{job.id}/{out_id}.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            output_index=0,
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
        )

        assert await storage_repo.delete_output(out_id, user_id=other.id) is False
        # Record still exists
        assert await storage_repo.get_output(out_id) is not None

    async def test_delete_upload_removes_db_record(
        self,
        storage_repo: StorageRepository,
        make_user: object,
        make_user_image: object,
    ) -> None:
        """Deleting an upload removes the DB row."""
        user = await make_user(email=f"delup-{uuid4().hex[:6]}@example.com")  # type: ignore[operator]
        image = await make_user_image(user=user, size_bytes=500)  # type: ignore[operator]

        assert await storage_repo.delete_user_image(image.id, user_id=user.id) is True
        assert await storage_repo.get_user_image(image.id) is None

    async def test_delete_output_nullifies_lineage_fk(
        self,
        storage_repo: StorageRepository,
        make_user: object,
        make_job: object,
        db_session: AsyncSession,
    ) -> None:
        """Deleting an output used as source_output_id SETs NULL on the referencing job."""
        user = await make_user(email=f"lineage-{uuid4().hex[:6]}@example.com")  # type: ignore[operator]

        # Create source job + output
        source_job = await make_job(user=user)  # type: ignore[operator]
        out_id = uuid4()
        await storage_repo.create_output(
            id=out_id,
            user_id=user.id,
            job_id=source_job.id,
            storage_key=f"users/{user.id}/outputs/{source_job.id}/{out_id}.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            output_index=0,
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
        )

        # Create child job referencing the output as source
        child_job = await make_job(user=user)  # type: ignore[operator]
        child_job.source_output_id = out_id
        child_job.source_job_id = source_job.id
        await db_session.flush()

        # Delete the source output (system-level, no user_id filter)
        assert await storage_repo.delete_output(out_id) is True
        await db_session.flush()

        # Reload child job — source_output_id should be NULL
        await db_session.refresh(child_job)
        assert child_job.source_output_id is None
        # source_job_id is NOT nullified (it's a separate FK)
        assert child_job.source_job_id == source_job.id
