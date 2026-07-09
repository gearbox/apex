"""Integration tests for DELETE /v1/content/{content_id}.

Tests OutputRepository/UserImageRepository deletion methods and FK SET NULL
behaviour directly against a real PostgreSQL database (no R2 involved).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repositories.output import OutputRepository
    from src.db.repositories.user_image import UserImageRepository


class TestDeleteContentIntegration:
    """Integration tests for content deletion."""

    async def test_delete_output_removes_db_record(
        self,
        output_repo: OutputRepository,
        make_user: object,
        make_job: object,
    ) -> None:
        """Deleting an output removes the DB row."""
        user = await make_user(email=f"delout-{uuid4().hex[:6]}@example.com")  # type: ignore[operator]
        job = await make_job(user=user)  # type: ignore[operator]
        out_id = uuid4()
        await output_repo.create(
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

        assert await output_repo.delete(out_id, user_id=user.id) is True
        assert await output_repo.get(out_id) is None

    async def test_delete_output_wrong_user_returns_false(
        self,
        output_repo: OutputRepository,
        make_user: object,
        make_job: object,
    ) -> None:
        """Cannot delete another user's output."""
        user = await make_user(email=f"delout-own-{uuid4().hex[:6]}@example.com")  # type: ignore[operator]
        other = await make_user(email=f"delout-other-{uuid4().hex[:6]}@example.com")  # type: ignore[operator]
        job = await make_job(user=user)  # type: ignore[operator]
        out_id = uuid4()
        await output_repo.create(
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

        assert await output_repo.delete(out_id, user_id=other.id) is False
        # Record still exists
        assert await output_repo.get(out_id) is not None

    async def test_delete_upload_removes_db_record(
        self,
        user_image_repo: UserImageRepository,
        make_user: object,
        make_user_image: object,
    ) -> None:
        """Deleting an upload removes the DB row."""
        user = await make_user(email=f"delup-{uuid4().hex[:6]}@example.com")  # type: ignore[operator]
        image = await make_user_image(user=user, size_bytes=500)  # type: ignore[operator]

        assert await user_image_repo.delete(image.id, user_id=user.id) is True
        assert await user_image_repo.get(image.id) is None

    async def test_delete_output_nullifies_lineage_fk(
        self,
        output_repo: OutputRepository,
        make_user: object,
        make_job: object,
        db_session: AsyncSession,
    ) -> None:
        """Deleting an output used as source_output_id SETs NULL on the referencing job."""
        user = await make_user(email=f"lineage-{uuid4().hex[:6]}@example.com")  # type: ignore[operator]

        # Create source job + output
        source_job = await make_job(user=user)  # type: ignore[operator]
        out_id = uuid4()
        await output_repo.create(
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
        assert await output_repo.delete(out_id) is True
        await db_session.flush()

        # Reload child job — source_output_id should be NULL
        await db_session.refresh(child_job)
        assert child_job.source_output_id is None
        # source_job_id is NOT nullified (it's a separate FK)
        assert child_job.source_job_id == source_job.id
