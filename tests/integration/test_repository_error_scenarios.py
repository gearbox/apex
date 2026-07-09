"""Cross-repository error and resilience integration tests.

These tests cover:
- Constraint violations (NOT NULL, CHECK, UNIQUE)
- FK violations
- Rollback behaviour after errors
- Large payload handling
- Timezone awareness across all DateTime columns
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.db.models.billing import TokenAccount, TokenTransaction
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage
from src.db.models.user import User

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repositories.billing import BillingRepository
    from src.db.repositories.job import JobRepository
    from src.db.repositories.output import OutputRepository
    from src.db.repositories.user import UserRepository
    from src.db.repositories.user_image import UserImageRepository

# ---------------------------------------------------------------------------
# FK constraint: GenerationOutput with non-existent job_id
# ---------------------------------------------------------------------------


async def test_output_fk_violation_raises(db_session: AsyncSession, make_user) -> None:
    """Inserting a GenerationOutput with a non-existent job_id raises IntegrityError."""
    user = await make_user(email=f"fk-{uuid4().hex[:6]}@example.com")
    out_id = uuid4()
    db_session.add(
        GenerationOutput(
            id=out_id,
            user_id=user.id,
            job_id=uuid4(),  # non-existent
            storage_key=f"users/{user.id}/outputs/fake/{out_id}.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            output_index=0,
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# ---------------------------------------------------------------------------
# CHECK constraint: TokenTransaction amount != 0
# ---------------------------------------------------------------------------


async def test_transaction_zero_amount_check_constraint(
    db_session: AsyncSession, make_user, make_token_account
) -> None:
    """Inserting a TokenTransaction with amount=0 violates the CHECK constraint."""
    user = await make_user(email=f"chk-{uuid4().hex[:6]}@example.com")
    account = await make_token_account(account_type="personal", user=user)
    db_session.add(
        TokenTransaction(
            id=uuid4(),
            account_id=account.id,
            transaction_type="credit",
            amount=0,
            balance_after=0,
            product_id="vex",
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# ---------------------------------------------------------------------------
# CHECK constraint: TokenAccount account_type consistency
# ---------------------------------------------------------------------------


async def test_personal_account_with_org_id_check_constraint(
    db_session: AsyncSession, make_user, make_org
) -> None:
    """A 'personal' account with organization_id set violates chk_account_owner."""
    user = await make_user(email=f"chkacct-{uuid4().hex[:6]}@example.com")
    org = await make_org()
    db_session.add(
        TokenAccount(
            id=uuid4(),
            account_type="personal",
            user_id=user.id,
            organization_id=org.id,  # invalid for personal
            product_id="vex",
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# ---------------------------------------------------------------------------
# Rollback behaviour: session remains usable after SAVEPOINT rollback
# ---------------------------------------------------------------------------


async def test_session_usable_after_savepoint_rollback(db_session: AsyncSession, make_user) -> None:
    """After an IntegrityError rolls back the current SAVEPOINT, the session can still execute queries."""
    user1 = await make_user(email=f"rb1-{uuid4().hex[:6]}@example.com")

    # Force an integrity error
    sp = await db_session.begin_nested()
    try:
        db_session.add(
            User(
                id=uuid4(),
                email=user1.email,  # duplicate — active unique index
                password_hash="x",
                product_id="vex",
            )
        )
        await db_session.flush()
        await sp.commit()
    except IntegrityError:
        await sp.rollback()

    # Session should still be functional
    result = await db_session.execute(select(User).where(User.id == user1.id))
    found = result.scalar_one_or_none()
    assert found is not None


# ---------------------------------------------------------------------------
# Unique constraint: Organization slug
# ---------------------------------------------------------------------------


async def test_duplicate_org_slug_raises(billing_repo: BillingRepository, make_user) -> None:
    """Creating two organizations with the same slug raises IntegrityError."""
    user = await make_user(email=f"slugdup-{uuid4().hex[:6]}@example.com")
    await billing_repo.create_organization(
        id=uuid4(), name="Org A", slug="dup-slug", owner_id=user.id, product_id="vex"
    )
    with pytest.raises(IntegrityError):
        await billing_repo.create_organization(
            id=uuid4(), name="Org B", slug="dup-slug", owner_id=user.id, product_id="vex"
        )


# ---------------------------------------------------------------------------
# Large payload: long prompt stored without truncation
# ---------------------------------------------------------------------------


async def test_large_prompt_stored_without_truncation(job_repo: JobRepository, make_user) -> None:
    """A very long prompt is stored and retrieved intact (Text column, no truncation)."""
    user = await make_user(email=f"bigprompt-{uuid4().hex[:6]}@example.com")
    long_prompt = "A" * 10_000
    job = await job_repo.create(
        id=uuid4(),
        user_id=user.id,
        name="Big Prompt Job",
        prompt=long_prompt,
        product_id="vex",
    )
    found = await job_repo.get(job.id)
    assert found is not None
    assert found.prompt == long_prompt
    assert len(found.prompt) == 10_000


# ---------------------------------------------------------------------------
# Timezone awareness: all DateTime(timezone=True) columns return tz-aware values
# ---------------------------------------------------------------------------


async def test_user_timestamps_are_timezone_aware(user_repo: UserRepository, make_user) -> None:
    """User.created_at and .updated_at are timezone-aware datetimes."""
    user = await make_user(email=f"tzuser-{uuid4().hex[:6]}@example.com")
    found = await user_repo.get_user(user.id)
    assert found is not None
    assert found.created_at.tzinfo is not None
    assert found.updated_at.tzinfo is not None


async def test_refresh_token_expires_at_is_timezone_aware(
    user_repo: UserRepository, make_user
) -> None:
    """RefreshToken.expires_at is timezone-aware."""
    user = await make_user(email=f"tztoken-{uuid4().hex[:6]}@example.com")
    expires = datetime.now(UTC) + timedelta(days=7)
    await user_repo.create_refresh_token(
        id=uuid4(),
        user_id=user.id,
        token_hash="tz_hash",
        family_id=uuid4(),
        expires_at=expires,
        product_id="vex",
    )
    token = await user_repo.get_refresh_token_by_hash("tz_hash")
    assert token is not None
    assert token.expires_at.tzinfo is not None


async def test_user_image_timestamps_are_timezone_aware(
    user_image_repo: UserImageRepository, make_user
) -> None:
    """UserImage.created_at and .expires_at are timezone-aware."""
    user = await make_user(email=f"tzimg-{uuid4().hex[:6]}@example.com")
    img_id = uuid4()
    image = await user_image_repo.create(
        id=img_id,
        user_id=user.id,
        storage_key=f"users/{user.id}/uploads/{img_id}.png",
        original_filename="tz.png",
        content_type="image/png",
        size_bytes=512,
        format="png",
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    found = await user_image_repo.get(image.id)
    assert found is not None
    assert found.created_at.tzinfo is not None
    assert found.expires_at.tzinfo is not None


async def test_generation_output_timestamps_are_timezone_aware(
    output_repo: OutputRepository, make_user, make_job
) -> None:
    """GenerationOutput.created_at and .expires_at are timezone-aware."""
    user = await make_user(email=f"tzout-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    out_id = uuid4()
    output = await output_repo.create(
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
    found = await output_repo.get(output.id)
    assert found is not None
    assert found.created_at.tzinfo is not None
    assert found.expires_at.tzinfo is not None


# ---------------------------------------------------------------------------
# NOT NULL violation: missing required field
# ---------------------------------------------------------------------------


async def test_user_image_missing_storage_key_raises(db_session: AsyncSession, make_user) -> None:
    """Inserting a UserImage without storage_key raises IntegrityError (NOT NULL)."""
    user = await make_user(email=f"notnull-{uuid4().hex[:6]}@example.com")
    db_session.add(
        UserImage(
            id=uuid4(),
            user_id=user.id,
            storage_key=None,  # type: ignore[arg-type]  # intentional NULL
            original_filename="x.png",
            content_type="image/png",
            size_bytes=100,
            format="png",
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# ---------------------------------------------------------------------------
# Unique constraint: RefreshToken.token_hash is globally unique
# ---------------------------------------------------------------------------


async def test_duplicate_refresh_token_hash_raises(user_repo: UserRepository, make_user) -> None:
    """Two refresh tokens with the same hash raise IntegrityError (unique constraint)."""
    user = await make_user(email=f"duprt-{uuid4().hex[:6]}@example.com")
    expires = datetime.now(UTC) + timedelta(days=7)
    await user_repo.create_refresh_token(
        id=uuid4(),
        user_id=user.id,
        token_hash="shared_hash",
        family_id=uuid4(),
        expires_at=expires,
        product_id="vex",
    )
    with pytest.raises(IntegrityError):
        await user_repo.create_refresh_token(
            id=uuid4(),
            user_id=user.id,
            token_hash="shared_hash",
            family_id=uuid4(),
            expires_at=expires,
            product_id="vex",
        )


# ---------------------------------------------------------------------------
# Unique constraint: UserImage.storage_key
# ---------------------------------------------------------------------------


async def test_duplicate_storage_key_for_output_raises(
    output_repo: OutputRepository, make_user, make_job
) -> None:
    """Two GenerationOutputs with the same storage_key raise IntegrityError."""
    user = await make_user(email=f"dupkey-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    key = f"users/{user.id}/outputs/{job.id}/shared.png"
    out_id1 = uuid4()
    await output_repo.create(
        id=out_id1,
        user_id=user.id,
        job_id=job.id,
        storage_key=key,
        content_type="image/png",
        size_bytes=100,
        format="png",
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    with pytest.raises(IntegrityError):
        await output_repo.create(
            id=uuid4(),
            user_id=user.id,
            job_id=job.id,
            storage_key=key,  # duplicate
            content_type="image/png",
            size_bytes=100,
            format="png",
            output_index=1,
            expires_at=datetime.now(UTC) + timedelta(days=7),
            product_id="vex",
        )


# ---------------------------------------------------------------------------
# Cascade delete: User → all child rows
# ---------------------------------------------------------------------------


async def test_cascade_delete_user_removes_all_owned_rows(
    db_session: AsyncSession, make_user, make_job, make_user_image
) -> None:
    """Deleting a User cascades to GenerationJob and UserImage rows."""
    user = await make_user(email=f"alldel-{uuid4().hex[:6]}@example.com")
    job = await make_job(user=user)
    image = await make_user_image(user=user)

    await db_session.delete(user)
    await db_session.flush()

    job_result = await db_session.execute(select(GenerationJob).where(GenerationJob.id == job.id))
    assert job_result.scalar_one_or_none() is None

    img_result = await db_session.execute(select(UserImage).where(UserImage.id == image.id))
    assert img_result.scalar_one_or_none() is None
