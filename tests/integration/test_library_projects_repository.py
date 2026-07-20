"""Integration tests for Library Phase 2 projects against a real database.

Covers: LibraryProjectRepository CRUD + owner/product isolation, name
uniqueness conflicts (via LibraryProjectService, which wraps the race-safe
SAVEPOINT pattern), ON DELETE SET NULL on project deletion, the
``list_assets`` project_id/expiring/query filters (both UNION branches),
and ``expiring_soon`` sort pagination stability.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

from src.api.services.library_project import LibraryProjectNameConflictError, LibraryProjectService
from src.core.enums import LibrarySort
from src.core.library_ref import LibraryAssetSource
from src.db.models.library import LibraryAssetMetadata
from src.db.models.storage import GenerationJob, GenerationOutput
from src.db.repositories.library import LibraryRepository
from src.db.repositories.library_project import LibraryProjectRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.user import User

OutputFactory = Callable[..., Coroutine[Any, Any, GenerationOutput]]


@pytest_asyncio.fixture
async def project_repo(db_session: AsyncSession) -> LibraryProjectRepository:
    return LibraryProjectRepository(db_session)


@pytest_asyncio.fixture
async def library_repo(db_session: AsyncSession) -> LibraryRepository:
    return LibraryRepository(db_session)


@pytest_asyncio.fixture
async def project_service(db_session: AsyncSession) -> LibraryProjectService:
    return LibraryProjectService(session=db_session)


@pytest_asyncio.fixture
async def make_output(
    db_session: AsyncSession,
    make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
    make_user: Callable[..., Coroutine[Any, Any, User]],
) -> OutputFactory:
    """Factory fixture: create a GenerationOutput row and flush it."""

    async def _factory(
        *,
        job: GenerationJob | None = None,
        user: User | None = None,
        content_type: str = "image/jpeg",
        product_id: str = "vex",
        expires_at: datetime | None = None,
        output_id: object = None,
    ) -> GenerationOutput:
        if user is None:
            user = await make_user(email=f"projout-{uuid4().hex[:8]}@example.com")
        if job is None:
            job = await make_job(user=user, status="completed", product_id=product_id)
        oid = output_id or uuid4()
        out = GenerationOutput(
            id=oid,
            user_id=user.id,
            job_id=job.id,
            storage_key=f"users/{user.id}/outputs/{job.id}/{oid}.jpeg",
            content_type=content_type,
            size_bytes=1000,
            format="jpeg",
            output_index=0,
            expires_at=expires_at or (datetime.now(UTC) + timedelta(days=7)),
            is_thumbnail=False,
            product_id=product_id,
        )
        db_session.add(out)
        await db_session.flush()
        return out

    return _factory


class TestBatchLookupScoping:
    """L1 — batch_names/batch_asset_counts must be scoped to (user_id,
    product_id), matching the module docstring's claim that every method
    takes both. A foreign-owner project id must never resolve."""

    async def test_batch_names_excludes_foreign_owner_project(
        self,
        project_repo: LibraryProjectRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
    ) -> None:
        owner = await make_user(email=f"batchnameowner-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"batchnameother-{uuid4().hex[:8]}@example.com")

        owned = await project_repo.create(
            user_id=owner.id, product_id="vex", name="Owned", description=None
        )
        foreign = await project_repo.create(
            user_id=other.id, product_id="vex", name="Foreign", description=None
        )

        names = await project_repo.batch_names(
            [owned.id, foreign.id], user_id=owner.id, product_id="vex"
        )
        assert names == {owned.id: "Owned"}

    async def test_batch_names_excludes_foreign_product(
        self,
        project_repo: LibraryProjectRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
    ) -> None:
        user = await make_user(email=f"batchnameprod-{uuid4().hex[:8]}@example.com")
        project = await project_repo.create(
            user_id=user.id, product_id="vex", name="Vex Project", description=None
        )

        names = await project_repo.batch_names([project.id], user_id=user.id, product_id="synthara")
        assert names == {}

    async def test_batch_asset_counts_excludes_foreign_owner_metadata(
        self,
        project_repo: LibraryProjectRepository,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        owner = await make_user(email=f"batchcntowner-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"batchcntother-{uuid4().hex[:8]}@example.com")

        # A project owned by `owner`, with a metadata row assigned to it
        # from another user's asset — not something that should normally
        # happen (project ownership is checked before assignment), but the
        # count query must be scoped by (user_id, product_id) regardless of
        # what rows exist, per the module docstring's structural guarantee.
        project = await project_repo.create(
            user_id=owner.id, product_id="vex", name="Owner Project", description=None
        )
        other_image = await make_user_image(user=other)
        await library_repo.upsert_metadata(
            other.id, "vex", LibraryAssetSource.UPLOAD, other_image.id, project_id=project.id
        )

        counts = await project_repo.batch_asset_counts(
            [project.id], user_id=owner.id, product_id="vex"
        )
        assert counts == {}

        counts_as_other = await project_repo.batch_asset_counts(
            [project.id], user_id=other.id, product_id="vex"
        )
        assert counts_as_other == {project.id: 1}


class TestProjectCRUDIsolation:
    async def test_create_and_get(
        self,
        project_repo: LibraryProjectRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
    ) -> None:
        user = await make_user(email=f"projcrud-{uuid4().hex[:8]}@example.com")
        project = await project_repo.create(
            user_id=user.id, product_id="vex", name="Vacation", description="Summer trip"
        )
        fetched = await project_repo.get(project.id, user_id=user.id, product_id="vex")
        assert fetched is not None
        assert fetched.name == "Vacation"
        assert fetched.description == "Summer trip"

    async def test_get_scoped_to_owner_returns_none_for_other_user(
        self,
        project_repo: LibraryProjectRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
    ) -> None:
        owner = await make_user(email=f"powner-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"pother-{uuid4().hex[:8]}@example.com")
        project = await project_repo.create(
            user_id=owner.id, product_id="vex", name="Owner Project", description=None
        )
        assert await project_repo.get(project.id, user_id=other.id, product_id="vex") is None

    async def test_get_scoped_to_product_returns_none_for_other_product(
        self,
        project_repo: LibraryProjectRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
    ) -> None:
        user = await make_user(email=f"pprod-{uuid4().hex[:8]}@example.com", product_id="vex")
        project = await project_repo.create(
            user_id=user.id, product_id="vex", name="Vex Project", description=None
        )
        assert await project_repo.get(project.id, user_id=user.id, product_id="synthara") is None

    async def test_list_by_user_excludes_other_users_projects(
        self,
        project_repo: LibraryProjectRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
    ) -> None:
        user_a = await make_user(email=f"plista-{uuid4().hex[:8]}@example.com")
        user_b = await make_user(email=f"plistb-{uuid4().hex[:8]}@example.com")
        await project_repo.create(user_id=user_a.id, product_id="vex", name="A", description=None)
        await project_repo.create(user_id=user_b.id, product_id="vex", name="B", description=None)

        rows = await project_repo.list_by_user(user_a.id, "vex", limit=20)
        assert {r.name for r in rows} == {"A"}

    async def test_update_renames_and_redescribes(
        self,
        project_repo: LibraryProjectRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
    ) -> None:
        user = await make_user(email=f"pupdate-{uuid4().hex[:8]}@example.com")
        project = await project_repo.create(
            user_id=user.id, product_id="vex", name="Old Name", description="old"
        )
        updated = await project_repo.update(
            project.id, user_id=user.id, product_id="vex", name="New Name"
        )
        assert updated is not None
        assert updated.name == "New Name"
        assert updated.description == "old"  # unchanged (UNSET_UPDATE default)

    async def test_delete_removes_project(
        self,
        project_repo: LibraryProjectRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
    ) -> None:
        user = await make_user(email=f"pdelete-{uuid4().hex[:8]}@example.com")
        project = await project_repo.create(
            user_id=user.id, product_id="vex", name="Deleteme", description=None
        )
        assert await project_repo.delete(project.id, user_id=user.id, product_id="vex") is True
        assert await project_repo.get(project.id, user_id=user.id, product_id="vex") is None

    async def test_delete_nonexistent_returns_false(
        self,
        project_repo: LibraryProjectRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
    ) -> None:
        user = await make_user(email=f"pdelnone-{uuid4().hex[:8]}@example.com")
        assert await project_repo.delete(uuid4(), user_id=user.id, product_id="vex") is False


class TestProjectNameConflict:
    async def test_duplicate_name_same_case_raises_conflict(
        self,
        project_service: LibraryProjectService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"pconflict-{uuid4().hex[:8]}@example.com")
        await project_service.create(user.id, "vex", "My Trip", None, session=db_session)

        with pytest.raises(LibraryProjectNameConflictError):
            await project_service.create(user.id, "vex", "My Trip", None, session=db_session)

    async def test_duplicate_name_different_case_raises_conflict(
        self,
        project_service: LibraryProjectService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"pconflictci-{uuid4().hex[:8]}@example.com")
        await project_service.create(user.id, "vex", "My Trip", None, session=db_session)

        with pytest.raises(LibraryProjectNameConflictError):
            await project_service.create(user.id, "vex", "MY TRIP", None, session=db_session)

    async def test_same_name_different_owner_is_allowed(
        self,
        project_service: LibraryProjectService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user_a = await make_user(email=f"psamea-{uuid4().hex[:8]}@example.com")
        user_b = await make_user(email=f"psameb-{uuid4().hex[:8]}@example.com")
        await project_service.create(user_a.id, "vex", "Shared Name", None, session=db_session)
        # Must not raise — uniqueness is scoped per (product_id, user_id).
        await project_service.create(user_b.id, "vex", "Shared Name", None, session=db_session)

    async def test_rename_to_existing_name_raises_conflict(
        self,
        project_service: LibraryProjectService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        from src.api.schemas.library import LibraryProjectPatch

        user = await make_user(email=f"prename-{uuid4().hex[:8]}@example.com")
        await project_service.create(user.id, "vex", "First", None, session=db_session)
        second = await project_service.create(user.id, "vex", "Second", None, session=db_session)

        with pytest.raises(LibraryProjectNameConflictError):
            await project_service.patch(
                second.id,
                LibraryProjectPatch(name="first"),
                user.id,
                "vex",
                session=db_session,
            )

    async def test_conflict_does_not_poison_session_for_subsequent_writes(
        self,
        project_service: LibraryProjectService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        """The SAVEPOINT-isolated create must leave the outer session usable afterward."""
        user = await make_user(email=f"ppoison-{uuid4().hex[:8]}@example.com")
        await project_service.create(user.id, "vex", "Taken", None, session=db_session)

        with pytest.raises(LibraryProjectNameConflictError):
            await project_service.create(user.id, "vex", "Taken", None, session=db_session)

        # Session must still be usable — a distinct name succeeds.
        created = await project_service.create(
            user.id, "vex", "Not Taken", None, session=db_session
        )
        assert created.name == "Not Taken"


class TestProjectServiceListGetDelete:
    """LibraryProjectService.list_projects/get/delete — the thin service-layer
    wrappers around LibraryProjectRepository. Other tests in this module and
    in test_library_bulk.py exercise `create`/`patch` through the service,
    but list_projects/get/delete were previously only exercised via
    `project_repo` directly, leaving the service methods themselves uncovered.
    """

    async def test_get_found_returns_project(
        self,
        project_service: LibraryProjectService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"psvcget-{uuid4().hex[:8]}@example.com")
        created = await project_service.create(
            user.id, "vex", "Service Get", "desc", session=db_session
        )

        fetched = await project_service.get(created.id, user.id, "vex", session=db_session)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "Service Get"
        assert fetched.description == "desc"

    async def test_get_cross_user_returns_none(
        self,
        project_service: LibraryProjectService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        owner = await make_user(email=f"psvcowner-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"psvcother-{uuid4().hex[:8]}@example.com")
        created = await project_service.create(
            owner.id, "vex", "Owner Only", None, session=db_session
        )

        assert await project_service.get(created.id, other.id, "vex", session=db_session) is None

    async def test_list_projects_returns_page_with_asset_counts(
        self,
        project_service: LibraryProjectService,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"psvclist-{uuid4().hex[:8]}@example.com")
        first = await project_service.create(user.id, "vex", "Alpha", None, session=db_session)
        await project_service.create(user.id, "vex", "Beta", None, session=db_session)

        image = await make_user_image(user=user)
        await library_repo.upsert_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, image.id, project_id=first.id
        )

        page = await project_service.list_projects(user.id, "vex", session=db_session, limit=30)
        assert page.has_more is False
        assert page.next_cursor is None
        names = {item.name: item.asset_count for item in page.items}
        assert names == {"Alpha": 1, "Beta": 0}

    async def test_list_projects_paginates_with_cursor(
        self,
        project_service: LibraryProjectService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"psvcpage-{uuid4().hex[:8]}@example.com")
        await project_service.create(user.id, "vex", "One", None, session=db_session)
        await project_service.create(user.id, "vex", "Two", None, session=db_session)
        await project_service.create(user.id, "vex", "Three", None, session=db_session)

        first_page = await project_service.list_projects(
            user.id, "vex", session=db_session, limit=2
        )
        assert len(first_page.items) == 2
        assert first_page.has_more is True
        assert first_page.next_cursor is not None

        second_page = await project_service.list_projects(
            user.id, "vex", session=db_session, limit=2, cursor=first_page.next_cursor
        )
        assert len(second_page.items) == 1
        assert second_page.has_more is False

        seen = {item.name for item in first_page.items} | {item.name for item in second_page.items}
        assert seen == {"One", "Two", "Three"}

    async def test_delete_found_returns_true(
        self,
        project_service: LibraryProjectService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"psvcdel-{uuid4().hex[:8]}@example.com")
        created = await project_service.create(
            user.id, "vex", "Deletable", None, session=db_session
        )

        assert await project_service.delete(created.id, user.id, "vex", session=db_session) is True
        assert await project_service.get(created.id, user.id, "vex", session=db_session) is None

    async def test_delete_not_found_returns_false(
        self,
        project_service: LibraryProjectService,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"psvcdelnf-{uuid4().hex[:8]}@example.com")
        assert await project_service.delete(uuid4(), user.id, "vex", session=db_session) is False


class TestProjectDeleteSetsNull:
    async def test_delete_project_unassigns_asset_not_delete_it(
        self,
        project_repo: LibraryProjectRepository,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        db_session: AsyncSession,
    ) -> None:
        user = await make_user(email=f"psetnull-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=user)
        project = await project_repo.create(
            user_id=user.id, product_id="vex", name="ToDelete", description=None
        )
        metadata = await library_repo.upsert_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, image.id, project_id=project.id
        )
        assert metadata.project_id == project.id

        assert await project_repo.delete(project.id, user_id=user.id, product_id="vex") is True

        # ON DELETE SET NULL fires at the DB level — the identity map may
        # hold a stale project_id, so re-fetch fresh from the DB.
        await db_session.refresh(metadata)
        result = await db_session.execute(
            select(LibraryAssetMetadata).where(LibraryAssetMetadata.id == metadata.id)
        )
        row = result.scalar_one()
        assert row.project_id is None

        # The asset itself must survive the project deletion untouched.
        image_repo_result = await db_session.get(type(image), image.id)
        assert image_repo_result is not None


class TestListAssetsProjectFilter:
    async def test_project_filter_matches_upload_branch(
        self,
        library_repo: LibraryRepository,
        project_repo: LibraryProjectRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        user = await make_user(email=f"projfiltup-{uuid4().hex[:8]}@example.com")
        project = await project_repo.create(
            user_id=user.id, product_id="vex", name="Filter Upload", description=None
        )
        in_project = await make_user_image(user=user)
        not_in_project = await make_user_image(user=user)
        await library_repo.upsert_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, in_project.id, project_id=project.id
        )

        rows = await library_repo.list_assets(user.id, "vex", limit=20, project_id=project.id)
        ids = {r.id for r in rows}
        assert in_project.id in ids
        assert not_in_project.id not in ids

    async def test_project_filter_matches_output_branch(
        self,
        library_repo: LibraryRepository,
        project_repo: LibraryProjectRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_output: OutputFactory,
    ) -> None:
        user = await make_user(email=f"projfiltout-{uuid4().hex[:8]}@example.com")
        project = await project_repo.create(
            user_id=user.id, product_id="vex", name="Filter Output", description=None
        )
        in_project = await make_output(user=user)
        not_in_project = await make_output(user=user)
        await library_repo.upsert_metadata(
            user.id, "vex", LibraryAssetSource.OUTPUT, in_project.id, project_id=project.id
        )

        rows = await library_repo.list_assets(user.id, "vex", limit=20, project_id=project.id)
        ids = {r.id for r in rows}
        assert in_project.id in ids
        assert not_in_project.id not in ids


class TestListAssetsExpiringFilter:
    async def test_expiring_true_matches_soon_to_expire(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        user = await make_user(email=f"expiring-{uuid4().hex[:8]}@example.com")
        soon = await make_user_image(user=user, expires_at=datetime.now(UTC) + timedelta(days=1))
        later = await make_user_image(user=user, expires_at=datetime.now(UTC) + timedelta(days=30))

        rows = await library_repo.list_assets(user.id, "vex", limit=20, expiring=True)
        ids = {r.id for r in rows}
        assert soon.id in ids
        assert later.id not in ids

    async def test_expiring_false_matches_not_soon(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        user = await make_user(email=f"notexpiring-{uuid4().hex[:8]}@example.com")
        soon = await make_user_image(user=user, expires_at=datetime.now(UTC) + timedelta(days=1))
        later = await make_user_image(user=user, expires_at=datetime.now(UTC) + timedelta(days=30))

        rows = await library_repo.list_assets(user.id, "vex", limit=20, expiring=False)
        ids = {r.id for r in rows}
        assert later.id in ids
        assert soon.id not in ids


class TestListAssetsSearchQuery:
    async def test_search_matches_display_title(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        user = await make_user(email=f"searchtitle-{uuid4().hex[:8]}@example.com")
        match = await make_user_image(user=user)
        no_match = await make_user_image(user=user)
        await library_repo.upsert_metadata(
            user.id, "vex", LibraryAssetSource.UPLOAD, match.id, display_title="Sunset Beach"
        )

        rows = await library_repo.list_assets(user.id, "vex", limit=20, query="sunset")
        ids = {r.id for r in rows}
        assert match.id in ids
        assert no_match.id not in ids

    async def test_search_matches_original_filename(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        user = await make_user(email=f"searchfile-{uuid4().hex[:8]}@example.com")
        match = await make_user_image(user=user, original_filename="vacation_photo.png")
        no_match = await make_user_image(user=user, original_filename="other.png")

        rows = await library_repo.list_assets(user.id, "vex", limit=20, query="vacation")
        ids = {r.id for r in rows}
        assert match.id in ids
        assert no_match.id not in ids

    async def test_search_matches_job_prompt_on_output_branch(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
        make_output: OutputFactory,
    ) -> None:
        user = await make_user(email=f"searchprompt-{uuid4().hex[:8]}@example.com")
        job = await make_job(user=user, status="completed", prompt="a dragon flying over mountains")
        match = await make_output(user=user, job=job)
        other_job = await make_job(user=user, status="completed", prompt="a quiet forest")
        no_match = await make_output(user=user, job=other_job)

        rows = await library_repo.list_assets(user.id, "vex", limit=20, query="dragon")
        ids = {r.id for r in rows}
        assert match.id in ids
        assert no_match.id not in ids

    async def test_search_does_not_match_prompt_on_upload_branch(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_job: Callable[..., Coroutine[Any, Any, GenerationJob]],
        make_output: OutputFactory,
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Unfiltered search must not accidentally drag uploads out via prompt matching."""
        user = await make_user(email=f"searchupnoprompt-{uuid4().hex[:8]}@example.com")
        job = await make_job(user=user, status="completed", prompt="a dragon flying")
        await make_output(user=user, job=job)
        upload = await make_user_image(user=user, original_filename="unrelated.png")

        rows = await library_repo.list_assets(user.id, "vex", limit=20, query="dragon")
        ids = {r.id for r in rows}
        assert upload.id not in ids

    async def test_search_respects_user_scoping(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        owner = await make_user(email=f"searchowner-{uuid4().hex[:8]}@example.com")
        other = await make_user(email=f"searchother-{uuid4().hex[:8]}@example.com")
        image = await make_user_image(user=owner, original_filename="secret_document.png")
        await library_repo.upsert_metadata(
            owner.id, "vex", LibraryAssetSource.UPLOAD, image.id, display_title="secret"
        )

        rows = await library_repo.list_assets(other.id, "vex", limit=20, query="secret")
        assert len(rows) == 0

    async def test_search_escapes_like_metacharacters(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Searching for a literal '%' must not act as a SQL LIKE wildcard.

        Without escaping, ``query="100%"`` wrapped as ``%100%%`` collapses to
        ``%100%`` (redundant wildcards) and would match ANY filename merely
        containing "100" — including one with no literal percent character
        at all. With escaping it must match only the filename that actually
        contains a literal "100%" substring.
        """
        user = await make_user(email=f"searchesc-{uuid4().hex[:8]}@example.com")
        literal_percent = await make_user_image(user=user, original_filename="100%done.png")
        no_literal_percent = await make_user_image(user=user, original_filename="100done.png")
        unrelated = await make_user_image(user=user, original_filename="somethingelsedone.png")

        rows = await library_repo.list_assets(user.id, "vex", limit=20, query="100%")
        ids = {r.id for r in rows}
        assert literal_percent.id in ids
        assert no_literal_percent.id not in ids
        assert unrelated.id not in ids


class TestExpiringSoonSortPagination:
    async def test_pagination_stable_no_dupes_no_gaps(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
        make_output: OutputFactory,
        db_session: AsyncSession,
    ) -> None:
        from src.api.schemas.pagination import decode_library_cursor, encode_library_cursor

        user = await make_user(email=f"expsort-{uuid4().hex[:8]}@example.com")
        base = datetime.now(UTC) + timedelta(hours=1)

        all_ids: set[Any] = set()
        for i in range(3):
            expires = base + timedelta(seconds=i)
            upload = await make_user_image(user=user, expires_at=expires)
            output = await make_output(user=user, expires_at=expires)
            all_ids.add(upload.id)
            all_ids.add(output.id)
        await db_session.flush()

        seen: list[Any] = []
        cursor: str | None = None
        limit = 2
        for _ in range(10):
            decoded = (
                decode_library_cursor(cursor, expected_sort="expiring_soon") if cursor else None
            )
            rows = await library_repo.list_assets(
                user.id, "vex", limit=limit, cursor=decoded, sort=LibrarySort.EXPIRING_SOON
            )
            has_more = len(rows) > limit
            page = rows[:limit]
            seen.extend(r.id for r in page)
            if not has_more or not page:
                break
            last = page[-1]
            cursor = encode_library_cursor(
                last.expires_at, last.source.value, last.id, sort="expiring_soon"
            )

        assert len(seen) == len(set(seen)), "pagination produced duplicate rows"
        assert set(seen) == all_ids, "pagination missed or invented rows"

    async def test_orders_ascending_by_expires_at(
        self,
        library_repo: LibraryRepository,
        make_user: Callable[..., Coroutine[Any, Any, User]],
        make_user_image: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        user = await make_user(email=f"expasc-{uuid4().hex[:8]}@example.com")
        now = datetime.now(UTC)
        soonest = await make_user_image(user=user, expires_at=now + timedelta(hours=1))
        latest = await make_user_image(user=user, expires_at=now + timedelta(days=6))

        rows = await library_repo.list_assets(
            user.id, "vex", limit=20, sort=LibrarySort.EXPIRING_SOON
        )
        ids_in_order = [r.id for r in rows]
        assert ids_in_order.index(soonest.id) < ids_in_order.index(latest.id)
