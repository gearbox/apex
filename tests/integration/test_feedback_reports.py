"""In-product problem reports against real PostgreSQL 16.

Drives the real ``FeedbackController`` / ``AdminFeedbackController`` +
``FeedbackService`` + ``FeedbackReportRepository`` over the per-test savepoint
session (the ops bus is a mock). Concurrency uses committed rows and two
independent sessions.

Contracts: C6 (ownership 404s leak nothing), C7 (short/NUL message → 400,
never 500), C9 (CHECK constraints), C11 (terminal-once under concurrency),
C12 (product scoping + admin-only), C13 (keyset pagination + filters), C17
(report survives a user hard-delete), C18 (migration 049 round-trip).
"""

from __future__ import annotations

import asyncio
import importlib.util
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import httpx
import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from litestar import Litestar
from litestar.datastructures import State
from litestar.di import Provide
from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.app import legal_acceptance_required_handler
from src.api.dependencies.common import get_product_config, get_product_id
from src.api.middleware.product import ProductMiddleware
from src.api.routes.admin_feedback import AdminFeedbackController
from src.api.routes.feedback import FeedbackController
from src.api.schemas.feedback import FeedbackAdminPatch
from src.api.schemas.ops_events import FeedbackSubmittedOpsPayload, OpsEventType
from src.api.security import JWTConfig, JWTService
from src.api.services.feedback import FeedbackService, InvalidFeedbackTransitionError
from src.api.services.legal.errors import LegalAcceptanceRequiredError
from src.api.services.token_revocation import TokenRevocationService
from src.core.enums import FeedbackCategory, FeedbackStatus, UserRole
from src.core.product_registry import SYNTHARA_CONFIG, VEX_CONFIG
from src.core.uid import new_id
from src.db.models.feedback import FeedbackReport
from src.db.models.user import User
from src.db.repositories.output import OutputRepository
from tests.integration.pg_locks import wait_for_row_lock_waiter
from tests.legal_support import make_legal_registry

if TYPE_CHECKING:
    import contextlib
    from collections.abc import Sequence
    from types import ModuleType

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.db.models.storage import GenerationOutput
    from tests.integration.conftest import JobFactory, UserFactory, UserImageFactory

JWT_SECRET = "integration-feedback-secret-key-32-bytes"
MESSAGE = "The generate button spins forever"


def _today() -> date:
    return datetime.now(UTC).date()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class Api:
    """Real feedback controllers over the test session; ops bus is a mock."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.jwt = JWTService(JWTConfig(secret_key=JWT_SECRET))
        self.registry = make_legal_registry()
        self.ops = MagicMock()
        self.ops.publish = AsyncMock()
        self.service = FeedbackService(session=session, ops_event_bus=self.ops)

    def client(self) -> contextlib.AbstractAsyncContextManager[httpx.AsyncClient]:
        """In-loop ASGI client (the app shares the test's asyncpg connection)."""
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=cast("Any", self._app())),
            base_url="http://testserver.local",
        )

    def _app(self) -> Litestar:
        return Litestar(
            route_handlers=[FeedbackController, AdminFeedbackController],
            middleware=[ProductMiddleware],
            dependencies={
                "product_config": Provide(get_product_config, sync_to_thread=False),
                "product_id": Provide(get_product_id, sync_to_thread=False),
                "session": Provide(lambda: self.session, sync_to_thread=False),
                "feedback_service": Provide(lambda: self.service, sync_to_thread=False),
            },
            exception_handlers={LegalAcceptanceRequiredError: legal_acceptance_required_handler},
            state=State(
                {
                    "jwt_service": self.jwt,
                    "token_revocation": TokenRevocationService(None, max_token_ttl_seconds=0),
                    "legal_registry": self.registry,
                }
            ),
        )

    def headers(self, user: User, *, product: str | None = None) -> dict[str, str]:
        """Bearer + product headers with a current legal digest for ``user``'s product."""
        product = product or user.product_id
        config = VEX_CONFIG if user.product_id == "vex" else SYNTHARA_CONFIG
        token, _ = self.jwt.create_access_token(
            user.id,
            product_id=user.product_id,
            legal_digest=self.registry.required_digest(config, today=_today()),
        )
        return {"Authorization": f"Bearer {token}", "X-Product-Id": product}


async def _user(make_user: UserFactory, *, product_id: str = "vex", admin: bool = False) -> User:
    user = await make_user(email=f"fb-{uuid4().hex[:10]}@example.com", product_id=product_id)
    if admin:
        user.role = UserRole.ADMIN
    return user


async def _output(session: AsyncSession, job_user: User, job_id: UUID) -> GenerationOutput:
    output_id = uuid4()
    return await OutputRepository(session).create(
        id=output_id,
        user_id=job_user.id,
        job_id=job_id,
        storage_key=f"users/{job_user.id}/outputs/{job_id}/{output_id}.png",
        content_type="image/png",
        size_bytes=1,
        format="png",
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id=job_user.product_id,
    )


def _report(user: User | None, **overrides: Any) -> FeedbackReport:
    fields: dict[str, Any] = {
        "id": new_id(),
        "product_id": user.product_id if user else "vex",
        "user_id": user.id if user else None,
        "category": FeedbackCategory.BUG.value,
        "status": FeedbackStatus.OPEN.value,
        "message": MESSAGE,
    }
    fields.update(overrides)
    return FeedbackReport(**fields)


async def _insert(session: AsyncSession, *reports: FeedbackReport) -> None:
    session.add_all(reports)
    await session.flush()


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


class TestSubmit:
    async def test_created_row_and_ids_only_ping(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        make_job: JobFactory,
        make_user_image: UserImageFactory,
    ) -> None:
        api = Api(db_session)
        user = await _user(make_user)
        job = await make_job(user=user)
        image = await make_user_image(user=user)

        async with api.client() as client:
            resp = await client.post(
                "/v1/feedback",
                json={
                    "category": "generation",
                    "message": f"   {MESSAGE}   ",
                    "job_id": str(job.id),
                    "asset_ref": f"upload:{image.id}",
                    "client_path": "/library",
                    "app_version": "2026.9.1",
                },
                headers={**api.headers(user), "User-Agent": "IntegrationUA/1.0"},
            )

        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert set(body) == {"id", "status", "created_at"}
        assert body["status"] == "open"
        assert body["created_at"] is not None

        row = (
            await db_session.execute(
                select(FeedbackReport).where(FeedbackReport.id == UUID(body["id"]))
            )
        ).scalar_one()
        assert row.product_id == "vex"
        assert row.user_id == user.id
        assert row.category == "generation"
        assert row.message == MESSAGE
        assert row.job_id == job.id
        assert (row.asset_source, row.asset_id) == ("upload", image.id)
        assert row.client_path == "/library"
        assert row.app_version == "2026.9.1"
        assert row.user_agent == "IntegrationUA/1.0"

        api.ops.publish.assert_awaited_once_with(
            event_type=OpsEventType.FEEDBACK_SUBMITTED,
            product_id="vex",
            payload=FeedbackSubmittedOpsPayload(
                report_id=row.id, user_id=user.id, category="generation", job_id=job.id
            ),
        )

    async def test_foreign_context_is_indistinguishable_from_missing(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        make_job: JobFactory,
        make_user_image: UserImageFactory,
    ) -> None:
        """C6 — another user's job/asset gets the same 404 body as a nonexistent id."""
        api = Api(db_session)
        owner = await _user(make_user)
        caller = await _user(make_user)
        job = await make_job(user=owner)
        image = await make_user_image(user=owner)
        output = await _output(db_session, owner, job.id)
        deleted_own_job = await make_job(user=caller, is_deleted=True)

        cases: Sequence[tuple[dict[str, str], dict[str, str]]] = [
            ({"job_id": str(job.id)}, {"job_id": str(uuid4())}),
            ({"job_id": str(deleted_own_job.id)}, {"job_id": str(uuid4())}),
            ({"asset_ref": f"upload:{image.id}"}, {"asset_ref": f"upload:{uuid4()}"}),
            ({"asset_ref": f"output:{output.id}"}, {"asset_ref": f"output:{uuid4()}"}),
        ]
        async with api.client() as client:
            for foreign, missing in cases:
                responses = [
                    await client.post(
                        "/v1/feedback",
                        json={"category": "bug", "message": MESSAGE, **context},
                        headers=api.headers(caller),
                    )
                    for context in (foreign, missing)
                ]
                assert [r.status_code for r in responses] == [404, 404], foreign
                assert responses[0].json() == responses[1].json()
                assert responses[0].json()["error"] in {"job_not_found", "asset_not_found"}

        count = (
            await db_session.execute(
                select(FeedbackReport).where(FeedbackReport.user_id == caller.id)
            )
        ).all()
        assert count == []
        api.ops.publish.assert_not_awaited()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"message": "      short                  "},
            {"message": "long enough but has a \u0000 NUL"},
            {"app_version": "1.0\u0000"},
            {"client_path": "/lib\u0000rary"},
            {"asset_ref": "output:\u0000"},
            {"asset_ref": "not-a-ref"},
        ],
    )
    async def test_bad_input_is_400_not_500(
        self, db_session: AsyncSession, make_user: UserFactory, overrides: dict[str, str]
    ) -> None:
        """C7 — asyncpg rejects NUL in text; it must never get that far."""
        api = Api(db_session)
        user = await _user(make_user)
        async with api.client() as client:
            resp = await client.post(
                "/v1/feedback",
                json={"category": "bug", "message": MESSAGE, **overrides},
                headers=api.headers(user),
            )
        assert resp.status_code == 400, resp.text
        api.ops.publish.assert_not_awaited()

    async def test_stale_legal_digest_still_submits(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """C14 — the in-product reporting function works before re-acceptance."""
        api = Api(db_session)
        user = await _user(make_user)
        token, _ = api.jwt.create_access_token(user.id, product_id="vex", legal_digest="stale")
        async with api.client() as client:
            resp = await client.post(
                "/v1/feedback",
                json={"category": "account", "message": MESSAGE},
                headers={"Authorization": f"Bearer {token}", "X-Product-Id": "vex"},
            )
            patch_resp = await client.patch(
                f"/v1/admin/feedback/{resp.json()['id']}",
                json={"status": "resolved"},
                headers={"Authorization": f"Bearer {token}", "X-Product-Id": "vex"},
            )
        assert resp.status_code == 201
        assert patch_resp.status_code == 428


# ---------------------------------------------------------------------------
# Constraints (C9) and user deletion (C17)
# ---------------------------------------------------------------------------


class TestConstraints:
    @pytest.mark.parametrize(
        ("overrides", "constraint"),
        [
            ({"asset_source": "output"}, "chk_feedback_asset_pair"),
            ({"asset_id": uuid4()}, "chk_feedback_asset_pair"),
            ({"status": "resolved"}, "chk_feedback_resolved_at_terminal"),
            ({"status": "dismissed"}, "chk_feedback_resolved_at_terminal"),
            ({"resolved_at": datetime.now(UTC)}, "chk_feedback_resolved_at_terminal"),
            (
                {"status": "in_progress", "resolved_at": datetime.now(UTC)},
                "chk_feedback_resolved_at_terminal",
            ),
        ],
    )
    async def test_check_constraints_enforced(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        overrides: dict[str, Any],
        constraint: str,
    ) -> None:
        user = await _user(make_user)
        with pytest.raises(IntegrityError, match=constraint):
            async with db_session.begin_nested():
                await _insert(db_session, _report(user, **overrides))

    async def test_consistent_rows_accepted(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        user = await _user(make_user)
        await _insert(
            db_session,
            _report(user, asset_source="output", asset_id=uuid4()),
            _report(user, status="resolved", resolved_at=datetime.now(UTC)),
            _report(user, status="in_progress"),
        )

    async def test_report_survives_user_hard_delete(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """C17 — reporter and resolver purges null the references, keep the report."""
        reporter = await _user(make_user)
        resolver = await _user(make_user, admin=True)
        report = _report(
            reporter, status="resolved", resolved_at=datetime.now(UTC), resolved_by=resolver.id
        )
        await _insert(db_session, report)

        await db_session.execute(delete(User).where(User.id.in_([reporter.id, resolver.id])))
        await db_session.refresh(report)

        assert report.user_id is None
        assert report.resolved_by is None
        assert report.message == MESSAGE
        view = await FeedbackService(session=db_session, ops_event_bus=MagicMock()).get_for_admin(
            report.id, product_id="vex"
        )
        assert view.user_id is None
        assert view.user_email is None


# ---------------------------------------------------------------------------
# Admin API
# ---------------------------------------------------------------------------


class TestAdminTriage:
    async def test_lifecycle_writes_resolution_once(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        api = Api(db_session)
        admin = await _user(make_user, admin=True)
        report = _report(await _user(make_user))
        await _insert(db_session, report)
        url = f"/v1/admin/feedback/{report.id}"

        async with api.client() as client:
            started = await client.patch(
                url, json={"status": "in_progress"}, headers=api.headers(admin)
            )
            resolved = await client.patch(
                url, json={"status": "resolved", "admin_note": "fixed"}, headers=api.headers(admin)
            )
            again = await client.patch(url, json={"status": "resolved"}, headers=api.headers(admin))
            dismissed = await client.patch(
                url, json={"status": "dismissed"}, headers=api.headers(admin)
            )
            noted = await client.patch(url, json={"admin_note": None}, headers=api.headers(admin))
            empty = await client.patch(url, json={}, headers=api.headers(admin))

        assert started.status_code == 200, started.text
        assert started.json()["resolved_at"] is None
        assert resolved.status_code == 200
        first = resolved.json()
        assert first["status"] == "resolved"
        assert first["admin_note"] == "fixed"
        assert first["resolved_by"] == str(admin.id)
        assert first["resolved_at"] is not None
        assert again.status_code == 409
        assert again.json()["error"] == "invalid_status_transition"
        assert dismissed.status_code == 409
        assert noted.status_code == 200
        assert noted.json()["admin_note"] is None
        assert noted.json()["resolved_at"] == first["resolved_at"]
        assert noted.json()["resolved_by"] == str(admin.id)
        assert empty.status_code == 400

    async def test_product_scoping_and_admin_only(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """C12 — a vex admin can't see synthara reports; non-admins are refused."""
        api = Api(db_session)
        vex_admin = await _user(make_user, admin=True)
        vex_user = await _user(make_user)
        syn_report = _report(await _user(make_user, product_id="synthara"))
        vex_report = _report(vex_user)
        await _insert(db_session, syn_report, vex_report)

        async with api.client() as client:
            listed = await client.get("/v1/admin/feedback", headers=api.headers(vex_admin))
            got = await client.get(
                f"/v1/admin/feedback/{syn_report.id}", headers=api.headers(vex_admin)
            )
            patched = await client.patch(
                f"/v1/admin/feedback/{syn_report.id}",
                json={"status": "resolved"},
                headers=api.headers(vex_admin),
            )
            cross = await client.get(
                "/v1/admin/feedback", headers=api.headers(vex_admin, product="synthara")
            )
            non_admin = await client.get("/v1/admin/feedback", headers=api.headers(vex_user))
            non_admin_patch = await client.patch(
                f"/v1/admin/feedback/{vex_report.id}",
                json={"status": "resolved"},
                headers=api.headers(vex_user),
            )

        ids = {item["id"] for item in listed.json()["items"]}
        assert str(vex_report.id) in ids
        assert str(syn_report.id) not in ids
        assert got.status_code == 404
        assert got.json()["error"] == "feedback_not_found"
        assert patched.status_code == 404
        # A vex token presented under the synthara product is rejected by auth_guard.
        assert cross.status_code == 401
        # Admin-only: the shared admin dependency refuses non-admins with 401.
        assert non_admin.status_code == 401
        assert non_admin_patch.status_code == 401
        await db_session.refresh(syn_report)
        await db_session.refresh(vex_report)
        assert syn_report.status == vex_report.status == "open"

    async def test_keyset_pagination_and_filters(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """C13 — stable across concurrent inserts; status/category filters apply."""
        api = Api(db_session)
        admin = await _user(make_user, admin=True)
        reporter = await _user(make_user)
        base = datetime.now(UTC) - timedelta(hours=1)
        reports = [
            _report(
                reporter,
                created_at=base + timedelta(minutes=i),
                category=(FeedbackCategory.BILLING if i % 2 else FeedbackCategory.BUG).value,
                status=(FeedbackStatus.IN_PROGRESS if i == 4 else FeedbackStatus.OPEN).value,
            )
            for i in range(5)
        ]
        # Same created_at as reports[2] — the id tiebreak must keep the order total.
        tie = _report(reporter, created_at=reports[2].created_at)
        await _insert(db_session, *reports, tie)
        newest_first = sorted([*reports, tie], key=lambda r: (r.created_at, r.id), reverse=True)

        seen: list[str] = []
        async with api.client() as client:
            cursor: str | None = None
            while True:
                params: dict[str, Any] = {"limit": 2}
                if cursor:
                    params["cursor"] = cursor
                page = (
                    await client.get(
                        "/v1/admin/feedback", params=params, headers=api.headers(admin)
                    )
                ).json()
                seen.extend(item["id"] for item in page["items"])
                if not seen[2:]:
                    # A report arriving mid-pagination must not shift later pages.
                    await _insert(db_session, _report(reporter))
                if not page["has_more"]:
                    assert page["next_cursor"] is None
                    break
                cursor = page["next_cursor"]

            billing = (
                await client.get(
                    "/v1/admin/feedback",
                    params={"category": "billing", "limit": 100},
                    headers=api.headers(admin),
                )
            ).json()
            in_progress = (
                await client.get(
                    "/v1/admin/feedback",
                    params={"status": "in_progress", "category": "bug"},
                    headers=api.headers(admin),
                )
            ).json()
            bad_cursor = await client.get(
                "/v1/admin/feedback", params={"cursor": "garbage"}, headers=api.headers(admin)
            )
            bad_filter = await client.get(
                "/v1/admin/feedback", params={"status": "closed"}, headers=api.headers(admin)
            )

        ours = {str(r.id) for r in newest_first}
        assert [i for i in seen if i in ours] == [str(r.id) for r in newest_first]
        assert len(seen) == len(set(seen))
        assert {i["id"] for i in billing["items"]} == {
            str(r.id) for r in reports if r.category == "billing"
        }
        assert [i["id"] for i in in_progress["items"]] == [str(reports[4].id)]
        assert bad_cursor.status_code == 400
        assert bad_cursor.json()["error"] == "invalid_cursor"
        assert bad_filter.status_code == 400


# ---------------------------------------------------------------------------
# C11 — terminal-once under concurrency (committed rows, two sessions)
# ---------------------------------------------------------------------------


async def _commit(engine: AsyncEngine, *rows: Any) -> None:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        for row in rows:
            session.add(row)
            await session.flush()
        await session.commit()


async def _purge(engine: AsyncEngine, report_id: UUID, user_ids: Sequence[UUID]) -> None:
    async with AsyncSession(bind=engine) as session:
        await session.execute(delete(FeedbackReport).where(FeedbackReport.id == report_id))
        await session.execute(delete(User).where(User.id.in_(user_ids)))
        await session.commit()


class TestConcurrentTerminalTransition:
    async def test_exactly_one_terminal_transition_wins(self, db_engine: AsyncEngine) -> None:
        reporter = User(
            id=new_id(), email=f"fb-rep-{new_id()}@example.com", password_hash="h", product_id="vex"
        )
        admin_a = User(
            id=new_id(),
            email=f"fb-a-{new_id()}@example.com",
            password_hash="h",
            product_id="vex",
            role=UserRole.ADMIN,
        )
        admin_b = User(
            id=new_id(),
            email=f"fb-b-{new_id()}@example.com",
            password_hash="h",
            product_id="vex",
            role=UserRole.ADMIN,
        )
        report = _report(reporter)
        await _commit(db_engine, reporter, admin_a, admin_b, report)

        try:
            async with (
                AsyncSession(bind=db_engine, expire_on_commit=False) as session_a,
                AsyncSession(bind=db_engine, expire_on_commit=False) as session_b,
            ):
                service_a = FeedbackService(session=session_a, ops_event_bus=MagicMock())
                service_b = FeedbackService(session=session_b, ops_event_bus=MagicMock())

                await service_a.update_by_admin(
                    report.id,
                    product_id="vex",
                    admin_id=admin_a.id,
                    patch=FeedbackAdminPatch(status=FeedbackStatus.RESOLVED),
                )
                await session_a.flush()  # A holds the row lock with the resolved state

                async def _dismiss() -> None:
                    await service_b.update_by_admin(
                        report.id,
                        product_id="vex",
                        admin_id=admin_b.id,
                        patch=FeedbackAdminPatch(status=FeedbackStatus.DISMISSED),
                    )
                    await session_b.commit()

                loser = asyncio.create_task(_dismiss())
                await wait_for_row_lock_waiter(db_engine, relation="feedback_reports")
                await session_a.commit()

                with pytest.raises(InvalidFeedbackTransitionError) as exc_info:
                    await loser
                assert exc_info.value.current is FeedbackStatus.RESOLVED
                await session_b.rollback()

            async with AsyncSession(bind=db_engine) as reader:
                row = (
                    await reader.execute(
                        select(
                            FeedbackReport.status,
                            FeedbackReport.resolved_by,
                            FeedbackReport.resolved_at,
                        ).where(FeedbackReport.id == report.id)
                    )
                ).one()
            assert row.status == "resolved"
            assert row.resolved_by == admin_a.id
            assert row.resolved_at is not None
        finally:
            await _purge(db_engine, report.id, [reporter.id, admin_a.id, admin_b.id])


# ---------------------------------------------------------------------------
# C18 — migration 049
# ---------------------------------------------------------------------------

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "049_feedback_reports.py"
)


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_049", _MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMigration049:
    async def test_round_trip(self, db_engine: AsyncEngine) -> None:
        migration = _load_migration()
        assert migration.down_revision == "048"
        async with db_engine.connect() as connection:
            transaction = await connection.begin()
            try:

                def _downgrade(sync_connection: Connection) -> None:
                    context = MigrationContext.configure(sync_connection)
                    with Operations.context(context):
                        migration.downgrade()

                def _upgrade(sync_connection: Connection) -> None:
                    context = MigrationContext.configure(sync_connection)
                    with Operations.context(context):
                        migration.upgrade()

                await connection.run_sync(_downgrade)
                gone = await connection.execute(text("SELECT to_regclass('feedback_reports')"))
                assert gone.scalar_one() is None

                await connection.run_sync(_upgrade)
                checks = await connection.execute(
                    text(
                        "SELECT conname FROM pg_constraint "
                        "WHERE conrelid = 'feedback_reports'::regclass AND contype = 'c'"
                    )
                )
                assert set(checks.scalars().all()) == {
                    "chk_feedback_asset_pair",
                    "chk_feedback_resolved_at_terminal",
                }
                fks = await connection.execute(
                    text(
                        "SELECT a.attname, c.confdeltype::text FROM pg_constraint c "
                        "JOIN pg_attribute a ON a.attrelid = c.conrelid "
                        "AND a.attnum = ANY(c.conkey) "
                        "WHERE c.conrelid = 'feedback_reports'::regclass AND c.contype = 'f'"
                    )
                )
                # 'n' = ON DELETE SET NULL
                assert dict(fks.tuples().all()) == {
                    "user_id": "n",
                    "job_id": "n",
                    "resolved_by": "n",
                }
                index = await connection.execute(
                    text(
                        "SELECT indexdef FROM pg_indexes "
                        "WHERE indexname = 'ix_feedback_reports_product_created'"
                    )
                )
                assert "(product_id, created_at, id)" in index.scalar_one()
            finally:
                await transaction.rollback()
