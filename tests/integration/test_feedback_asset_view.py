"""Admin view of the asset a feedback report references, against real PostgreSQL 16.

Drives the real ``ContentProxyController`` (``content_auth_guard``, the
admin dependency, ``_stream_from_r2``) + real ``FeedbackService`` /
``ContentProxyService`` / repositories over the per-test savepoint session.
Only R2 is faked (in-memory ``stream_object``).

Contracts: A1 (cookie-authenticated, under ``/v1/content``), A2 (resolution is
scoped to the reporter), A3 (``no-store``), A4 (audit on view start only),
A5 (``asset_url``), A6 (error codes), T8 (owner routes unchanged).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from litestar import Litestar
from litestar.datastructures import State
from litestar.di import Provide
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError

from src.api.dependencies.common import get_product_config, get_product_id
from src.api.middleware.product import ProductMiddleware
from src.api.routes.admin_feedback import AdminFeedbackController
from src.api.routes.content import ContentProxyController
from src.api.security import JWTConfig, JWTService
from src.api.services.content_proxy import ContentProxyService
from src.api.services.feedback import FeedbackService
from src.api.services.media import FEEDBACK_ASSET_PATH
from src.api.services.storage.exceptions import StorageError, StorageRangeNotSatisfiableError
from src.api.services.token_revocation import TokenRevocationService
from src.core.config import get_settings
from src.core.enums import FeedbackCategory, FeedbackStatus, UserRole
from src.core.uid import new_id
from src.db.models.admin import AdminAuditLog
from src.db.models.feedback import FeedbackReport
from src.db.models.user import User
from src.db.repositories.output import OutputRepository
from tests.integration.test_content_range_streaming import PAYLOAD, _StubR2
from tests.legal_support import make_legal_registry

if TYPE_CHECKING:
    import contextlib
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.storage.r2 import ObjectStream
    from src.db.models.storage import GenerationOutput
    from tests.integration.conftest import JobFactory, UserFactory, UserImageFactory

JWT_SECRET = "integration-feedback-asset-secret-key-32b"
MESSAGE = "SECRET-MESSAGE the render came out black"
CLIENT_PATH = "/secret-client-path/library"
NO_STORE = "private, no-store"
SIZE = len(PAYLOAD)


class Api:
    """Real content + admin-feedback controllers; R2 is an in-memory stub."""

    def __init__(self, session: AsyncSession, *, r2: _StubR2 | None = None) -> None:
        self.session = session
        self.r2 = r2 or _StubR2()
        self.jwt = JWTService(JWTConfig(secret_key=JWT_SECRET))
        self.registry = make_legal_registry()
        ops = MagicMock()
        ops.publish = AsyncMock()
        self.service = FeedbackService(session=session, ops_event_bus=ops)
        self.proxy = ContentProxyService(storage=MagicMock(), settings=get_settings())

    def client(self) -> contextlib.AbstractAsyncContextManager[httpx.AsyncClient]:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=cast("Any", self._app())),
            base_url="http://testserver.local",
        )

    def _app(self) -> Litestar:
        return Litestar(
            route_handlers=[ContentProxyController, AdminFeedbackController],
            middleware=[ProductMiddleware],
            dependencies={
                "product_config": Provide(get_product_config, sync_to_thread=False),
                "product_id": Provide(get_product_id, sync_to_thread=False),
                "session": Provide(lambda: self.session, sync_to_thread=False),
                "feedback_service": Provide(lambda: self.service, sync_to_thread=False),
                "content_proxy": Provide(lambda: self.proxy, sync_to_thread=False),
                "r2_storage": Provide(lambda: self.r2, sync_to_thread=False),
            },
            state=State(
                {
                    "jwt_service": self.jwt,
                    "token_revocation": TokenRevocationService(None, max_token_ttl_seconds=0),
                    "legal_registry": self.registry,
                }
            ),
        )

    def bearer(self, user: User, *, product: str | None = None) -> dict[str, str]:
        token, _ = self.jwt.create_access_token(
            user.id,
            product_id=user.product_id,
            legal_digest=self.registry.required_digest(
                _config_for(user.product_id), today=datetime.now(UTC).date()
            ),
        )
        return {"Authorization": f"Bearer {token}", "X-Product-Id": product or user.product_id}

    def cookie(self, user: User) -> dict[str, str]:
        token, _ = self.jwt.create_content_token(
            user.id, product_id=user.product_id, ttl=timedelta(hours=1)
        )
        return {"Cookie": f"{get_settings().content_cookie_name}={token}"}


def _config_for(product_id: str) -> Any:
    from src.core.product_registry import SYNTHARA_CONFIG, VEX_CONFIG

    return VEX_CONFIG if product_id == "vex" else SYNTHARA_CONFIG


async def _user(make_user: UserFactory, *, product_id: str = "vex", admin: bool = False) -> User:
    user = await make_user(email=f"fa-{uuid4().hex[:10]}@example.com", product_id=product_id)
    if admin:
        user.role = UserRole.ADMIN
    return user


async def _output(
    session: AsyncSession,
    owner: User,
    job_id: UUID,
    *,
    content_type: str = "image/png",
    size_bytes: int = SIZE,
) -> GenerationOutput:
    output_id = uuid4()
    ext = content_type.split("/")[1]
    return await OutputRepository(session).create(
        id=output_id,
        user_id=owner.id,
        job_id=job_id,
        storage_key=f"users/{owner.id}/outputs/{job_id}/{output_id}.{ext}",
        content_type=content_type,
        size_bytes=size_bytes,
        format=ext,
        output_index=0,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id=owner.product_id,
    )


async def _report(session: AsyncSession, reporter: User | None, **overrides: Any) -> FeedbackReport:
    fields: dict[str, Any] = {
        "id": new_id(),
        "product_id": reporter.product_id if reporter else "vex",
        "user_id": reporter.id if reporter else None,
        "category": FeedbackCategory.BUG.value,
        "status": FeedbackStatus.OPEN.value,
        "message": MESSAGE,
        "client_path": CLIENT_PATH,
    }
    fields.update(overrides)
    report = FeedbackReport(**fields)
    session.add(report)
    await session.flush()
    return report


async def _reported_output(
    api: Api, make_job: JobFactory, reporter: User, **output_kwargs: Any
) -> tuple[FeedbackReport, GenerationOutput]:
    job = await make_job(user=reporter)
    output = await _output(api.session, reporter, job.id, **output_kwargs)
    report = await _report(
        api.session, reporter, asset_source="output", asset_id=output.id, job_id=job.id
    )
    return report, output


async def _audit_rows(session: AsyncSession) -> list[AdminAuditLog]:
    result = await session.execute(
        select(AdminAuditLog).where(AdminAuditLog.action == "feedback.asset.view")
    )
    return list(result.scalars().all())


def _url(report: FeedbackReport) -> str:
    return f"{FEEDBACK_ASSET_PATH}/{report.id}"


class TestAdminOpensAsset:
    async def test_t1_bearer_streams_image(
        self, db_session: AsyncSession, make_user: UserFactory, make_job: JobFactory
    ) -> None:
        api = Api(db_session)
        admin, reporter = await _user(make_user, admin=True), await _user(make_user)
        report, output = await _reported_output(api, make_job, reporter)

        async with api.client() as client:
            resp = await client.get(_url(report), headers=api.bearer(admin))

        assert resp.status_code == 200
        assert resp.content == PAYLOAD
        assert resp.headers["Cache-Control"] == NO_STORE
        assert resp.headers["Content-Disposition"] == "inline"
        assert resp.headers["ETag"] == f'"{output.id}"'

    async def test_t2_content_cookie_only(
        self, db_session: AsyncSession, make_user: UserFactory, make_job: JobFactory
    ) -> None:
        """A1 — <img>/<video>/new-tab navigation send only the cookie, no Authorization."""
        api = Api(db_session)
        admin, reporter = await _user(make_user, admin=True), await _user(make_user)
        report, _ = await _reported_output(api, make_job, reporter)

        async with api.client() as client:
            resp = await client.get(
                _url(report), headers={**api.cookie(admin), "X-Product-Id": "vex"}
            )

        assert resp.status_code == 200
        assert resp.content == PAYLOAD
        assert resp.headers["Cache-Control"] == NO_STORE

    async def test_upload_asset(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        make_user_image: UserImageFactory,
    ) -> None:
        api = Api(db_session)
        admin, reporter = await _user(make_user, admin=True), await _user(make_user)
        image = await make_user_image(user=reporter, size_bytes=SIZE)
        report = await _report(db_session, reporter, asset_source="upload", asset_id=image.id)

        async with api.client() as client:
            resp = await client.get(_url(report), headers=api.bearer(admin))

        assert resp.status_code == 200
        assert resp.content == PAYLOAD

    async def test_t3_video_ranges(
        self, db_session: AsyncSession, make_user: UserFactory, make_job: JobFactory
    ) -> None:
        api = Api(db_session, r2=_StubR2(content_type="video/mp4"))
        admin, reporter = await _user(make_user, admin=True), await _user(make_user)
        report, _ = await _reported_output(api, make_job, reporter, content_type="video/mp4")
        headers = api.bearer(admin)

        async with api.client() as client:
            partial = await client.get(_url(report), headers={**headers, "Range": "bytes=0-99"})
            beyond = await client.get(
                _url(report), headers={**headers, "Range": f"bytes={SIZE}-{SIZE + 50}"}
            )

        assert partial.status_code == 206
        assert partial.headers["Content-Range"] == f"bytes 0-99/{SIZE}"
        assert partial.headers["Cache-Control"] == NO_STORE
        assert partial.content == PAYLOAD
        assert beyond.status_code == 416
        assert beyond.headers["Content-Range"] == f"bytes */{SIZE}"


class TestAccessControl:
    async def test_t4_non_admin_and_cross_product(
        self, db_session: AsyncSession, make_user: UserFactory, make_job: JobFactory
    ) -> None:
        api = Api(db_session)
        reporter = await _user(make_user)
        syn_admin = await _user(make_user, product_id="synthara", admin=True)
        report, _ = await _reported_output(api, make_job, reporter)

        async with api.client() as client:
            own = await client.get(_url(report), headers=api.bearer(reporter))
            other_product = await client.get(_url(report), headers=api.bearer(syn_admin))

        # Same refusal as the other admin routes (shared admin dependency).
        assert own.status_code == 401
        assert own.content != PAYLOAD
        # An admin of product B: the report is invisible in product B.
        assert other_product.status_code == 404
        assert other_product.json()["error"] == "feedback_not_found"
        assert api.r2.calls == []
        assert await _audit_rows(db_session) == []

    async def test_unknown_report(self, db_session: AsyncSession, make_user: UserFactory) -> None:
        api = Api(db_session)
        admin = await _user(make_user, admin=True)
        async with api.client() as client:
            resp = await client.get(f"{FEEDBACK_ASSET_PATH}/{uuid4()}", headers=api.bearer(admin))
        assert resp.status_code == 404
        assert resp.json()["error"] == "feedback_not_found"

    async def test_no_credentials(self, db_session: AsyncSession) -> None:
        api = Api(db_session)
        async with api.client() as client:
            resp = await client.get(
                f"{FEEDBACK_ASSET_PATH}/{uuid4()}", headers={"X-Product-Id": "vex"}
            )
        assert resp.status_code == 401


class TestAssetNotFound:
    async def test_t5_variants(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        make_job: JobFactory,
    ) -> None:
        api = Api(db_session)
        admin, reporter = await _user(make_user, admin=True), await _user(make_user)
        no_asset = await _report(db_session, reporter)
        deleted, output = await _reported_output(api, make_job, reporter)
        await OutputRepository(db_session).delete(output.id, user_id=reporter.id)
        purged_report, _ = await _reported_output(api, make_job, reporter)
        await db_session.execute(
            update(FeedbackReport)
            .where(FeedbackReport.id == purged_report.id)
            .values(user_id=None)
            .execution_options(synchronize_session=False)
        )
        urls = [_url(r) for r in (no_asset, deleted, purged_report)]
        headers = api.bearer(admin)  # read admin attributes before the identity map is expired
        await db_session.commit()
        db_session.expire_all()

        async with api.client() as client:
            responses = [await client.get(url, headers=headers) for url in urls]

        for resp in responses:
            assert resp.status_code == 404
            assert resp.json()["error"] == "asset_not_found"
        assert api.r2.calls == []
        assert await _audit_rows(db_session) == []

    async def test_t6_tampered_row_pointing_at_another_users_output(
        self, db_session: AsyncSession, make_user: UserFactory, make_job: JobFactory
    ) -> None:
        """A2 — resolution is scoped to the reporter, not to whoever owns the asset."""
        api = Api(db_session)
        admin, reporter = await _user(make_user, admin=True), await _user(make_user)
        victim = await _user(make_user)
        victim_job = await make_job(user=victim)
        victim_output = await _output(db_session, victim, victim_job.id)
        report = await _report(
            db_session, reporter, asset_source="output", asset_id=victim_output.id
        )

        async with api.client() as client:
            resp = await client.get(_url(report), headers=api.bearer(admin))

        assert resp.status_code == 404
        assert resp.json()["error"] == "asset_not_found"
        assert api.r2.calls == []
        assert await _audit_rows(db_session) == []


class TestAudit:
    async def test_t7_rows_only_on_view_start(
        self, db_session: AsyncSession, make_user: UserFactory, make_job: JobFactory
    ) -> None:
        api = Api(db_session, r2=_StubR2(content_type="video/mp4"))
        admin, reporter = await _user(make_user, admin=True), await _user(make_user)
        report, output = await _reported_output(api, make_job, reporter, content_type="video/mp4")
        headers = api.bearer(admin)

        async with api.client() as client:
            await client.get(_url(report), headers=headers)
            assert len(await _audit_rows(db_session)) == 1
            await client.get(_url(report), headers={**headers, "Range": "bytes=0-"})
            assert len(await _audit_rows(db_session)) == 2
            mid = await client.get(_url(report), headers={**headers, "Range": "bytes=50-"})
            assert mid.status_code == 206
            assert len(await _audit_rows(db_session)) == 2
            bad = await client.get(_url(report), headers={**headers, "Range": "bytes=999-"})
            assert bad.status_code == 416
            assert len(await _audit_rows(db_session)) == 2

        rows = await _audit_rows(db_session)
        for row in rows:
            assert row.actor_id == admin.id
            assert row.target_user_id == reporter.id
            assert row.product_id == "vex"
            assert row.action == "feedback.asset.view"
            assert row.source == "api"
            assert row.detail == f"report {report.id} asset output:{output.id}"
            assert MESSAGE not in row.detail
            assert CLIENT_PATH not in row.detail

    async def test_open_failures_and_304_are_not_audited(
        self, db_session: AsyncSession, make_user: UserFactory, make_job: JobFactory
    ) -> None:
        """Only a successfully opened R2 stream starts an auditable view."""
        api = Api(db_session)
        admin, reporter = await _user(make_user, admin=True), await _user(make_user)
        report, output = await _reported_output(api, make_job, reporter)
        headers = api.bearer(admin)

        @asynccontextmanager
        async def storage_failure(
            _key: str, *, range_header: str | None = None
        ) -> AsyncIterator[ObjectStream]:
            del range_header
            raise StorageError("R2 unavailable")
            yield  # pragma: no cover - makes this an async generator

        api.r2.stream_object = storage_failure  # type: ignore[method-assign]
        with patch("src.api.routes.content.logger") as route_logger:
            async with api.client() as client:
                failed = await client.get(_url(report), headers=headers)
                not_modified = await client.get(
                    _url(report), headers={**headers, "If-None-Match": f'"{output.id}"'}
                )

        assert failed.status_code == 502
        assert not_modified.status_code == 304
        assert await _audit_rows(db_session) == []
        route_logger.info.assert_not_called()

    async def test_stale_size_r2_range_rejection_is_not_audited(
        self, db_session: AsyncSession, make_user: UserFactory, make_job: JobFactory
    ) -> None:
        api = Api(db_session)
        admin, reporter = await _user(make_user, admin=True), await _user(make_user)
        report, _ = await _reported_output(api, make_job, reporter)

        @asynccontextmanager
        async def range_failure(
            _key: str, *, range_header: str | None = None
        ) -> AsyncIterator[ObjectStream]:
            del range_header
            raise StorageRangeNotSatisfiableError("stale size")
            yield  # pragma: no cover - makes this an async generator

        api.r2.stream_object = range_failure  # type: ignore[method-assign]
        async with api.client() as client:
            response = await client.get(
                _url(report), headers={**api.bearer(admin), "Range": "bytes=0-99"}
            )

        assert response.status_code == 416
        assert await _audit_rows(db_session) == []

    async def test_reporter_purge_keeps_view_audit_and_nulls_its_target(
        self, db_session: AsyncSession, make_user: UserFactory, make_job: JobFactory
    ) -> None:
        """The view trail survives reporter erasure while its personal target does not."""
        api = Api(db_session)
        admin, reporter = await _user(make_user, admin=True), await _user(make_user)
        report, _ = await _reported_output(api, make_job, reporter)

        async with api.client() as client:
            response = await client.get(_url(report), headers=api.bearer(admin))
        assert response.status_code == 200
        audit = (await _audit_rows(db_session))[0]
        audit_id, report_id, admin_id = audit.id, report.id, admin.id

        await db_session.execute(delete(User).where(User.id == reporter.id))
        await db_session.flush()
        db_session.expire_all()

        surviving_audit = await db_session.get(AdminAuditLog, audit_id)
        surviving_report = await db_session.get(FeedbackReport, report_id)
        assert surviving_audit is not None
        assert surviving_audit.target_user_id is None
        assert surviving_audit.actor_id == admin_id
        assert surviving_report is not None
        assert surviving_report.user_id is None

    async def test_target_purge_nulls_a_preexisting_role_audit_row(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """The migration also fixes the pre-existing role/permission audit trap."""
        actor, target = await _user(make_user, admin=True), await _user(make_user)
        audit = AdminAuditLog(
            id=new_id(),
            actor_id=actor.id,
            target_user_id=target.id,
            product_id="vex",
            action="role.grant",
            detail="role changed from user to admin",
            source="api",
        )
        db_session.add(audit)
        await db_session.flush()
        audit_id, actor_id = audit.id, actor.id

        await db_session.execute(delete(User).where(User.id == target.id))
        await db_session.flush()
        db_session.expire_all()

        surviving_audit = await db_session.get(AdminAuditLog, audit_id)
        assert surviving_audit is not None
        assert surviving_audit.target_user_id is None
        assert surviving_audit.actor_id == actor_id

    async def test_actor_purge_remains_blocked_by_audit_rows(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """Actor deletion stays deliberate: accountability rows retain their actor FK."""
        actor, target = await _user(make_user, admin=True), await _user(make_user)
        db_session.add(
            AdminAuditLog(
                id=new_id(),
                actor_id=actor.id,
                target_user_id=target.id,
                product_id="vex",
                action="role.grant",
                detail="role changed from user to admin",
                source="api",
            )
        )
        await db_session.flush()

        with pytest.raises(IntegrityError):
            await db_session.execute(delete(User).where(User.id == actor.id))
            await db_session.flush()
        await db_session.rollback()


class TestOwnerRoutesUnchanged:
    async def test_t8_admin_still_404s_on_owner_url_and_owner_keeps_immutable_cache(
        self, db_session: AsyncSession, make_user: UserFactory, make_job: JobFactory
    ) -> None:
        api = Api(db_session)
        admin, reporter = await _user(make_user, admin=True), await _user(make_user)
        _, output = await _reported_output(api, make_job, reporter)
        owner_url = f"/v1/content/outputs/{output.id}"

        async with api.client() as client:
            as_admin = await client.get(owner_url, headers=api.bearer(admin))
            as_owner = await client.get(owner_url, headers=api.bearer(reporter))

        assert as_admin.status_code == 404
        assert as_owner.status_code == 200
        assert as_owner.headers["Cache-Control"] == (f"private, max-age={api.proxy.ttl}, immutable")


class TestAssetUrlInAdminResponses:
    async def test_t9_list_detail_and_patch(
        self, db_session: AsyncSession, make_user: UserFactory, make_job: JobFactory
    ) -> None:
        api = Api(db_session)
        admin, reporter = await _user(make_user, admin=True), await _user(make_user)
        with_asset, _ = await _reported_output(api, make_job, reporter)
        without_asset = await _report(db_session, reporter)
        headers = api.bearer(admin)

        async with api.client() as client:
            listed = await client.get("/v1/admin/feedback", headers=headers)
            detail = await client.get(f"/v1/admin/feedback/{with_asset.id}", headers=headers)
            patched = await client.patch(
                f"/v1/admin/feedback/{with_asset.id}",
                json={"admin_note": "looking"},
                headers=headers,
            )
            plain = await client.get(f"/v1/admin/feedback/{without_asset.id}", headers=headers)

        by_id = {item["id"]: item for item in listed.json()["items"]}
        assert by_id[str(with_asset.id)]["asset_url"] == _url(with_asset)
        assert by_id[str(without_asset.id)]["asset_url"] is None
        assert detail.json()["asset_url"] == _url(with_asset)
        assert patched.json()["asset_url"] == _url(with_asset)
        assert plain.json()["asset_url"] is None
        assert plain.json()["asset_ref"] is None
