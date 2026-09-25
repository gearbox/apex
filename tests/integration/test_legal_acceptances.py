"""Legal acceptance against real PostgreSQL 16.

Contracts: C6 (register validation creates no user), C7 (signup rows +
rollback), C8 (lgl in register/login/refresh tokens), C11 (re-acceptance
flow), C12 (idempotent acceptance), C13 (withdrawal on closure), C14 (CHECK
constraint), C15 (product scoping). Round-2 remediation: F1 (event order is
``seq``, ``created_at`` is insertion time), F2 (no ACCEPT after closure), F3
(concurrency tests synchronise on ``pg_locks``, plus a no-lock canary).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import msgspec
import pytest
from sqlalchemy import DateTime, delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.security import JWTConfig, JWTService, PasswordService
from src.api.security.guards import _enforce_legal_acceptance
from src.api.services.auth import AuthService
from src.api.services.legal.acceptance import LegalAcceptanceService, RequestContext
from src.api.services.legal.errors import (
    LegalAcceptanceRequiredError,
    LegalAccountInactiveError,
    LegalSubmissionIncompleteError,
    LegalVersionStaleError,
)
from src.api.services.token_revocation import TokenRevocationService
from src.api.services.user import UserService
from src.core.enums import LegalAcceptanceSource, LegalAction, LegalDocumentType, Product
from src.core.product_registry import SYNTHARA_CONFIG, VEX_CONFIG
from src.core.uid import new_id
from src.db.models.legal import LegalAcceptance
from src.db.models.user import User
from src.db.repositories.legal import LegalAcceptanceRepository
from src.db.repositories.user import UserRepository
from tests.integration.pg_locks import wait_for_advisory_lock_waiter
from tests.legal_support import accept_all_current, make_legal_document, make_legal_registry

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.api.services.legal.registry import LegalDocument, LegalDocumentRegistry
    from tests.integration.conftest import UserFactory

JWT_SECRET = "integration-legal-secret-key-32-bytes-long"
CONTEXT = RequestContext(ip_address="192.0.2.44", user_agent="IntegrationUA/2.0")
PASSWORD = "correct horse battery staple"


def _today() -> date:
    return datetime.now(UTC).date()


def _jwt() -> JWTService:
    return JWTService(JWTConfig(secret_key=JWT_SECRET))


def _legal(session: AsyncSession, registry: LegalDocumentRegistry) -> LegalAcceptanceService:
    return LegalAcceptanceService(
        registry=registry, repository=LegalAcceptanceRepository(session), session=session
    )


def _auth(
    session: AsyncSession, registry: LegalDocumentRegistry, jwt_service: JWTService
) -> AuthService:
    return AuthService(
        repository=UserRepository(session),
        jwt_service=jwt_service,
        password_service=PasswordService(),
        token_revocation_service=TokenRevocationService(None, max_token_ttl_seconds=0),
        legal_acceptance_service=_legal(session, registry),
        session=session,
    )


def _claims(token: str) -> dict[str, Any]:
    claims: dict[str, Any] = pyjwt.decode(token, options={"verify_signature": False})
    return claims


async def _rows(session: AsyncSession, user_id: UUID) -> list[LegalAcceptance]:
    result = await session.execute(
        select(LegalAcceptance)
        .where(LegalAcceptance.user_id == user_id)
        .order_by(LegalAcceptance.doc_type, LegalAcceptance.seq)
    )
    return list(result.scalars().all())


async def _seqs(session: AsyncSession, user_id: UUID) -> dict[UUID, int]:
    """``{row id: seq}`` — read as columns so no identity-map state is involved."""
    result = await session.execute(
        select(LegalAcceptance.id, LegalAcceptance.seq).where(LegalAcceptance.user_id == user_id)
    )
    return dict(result.tuples().all())


async def _consent_events(session: AsyncSession, user_id: UUID) -> list[tuple[LegalAction, int]]:
    """Sensitive-data consent events for a user as ``(action, seq)``, in seq order."""
    result = await session.execute(
        select(LegalAcceptance.action, LegalAcceptance.seq)
        .where(
            LegalAcceptance.user_id == user_id,
            LegalAcceptance.doc_type == LegalDocumentType.SENSITIVE_DATA_CONSENT,
        )
        .order_by(LegalAcceptance.seq)
    )
    return [(LegalAction(action), seq) for action, seq in result.all()]


async def _commit_user(engine: AsyncEngine, email_prefix: str) -> User:
    """Commit a vex user outside any test transaction (for multi-session tests)."""
    user = User(
        id=new_id(),
        email=f"{email_prefix}-{new_id()}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        session.add(user)
        await session.commit()
    return user


async def _delete_user(engine: AsyncEngine, user_id: UUID) -> None:
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(delete(User).where(User.id == user_id))
        await session.commit()


def _user_service(session: AsyncSession, registry: LegalDocumentRegistry) -> UserService:
    return UserService(
        repository=UserRepository(session),
        password_service=PasswordService(),
        age_verification_service=MagicMock(),
        token_revocation_service=TokenRevocationService(None, max_token_ttl_seconds=0),
        legal_acceptance_service=_legal(session, registry),
        session=session,
    )


async def _accept_committed(
    engine: AsyncEngine,
    registry: LegalDocumentRegistry,
    user_id: UUID,
    documents: Sequence[LegalDocument],
) -> int:
    """Run ``record_acceptances`` in its own committed transaction."""
    async with AsyncSession(bind=engine, expire_on_commit=False) as session, session.begin():
        return await _legal(session, registry).record_acceptances(
            user_id=user_id,
            product=VEX_CONFIG,
            documents=documents,
            source=LegalAcceptanceSource.REACCEPT,
            context=CONTEXT,
        )


async def _user_count(session: AsyncSession) -> int:
    return int((await session.execute(select(func.count()).select_from(User))).scalar_one())


def _guard_connection(registry: LegalDocumentRegistry) -> MagicMock:
    connection = MagicMock()
    connection.scope = {"method": "POST"}
    connection.state = {"product_config": VEX_CONFIG}
    connection.app.state = {"legal_registry": registry}
    return connection


def _mutating_handler() -> MagicMock:
    handler = MagicMock()
    handler.opt = {}
    return handler


class TestRegisterValidation:
    """C6 — invalid submissions fail before any user row exists."""

    @pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "version", "sha"])
    async def test_invalid_submission_creates_no_user(
        self, db_session: AsyncSession, case: str
    ) -> None:
        registry = make_legal_registry()
        submission = accept_all_current(registry, today=_today())
        expected: type[Exception] = LegalSubmissionIncompleteError
        product = "vex"
        if case == "missing":
            submission = submission[1:]
        elif case == "extra":
            product = "synthara"  # requires nothing → every entry is extra
        elif case == "duplicate":
            submission = [*submission, submission[0]]
        elif case == "version":
            first = msgspec.structs.replace(submission[0], version=date(2019, 1, 1))
            submission = [first, *submission[1:]]
            expected = LegalVersionStaleError
        else:
            first = msgspec.structs.replace(submission[0], sha256="e" * 64)
            submission = [first, *submission[1:]]
            expected = LegalVersionStaleError

        before = await _user_count(db_session)
        with pytest.raises(expected):
            await _auth(db_session, registry, _jwt()).register(
                email=f"c6-{case}@example.com",
                password=PASSWORD,
                product_id=product,
                accepted_documents=submission,
                context=CONTEXT,
            )
        assert await _user_count(db_session) == before

    async def test_stale_detail_lists_current(self, db_session: AsyncSession) -> None:
        registry = make_legal_registry()
        submission = [
            msgspec.structs.replace(d, sha256="e" * 64)
            for d in accept_all_current(registry, today=_today())
        ]
        with pytest.raises(LegalVersionStaleError) as exc_info:
            await _auth(db_session, registry, _jwt()).register(
                email="c6-detail@example.com",
                password=PASSWORD,
                product_id="vex",
                accepted_documents=submission,
                context=CONTEXT,
            )
        current = exc_info.value.detail["current"]
        assert {c["sha256"] for c in current} == {
            d.sha256 for d in registry.list_current(Product.VEX, today=_today())
        }


class TestRegisterRecords:
    """C7 — three ACCEPT rows in the user's transaction; rollback removes all."""

    async def test_signup_inserts_three_accept_rows(self, db_session: AsyncSession) -> None:
        registry = make_legal_registry()
        user, _ = await _auth(db_session, registry, _jwt()).register(
            email="c7@example.com",
            password=PASSWORD,
            product_id="vex",
            accepted_documents=accept_all_current(registry, today=_today()),
            context=CONTEXT,
        )

        rows = await _rows(db_session, user.id)
        assert len(rows) == 3
        expected_sha = {
            d.doc_type: d.sha256 for d in registry.list_current(Product.VEX, today=_today())
        }
        for row in rows:
            assert row.action == LegalAction.ACCEPT
            assert row.source == LegalAcceptanceSource.SIGNUP
            assert row.product_id == "vex"
            assert row.content_sha256 == expected_sha[LegalDocumentType(row.doc_type)]
            assert row.ip_address == CONTEXT.ip_address
            assert row.user_agent == CONTEXT.user_agent
            assert row.created_at is not None

    async def test_failure_after_insert_rolls_everything_back(
        self, db_session: AsyncSession
    ) -> None:
        registry = make_legal_registry()
        email = "c7-rollback@example.com"
        savepoint = await db_session.begin_nested()
        with (
            patch.object(AuthService, "_create_token_pair", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await _auth(db_session, registry, _jwt()).register(
                email=email,
                password=PASSWORD,
                product_id="vex",
                accepted_documents=accept_all_current(registry, today=_today()),
                context=CONTEXT,
            )
        await savepoint.rollback()

        assert (
            await db_session.execute(select(User).where(User.email == email))
        ).scalar_one_or_none() is None
        count = (
            await db_session.execute(select(func.count()).select_from(LegalAcceptance))
        ).scalar_one()
        assert count == 0


class TestDigestInTokens:
    """C8 — register/login/refresh tokens carry lgl; synthara tokens don't."""

    async def test_vex_register_login_refresh_carry_lgl(self, db_session: AsyncSession) -> None:
        registry = make_legal_registry()
        jwt_service = _jwt()
        auth = _auth(db_session, registry, jwt_service)
        expected = registry.required_digest(VEX_CONFIG, today=_today())
        assert expected is not None

        _, registered = await auth.register(
            email="c8@example.com",
            password=PASSWORD,
            product_id="vex",
            accepted_documents=accept_all_current(registry, today=_today()),
            context=CONTEXT,
        )
        _, logged_in = await auth.login(email="c8@example.com", password=PASSWORD, product_id="vex")
        refreshed, _ = await auth.refresh_tokens(logged_in.refresh_token)

        for pair in (registered, logged_in, refreshed):
            assert _claims(pair.access_token)["lgl"] == expected
            payload = jwt_service.decode_access_token(pair.access_token)
            assert payload is not None
            assert payload.legal_digest == expected

    async def test_synthara_tokens_have_no_lgl(self, db_session: AsyncSession) -> None:
        registry = make_legal_registry()
        auth = _auth(db_session, registry, _jwt())
        _, registered = await auth.register(
            email="c8-synthara@example.com",
            password=PASSWORD,
            product_id="synthara",
            accepted_documents=[],
            context=CONTEXT,
        )
        _, logged_in = await auth.login(
            email="c8-synthara@example.com", password=PASSWORD, product_id="synthara"
        )
        refreshed, _ = await auth.refresh_tokens(logged_in.refresh_token)
        for pair in (registered, logged_in, refreshed):
            assert "lgl" not in _claims(pair.access_token)


class TestReacceptanceFlow:
    """C11 — a flagged v2 effective today stales old tokens until re-accept + refresh."""

    async def test_full_flow(self, db_session: AsyncSession) -> None:
        today = _today()
        v1 = make_legal_registry()
        jwt_service = _jwt()
        user, v1_tokens = await _auth(db_session, v1, jwt_service).register(
            email="c11@example.com",
            password=PASSWORD,
            product_id="vex",
            accepted_documents=accept_all_current(v1, today=today),
            context=CONTEXT,
        )
        user_id = user.id

        # Publish terms v2 (flagged) effective today.
        v2 = make_legal_registry(
            *(v1.current(Product.VEX, t, today=today) for t in LegalDocumentType),
            make_legal_document(LegalDocumentType.TERMS, today, content="# Terms v2\n"),
        )
        old_payload = jwt_service.decode_access_token(v1_tokens.access_token)
        assert old_payload is not None
        with pytest.raises(LegalAcceptanceRequiredError):
            _enforce_legal_acceptance(_guard_connection(v2), _mutating_handler(), old_payload)

        legal_v2 = _legal(db_session, v2)
        documents = legal_v2.validate_submission(
            VEX_CONFIG, accept_all_current(v2, today=today), today=today
        )
        inserted = await legal_v2.record_acceptances(
            user_id=user_id,
            product=VEX_CONFIG,
            documents=documents,
            source=LegalAcceptanceSource.REACCEPT,
            context=CONTEXT,
        )
        assert inserted == 1  # only terms changed

        refreshed, _ = await _auth(db_session, v2, jwt_service).refresh_tokens(
            v1_tokens.refresh_token
        )
        new_payload = jwt_service.decode_access_token(refreshed.access_token)
        assert new_payload is not None
        assert new_payload.legal_digest == v2.required_digest(VEX_CONFIG, today=today)
        _enforce_legal_acceptance(_guard_connection(v2), _mutating_handler(), new_payload)

        reaccept_rows = [
            r
            for r in await _rows(db_session, user_id)
            if r.source == LegalAcceptanceSource.REACCEPT
        ]
        assert [(r.doc_type, r.version) for r in reaccept_rows] == [
            (LegalDocumentType.TERMS, today)
        ]


class TestIdempotentAcceptance:
    """C12 — re-submitting current accepted versions inserts 0 rows."""

    async def test_second_submission_inserts_nothing(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        user = await make_user(email="c12@example.com")
        registry = make_legal_registry()
        legal = _legal(db_session, registry)
        docs = legal.validate_submission(
            VEX_CONFIG, accept_all_current(registry, today=_today()), today=_today()
        )
        kwargs: dict[str, Any] = {
            "user_id": user.id,
            "product": VEX_CONFIG,
            "documents": docs,
            "source": LegalAcceptanceSource.REACCEPT,
            "context": CONTEXT,
        }
        assert await legal.record_acceptances(**kwargs) == 3
        assert await legal.record_acceptances(**kwargs) == 0
        assert len(await _rows(db_session, user.id)) == 3

    async def _race_two_submissions(
        self, db_engine: AsyncEngine, *, prove_second_blocks: bool
    ) -> tuple[int, int, list[LegalAcceptance]]:
        """First submit inserts and holds its transaction open; then a second submits.

        With ``prove_second_blocks`` the first is released only once
        ``pg_locks`` shows the second waiting on the ledger lock; otherwise the
        second runs to completion first (the no-lock canary, where nothing blocks).
        """
        user = await _commit_user(db_engine, "c12-race")
        registry = make_legal_registry()
        documents = registry.list_current(Product.VEX, today=_today())
        first_inserted = asyncio.Event()
        release_first = asyncio.Event()

        async def submit(*, first: bool) -> int:
            async with (
                AsyncSession(bind=db_engine, expire_on_commit=False) as session,
                session.begin(),
            ):
                inserted = await _legal(session, registry).record_acceptances(
                    user_id=user.id,
                    product=VEX_CONFIG,
                    documents=documents,
                    source=LegalAcceptanceSource.REACCEPT,
                    context=CONTEXT,
                )
                if first:
                    first_inserted.set()
                    await release_first.wait()
                return inserted

        try:
            first_task = asyncio.create_task(submit(first=True))
            await first_inserted.wait()
            second_task = asyncio.create_task(submit(first=False))
            if prove_second_blocks:
                await wait_for_advisory_lock_waiter(db_engine)
            else:
                await asyncio.wait_for(asyncio.shield(second_task), timeout=5.0)
            release_first.set()
            first, second = await asyncio.gather(first_task, second_task)

            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                rows = await _rows(session, user.id)
            return first, second, rows
        finally:
            release_first.set()
            await _delete_user(db_engine, user.id)

    async def test_concurrent_reacceptance_inserts_once(self, db_engine: AsyncEngine) -> None:
        """A transaction-scoped user lock turns concurrent submits into 3 then 0."""
        first, second, rows = await self._race_two_submissions(db_engine, prove_second_blocks=True)
        assert (first, second) == (3, 0)
        assert {row.doc_type for row in rows} == set(LegalDocumentType)
        assert all(row.action == LegalAction.ACCEPT for row in rows)

    async def test_concurrent_reacceptance_duplicates_without_lock(
        self, db_engine: AsyncEngine
    ) -> None:
        """Canary: with the ledger lock disabled, the same race double-inserts.

        Exists only to prove ``test_concurrent_reacceptance_inserts_once`` is
        meaningful — if this stopped producing duplicates, the positive test
        would no longer be exercising the lock.
        """
        with patch.object(LegalAcceptanceRepository, "lock_user_ledger", AsyncMock()):
            first, second, rows = await self._race_two_submissions(
                db_engine, prove_second_blocks=False
            )
        assert (first, second) == (3, 3)
        assert len(rows) == 6

    async def test_accept_after_withdraw_same_version_inserts(
        self, db_session: AsyncSession
    ) -> None:
        user = User(
            id=new_id(),
            email=f"c12-withdraw-{new_id()}@example.com",
            password_hash="hash",
            product_id="vex",
        )
        db_session.add(user)
        await db_session.flush()
        registry = make_legal_registry()
        service = _legal(db_session, registry)
        today = _today()
        consent = registry.current(
            Product.VEX, LegalDocumentType.SENSITIVE_DATA_CONSENT, today=today
        )

        assert (
            await service.record_acceptances(
                user_id=user.id,
                product=VEX_CONFIG,
                documents=[consent],
                source=LegalAcceptanceSource.REACCEPT,
                context=CONTEXT,
            )
            == 1
        )
        await service.record_consent_withdrawal(
            user_id=user.id, product=VEX_CONFIG, context=CONTEXT, today=today
        )
        assert (
            await service.record_acceptances(
                user_id=user.id,
                product=VEX_CONFIG,
                documents=[consent],
                source=LegalAcceptanceSource.REACCEPT,
                context=CONTEXT,
            )
            == 1
        )

        rows = [
            row
            for row in await _rows(db_session, user.id)
            if row.doc_type == LegalDocumentType.SENSITIVE_DATA_CONSENT
        ]
        assert [row.action for row in rows] == [
            LegalAction.ACCEPT,
            LegalAction.WITHDRAW,
            LegalAction.ACCEPT,
        ]


class TestWithdrawal:
    """C13 — closing a vex account records one consent WITHDRAW; synthara none."""

    def _user_service(self, session: AsyncSession) -> UserService:
        return _user_service(session, make_legal_registry())

    async def test_vex_closure_inserts_withdraw(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        user = await make_user(email="c13@example.com")
        await self._user_service(db_session).deactivate_account(
            user.id, product=VEX_CONFIG, context=CONTEXT
        )
        rows = await _rows(db_session, user.id)
        assert [(r.doc_type, r.action, r.source, r.content_sha256) for r in rows] == [
            (
                LegalDocumentType.SENSITIVE_DATA_CONSENT,
                LegalAction.WITHDRAW,
                LegalAcceptanceSource.ACCOUNT_CLOSURE,
                None,
            )
        ]
        assert rows[0].ip_address == CONTEXT.ip_address

    async def test_synthara_closure_inserts_nothing(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        user = await make_user(email="c13-s@example.com", product_id="synthara")
        await self._user_service(db_session).deactivate_account(
            user.id, product=SYNTHARA_CONFIG, context=CONTEXT
        )
        assert await _rows(db_session, user.id) == []


class TestAppendOnlyConstraint:
    """C14 — the CHECK constraint rejects an ACCEPT without a content hash."""

    async def test_accept_without_hash_rejected(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        user = await make_user(email="c14@example.com")
        savepoint = await db_session.begin_nested()
        db_session.add(
            LegalAcceptance(
                id=new_id(),
                user_id=user.id,
                product_id="vex",
                doc_type=LegalDocumentType.TERMS,
                action=LegalAction.ACCEPT,
                version=date(2020, 1, 1),
                content_sha256=None,
                source=LegalAcceptanceSource.SIGNUP,
            )
        )
        with pytest.raises(IntegrityError, match="ck_legal_acceptances_accept_has_hash"):
            await db_session.flush()
        await savepoint.rollback()

    async def test_withdraw_without_hash_allowed(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        user = await make_user(email="c14-w@example.com")
        db_session.add(
            LegalAcceptance(
                id=new_id(),
                user_id=user.id,
                product_id="vex",
                doc_type=LegalDocumentType.SENSITIVE_DATA_CONSENT,
                action=LegalAction.WITHDRAW,
                version=date(2020, 1, 1),
                content_sha256=None,
                source=LegalAcceptanceSource.ACCOUNT_CLOSURE,
            )
        )
        await db_session.flush()


class TestProductScoping:
    """C15 — latest_per_type never crosses products for the same user id."""

    async def test_other_product_rows_are_invisible(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        user = await make_user(email="c15@example.com")
        for product_id, version in (("vex", date(2020, 1, 1)), ("synthara", date(2021, 1, 1))):
            db_session.add(
                LegalAcceptance(
                    id=new_id(),
                    user_id=user.id,
                    product_id=product_id,
                    doc_type=LegalDocumentType.TERMS,
                    action=LegalAction.ACCEPT,
                    version=version,
                    content_sha256="a" * 64,
                    source=LegalAcceptanceSource.SIGNUP,
                )
            )
        await db_session.flush()
        repo = LegalAcceptanceRepository(db_session)

        vex = await repo.latest_per_type(user_id=user.id, product_id="vex")
        synthara = await repo.latest_per_type(user_id=user.id, product_id="synthara")

        assert {t: r.version for t, r in vex.items()} == {LegalDocumentType.TERMS: date(2020, 1, 1)}
        assert {t: r.product_id for t, r in synthara.items()} == {
            LegalDocumentType.TERMS: "synthara"
        }

    async def test_latest_is_most_recent_event(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        user = await make_user(email="c15-latest@example.com")
        for action in (LegalAction.ACCEPT, LegalAction.WITHDRAW):
            db_session.add(
                LegalAcceptance(
                    id=new_id(),
                    user_id=user.id,
                    product_id="vex",
                    doc_type=LegalDocumentType.SENSITIVE_DATA_CONSENT,
                    action=action,
                    version=date(2020, 1, 1),
                    content_sha256="a" * 64 if action == LegalAction.ACCEPT else None,
                    source=LegalAcceptanceSource.SIGNUP,
                )
            )
            await db_session.flush()
        latest = await LegalAcceptanceRepository(db_session).latest_per_type(
            user_id=user.id, product_id="vex"
        )
        winner = latest[LegalDocumentType.SENSITIVE_DATA_CONSENT]
        seqs = await _seqs(db_session, user.id)
        assert winner.action == LegalAction.WITHDRAW
        assert seqs[winner.id] == max(seqs.values())
        assert len(set(seqs.values())) == 2


class TestEventOrdering:
    """F1 — order is the serialized write order (seq), not transaction start time."""

    async def test_withdraw_after_accept_with_earlier_tx_start_is_latest(
        self, db_engine: AsyncEngine
    ) -> None:
        user = await _commit_user(db_engine, "f1-order")
        registry = make_legal_registry()
        today = _today()
        try:
            async with (
                AsyncSession(bind=db_engine, expire_on_commit=False) as closure,
                closure.begin(),
            ):
                # T1 (closure) starts first: pin its snapshot/transaction start,
                # and force a real gap before T2 begins.
                await closure.execute(text("SELECT 1"))
                await closure.execute(text("SELECT pg_sleep(0.01)"))

                # T2 (acceptance) starts later, takes the lock first, commits.
                inserted = await _accept_committed(
                    db_engine,
                    registry,
                    user.id,
                    registry.list_current(Product.VEX, today=today),
                )
                assert inserted == 3

                # T1 now takes the lock and writes WITHDRAW after T2's ACCEPT.
                await _legal(closure, registry).record_consent_withdrawal(
                    user_id=user.id, product=VEX_CONFIG, context=CONTEXT, today=today
                )

            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                latest = await LegalAcceptanceRepository(session).latest_per_type(
                    user_id=user.id, product_id="vex"
                )
                digest = await _legal(session, registry).satisfied_digest(
                    user_id=user.id, product=VEX_CONFIG, today=today
                )
                events = await _consent_events(session, user.id)
            assert latest[LegalDocumentType.SENSITIVE_DATA_CONSENT].action == LegalAction.WITHDRAW
            assert digest is None
            (accept_seq,) = [seq for action, seq in events if action == LegalAction.ACCEPT]
            (withdraw_seq,) = [seq for action, seq in events if action == LegalAction.WITHDRAW]
            assert withdraw_seq > accept_seq
        finally:
            await _delete_user(db_engine, user.id)

    async def test_created_at_is_insertion_time(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """created_at is clock_timestamp() at INSERT, not the transaction start."""
        user = await make_user(email="f1-created-at@example.com")
        await db_session.execute(text("SELECT pg_sleep(0.05)"))
        row = LegalAcceptance(
            id=new_id(),
            user_id=user.id,
            product_id="vex",
            doc_type=LegalDocumentType.TERMS,
            action=LegalAction.ACCEPT,
            version=date(2020, 1, 1),
            content_sha256="a" * 64,
            source=LegalAcceptanceSource.SIGNUP,
        )
        db_session.add(row)
        await db_session.flush()

        created_at, tx_start = (
            await db_session.execute(
                select(
                    LegalAcceptance.created_at,
                    func.transaction_timestamp(type_=DateTime(timezone=True)),
                ).where(LegalAcceptance.id == row.id)
            )
        ).one()
        assert (created_at - tx_start).total_seconds() >= 0.05


class TestAcceptanceVsClosure:
    """F2 — an acceptance can never become the latest event after account closure."""

    async def test_acceptance_waiting_on_closure_is_rejected(self, db_engine: AsyncEngine) -> None:
        user = await _commit_user(db_engine, "f2-wait")
        registry = make_legal_registry()
        documents = registry.list_current(Product.VEX, today=_today())
        assert await _accept_committed(db_engine, registry, user.id, documents) == 3
        closure_locked = asyncio.Event()
        release_closure = asyncio.Event()

        async def close_account() -> None:
            async with (
                AsyncSession(bind=db_engine, expire_on_commit=False) as session,
                session.begin(),
            ):
                service = _user_service(session, registry)
                repo = service._legal._repo
                original_lock = repo.lock_user_ledger

                async def lock_then_pause(*, user_id: UUID) -> None:
                    await original_lock(user_id=user_id)
                    closure_locked.set()
                    await release_closure.wait()

                repo.lock_user_ledger = lock_then_pause  # type: ignore[method-assign]
                await service.deactivate_account(user.id, product=VEX_CONFIG, context=CONTEXT)

        try:
            closure_task = asyncio.create_task(close_account())
            await closure_locked.wait()
            accept_task = asyncio.create_task(
                _accept_committed(db_engine, registry, user.id, documents)
            )
            await wait_for_advisory_lock_waiter(db_engine)
            release_closure.set()
            await closure_task
            with pytest.raises(LegalAccountInactiveError):
                await accept_task

            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                latest = await LegalAcceptanceRepository(session).latest_per_type(
                    user_id=user.id, product_id="vex"
                )
                events = await _consent_events(session, user.id)
            assert latest[LegalDocumentType.SENSITIVE_DATA_CONSENT].action == LegalAction.WITHDRAW
            withdraw_seq = next(seq for action, seq in events if action == LegalAction.WITHDRAW)
            assert not [
                seq for action, seq in events if action == LegalAction.ACCEPT and seq > withdraw_seq
            ]
        finally:
            release_closure.set()
            await _delete_user(db_engine, user.id)

    async def test_acceptance_before_closure_then_withdraw_is_latest(
        self, db_engine: AsyncEngine
    ) -> None:
        user = await _commit_user(db_engine, "f2-before")
        registry = make_legal_registry()
        documents = registry.list_current(Product.VEX, today=_today())
        accepted = asyncio.Event()
        release_acceptance = asyncio.Event()

        async def accept() -> int:
            async with (
                AsyncSession(bind=db_engine, expire_on_commit=False) as session,
                session.begin(),
            ):
                inserted = await _legal(session, registry).record_acceptances(
                    user_id=user.id,
                    product=VEX_CONFIG,
                    documents=documents,
                    source=LegalAcceptanceSource.REACCEPT,
                    context=CONTEXT,
                )
                accepted.set()
                await release_acceptance.wait()
                return inserted

        async def close_account() -> None:
            async with (
                AsyncSession(bind=db_engine, expire_on_commit=False) as session,
                session.begin(),
            ):
                await _user_service(session, registry).deactivate_account(
                    user.id, product=VEX_CONFIG, context=CONTEXT
                )

        try:
            accept_task = asyncio.create_task(accept())
            await accepted.wait()
            closure_task = asyncio.create_task(close_account())
            await wait_for_advisory_lock_waiter(db_engine)
            release_acceptance.set()
            assert await accept_task == 3
            await closure_task

            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                latest = await LegalAcceptanceRepository(session).latest_per_type(
                    user_id=user.id, product_id="vex"
                )
                events = await _consent_events(session, user.id)
                is_active = (
                    await session.execute(select(User.is_active).where(User.id == user.id))
                ).scalar_one()
            assert latest[LegalDocumentType.SENSITIVE_DATA_CONSENT].action == LegalAction.WITHDRAW
            assert [action for action, _ in events] == [LegalAction.ACCEPT, LegalAction.WITHDRAW]
            assert is_active is False
        finally:
            release_acceptance.set()
            await _delete_user(db_engine, user.id)
