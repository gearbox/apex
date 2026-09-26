"""LegalAcceptanceService with a mocked repository.

DB-backed behaviour (rollback, CHECK constraint, product scoping, real
idempotency across requests) lives in tests/integration/test_legal_acceptances.py.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call
from uuid import uuid4

import msgspec
import pytest
import structlog

from src.api.services.legal.acceptance import (
    AcceptedDocument,
    LegalAcceptanceService,
    RequestContext,
)
from src.api.services.legal.errors import (
    LegalAccountInactiveError,
    LegalSubmissionIncompleteError,
    LegalVersionStaleError,
)
from src.core.enums import LegalAcceptanceSource, LegalAction, LegalDocumentType, Product
from src.core.product_registry import SYNTHARA_CONFIG, VEX_CONFIG
from src.db.models.legal import LegalAcceptance
from src.db.repositories.legal import LegalAcceptanceRepository
from tests.legal_support import (
    DEFAULT_LEGAL_VERSION,
    accept_all_current,
    accepted,
    make_legal_document,
    make_legal_registry,
)

pytestmark = pytest.mark.unit

TODAY = date(2026, 10, 1)
CONTEXT = RequestContext(ip_address="198.51.100.23", user_agent="Mozilla/5.0 (secret-ua)")


def _event(
    doc_type: LegalDocumentType,
    *,
    action: LegalAction = LegalAction.ACCEPT,
    version: date = DEFAULT_LEGAL_VERSION,
) -> LegalAcceptance:
    return LegalAcceptance(
        id=uuid4(),
        user_id=uuid4(),
        product_id="vex",
        doc_type=doc_type,
        action=action,
        version=version,
        content_sha256="0" * 64 if action == LegalAction.ACCEPT else None,
        source=LegalAcceptanceSource.SIGNUP,
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
    )


def _service(
    latest: dict[LegalDocumentType, LegalAcceptance] | None = None,
    **registry_kwargs: Any,
) -> tuple[LegalAcceptanceService, MagicMock, AsyncMock]:
    repo = MagicMock(spec=LegalAcceptanceRepository)
    repo.latest_per_type = AsyncMock(return_value=latest or {})
    repo.is_user_active = AsyncMock(return_value=True)
    session = AsyncMock()
    registry = registry_kwargs.get("registry") or make_legal_registry()
    return (
        LegalAcceptanceService(registry=registry, repository=repo, session=session),
        repo,
        session,
    )


class TestValidateSubmission:
    """C6 (logic half) — exact required set at current version + sha."""

    def test_valid_submission_returns_current_documents(self) -> None:
        service, _, _ = _service()
        docs = service.validate_submission(VEX_CONFIG, accept_all_current(), today=TODAY)
        assert [d.doc_type for d in docs] == sorted(VEX_CONFIG.required_legal_documents)

    def test_missing_type_is_incomplete(self) -> None:
        service, _, _ = _service()
        submission = [d for d in accept_all_current() if d.doc_type != LegalDocumentType.PRIVACY]
        with pytest.raises(LegalSubmissionIncompleteError) as exc_info:
            service.validate_submission(VEX_CONFIG, submission, today=TODAY)
        assert exc_info.value.detail == {
            "missing": ["privacy"],
            "unexpected": [],
            "duplicated": [],
        }

    def test_extra_type_is_incomplete(self) -> None:
        service, _, _ = _service()
        terms_only = replace(
            VEX_CONFIG, required_legal_documents=frozenset({LegalDocumentType.TERMS})
        )
        with pytest.raises(LegalSubmissionIncompleteError) as exc_info:
            service.validate_submission(terms_only, accept_all_current(), today=TODAY)
        assert exc_info.value.detail["unexpected"] == ["privacy", "sensitive_data_consent"]

    def test_duplicate_type_is_incomplete(self) -> None:
        service, _, _ = _service()
        submission = accept_all_current()
        with pytest.raises(LegalSubmissionIncompleteError) as exc_info:
            service.validate_submission(VEX_CONFIG, [*submission, submission[0]], today=TODAY)
        assert exc_info.value.detail["duplicated"] == [submission[0].doc_type.value]

    def test_empty_required_set_accepts_empty_submission(self) -> None:
        service, _, _ = _service()
        assert service.validate_submission(SYNTHARA_CONFIG, [], today=TODAY) == ()

    def test_empty_required_set_rejects_any_submission(self) -> None:
        service, _, _ = _service()
        with pytest.raises(LegalSubmissionIncompleteError):
            service.validate_submission(SYNTHARA_CONFIG, accept_all_current(), today=TODAY)

    def test_wrong_version_is_stale_with_current_detail(self) -> None:
        service, _, _ = _service()
        submission = [
            msgspec.structs.replace(d, version=date(2019, 1, 1))
            if d.doc_type == LegalDocumentType.TERMS
            else d
            for d in accept_all_current()
        ]
        with pytest.raises(LegalVersionStaleError) as exc_info:
            service.validate_submission(VEX_CONFIG, submission, today=TODAY)
        current = exc_info.value.detail["current"]
        assert {c["doc_type"] for c in current} == {t.value for t in LegalDocumentType}
        assert all(c["version"] == "2020-01-01" and len(c["sha256"]) == 64 for c in current)

    def test_wrong_sha_is_stale(self) -> None:
        service, _, _ = _service()
        submission = [msgspec.structs.replace(d, sha256="f" * 64) for d in accept_all_current()]
        with pytest.raises(LegalVersionStaleError):
            service.validate_submission(VEX_CONFIG, submission, today=TODAY)

    def test_wire_format_rejects_malformed_sha(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"doc_type": "terms", "version": "2026-10-01", "sha256": "ABC"}',
                type=AcceptedDocument,
            )


class TestRecordAcceptances:
    async def test_inserts_one_accept_row_per_document(self) -> None:
        service, repo, session = _service()
        docs = service.validate_submission(VEX_CONFIG, accept_all_current(), today=TODAY)
        user_id = uuid4()

        inserted = await service.record_acceptances(
            user_id=user_id,
            product=VEX_CONFIG,
            documents=docs,
            source=LegalAcceptanceSource.SIGNUP,
            context=CONTEXT,
        )

        assert inserted == 3
        rows: list[LegalAcceptance] = repo.add_many.call_args.args[0]
        assert {r.doc_type for r in rows} == set(LegalDocumentType)
        assert all(r.action == LegalAction.ACCEPT for r in rows)
        assert all(r.user_id == user_id and r.product_id == "vex" for r in rows)
        assert {r.content_sha256 for r in rows} == {d.sha256 for d in docs}
        assert all(r.ip_address == CONTEXT.ip_address for r in rows)
        assert all(r.user_agent == CONTEXT.user_agent for r in rows)
        session.flush.assert_awaited_once()
        assert repo.mock_calls[:3] == [
            call.lock_user_ledger(user_id=user_id),
            call.is_user_active(user_id=user_id),
            call.latest_per_type(user_id=user_id, product_id="vex"),
        ]

    async def test_inactive_user_is_rejected_after_lock(self) -> None:
        """F2 — a closure committed while we waited on the lock blocks the ACCEPT."""
        service, repo, session = _service()
        repo.is_user_active.return_value = False
        docs = service.validate_submission(VEX_CONFIG, accept_all_current(), today=TODAY)
        user_id = uuid4()

        with pytest.raises(LegalAccountInactiveError) as exc_info:
            await service.record_acceptances(
                user_id=user_id,
                product=VEX_CONFIG,
                documents=docs,
                source=LegalAcceptanceSource.REACCEPT,
                context=CONTEXT,
            )

        assert (exc_info.value.user_id, exc_info.value.product_id) == (user_id, "vex")
        assert repo.mock_calls == [
            call.lock_user_ledger(user_id=user_id),
            call.is_user_active(user_id=user_id),
        ]
        session.flush.assert_not_awaited()

    async def test_idempotent_resubmission_inserts_nothing(self) -> None:
        """C12 — already-accepted current versions insert 0 rows."""
        latest = {t: _event(t) for t in LegalDocumentType}
        service, repo, session = _service(latest)
        docs = service.validate_submission(VEX_CONFIG, accept_all_current(), today=TODAY)

        inserted = await service.record_acceptances(
            user_id=uuid4(),
            product=VEX_CONFIG,
            documents=docs,
            source=LegalAcceptanceSource.REACCEPT,
            context=CONTEXT,
        )

        assert inserted == 0
        repo.add_many.assert_not_called()
        session.flush.assert_not_awaited()

    async def test_withdrawn_type_is_reinserted(self) -> None:
        latest = {t: _event(t) for t in LegalDocumentType}
        latest[LegalDocumentType.SENSITIVE_DATA_CONSENT] = _event(
            LegalDocumentType.SENSITIVE_DATA_CONSENT, action=LegalAction.WITHDRAW
        )
        service, repo, _ = _service(latest)
        docs = service.validate_submission(VEX_CONFIG, accept_all_current(), today=TODAY)

        inserted = await service.record_acceptances(
            user_id=uuid4(),
            product=VEX_CONFIG,
            documents=docs,
            source=LegalAcceptanceSource.REACCEPT,
            context=CONTEXT,
        )

        assert inserted == 1
        (row,) = repo.add_many.call_args.args[0]
        assert row.doc_type == LegalDocumentType.SENSITIVE_DATA_CONSENT

    async def test_truncates_long_user_agent_and_ip(self) -> None:
        service, repo, _ = _service()
        docs = service.validate_submission(VEX_CONFIG, accept_all_current(), today=TODAY)
        await service.record_acceptances(
            user_id=uuid4(),
            product=VEX_CONFIG,
            documents=docs,
            source=LegalAcceptanceSource.SIGNUP,
            context=RequestContext(ip_address="1" * 100, user_agent="u" * 2000),
        )
        rows: list[LegalAcceptance] = repo.add_many.call_args.args[0]
        assert all(len(r.user_agent or "") == 512 for r in rows)
        assert all(len(r.ip_address or "") == 64 for r in rows)

    async def test_log_never_contains_ip_or_user_agent(self) -> None:
        """C17 — legal.acceptance.recorded carries ids and types only."""
        service, _, _ = _service()
        docs = service.validate_submission(VEX_CONFIG, accept_all_current(), today=TODAY)
        user_id = uuid4()

        with structlog.testing.capture_logs() as logs:
            await service.record_acceptances(
                user_id=user_id,
                product=VEX_CONFIG,
                documents=docs,
                source=LegalAcceptanceSource.SIGNUP,
                context=CONTEXT,
            )

        (event,) = [e for e in logs if e["event"] == "legal.acceptance.recorded"]
        assert event["user_id"] == str(user_id)
        assert event["product_id"] == "vex"
        assert sorted(event["doc_types"]) == sorted(LegalDocumentType)
        assert event["source"] == "signup"
        rendered = repr(event)
        assert CONTEXT.ip_address is not None
        assert CONTEXT.user_agent is not None
        assert CONTEXT.ip_address not in rendered
        assert CONTEXT.user_agent not in rendered
        assert "ip" not in {k.lower() for k in event}
        assert all("agent" not in k.lower() for k in event)


class TestConsentWithdrawal:
    async def test_vex_inserts_one_withdraw_row(self) -> None:
        service, repo, session = _service()
        await service.record_consent_withdrawal(
            user_id=uuid4(), product=VEX_CONFIG, context=CONTEXT, today=TODAY
        )
        (row,) = repo.add_many.call_args.args[0]
        assert row.doc_type == LegalDocumentType.SENSITIVE_DATA_CONSENT
        assert row.action == LegalAction.WITHDRAW
        assert row.content_sha256 is None
        assert row.source == LegalAcceptanceSource.ACCOUNT_CLOSURE
        assert row.version == DEFAULT_LEGAL_VERSION
        session.flush.assert_awaited_once()

    async def test_synthara_is_noop(self) -> None:
        service, repo, _ = _service()
        await service.record_consent_withdrawal(
            user_id=uuid4(), product=SYNTHARA_CONFIG, context=CONTEXT, today=TODAY
        )
        repo.add_many.assert_not_called()

    async def test_withdrawal_uses_passed_today(self) -> None:
        old = make_legal_document(LegalDocumentType.SENSITIVE_DATA_CONSENT, date(2026, 1, 1))
        current = make_legal_document(LegalDocumentType.SENSITIVE_DATA_CONSENT, TODAY)
        registry = make_legal_registry(
            make_legal_document(LegalDocumentType.TERMS),
            make_legal_document(LegalDocumentType.PRIVACY),
            old,
            current,
        )
        service, repo, _ = _service(registry=registry)

        await service.record_consent_withdrawal(
            user_id=uuid4(), product=VEX_CONFIG, context=CONTEXT, today=date(2026, 5, 1)
        )

        (row,) = repo.add_many.call_args.args[0]
        assert row.version == old.version

    async def test_withdrawal_locks_before_its_ledger_read(self) -> None:
        service, repo, _ = _service()
        user_id = uuid4()

        await service.record_consent_withdrawal(
            user_id=user_id, product=VEX_CONFIG, context=CONTEXT, today=TODAY
        )

        assert repo.mock_calls[:1] == [call.lock_user_ledger(user_id=user_id)]
        # Closure deactivates the user in the same transaction: never gate on is_active.
        repo.is_user_active.assert_not_called()


class TestSatisfiedDigestAndStatus:
    async def test_all_accepted_yields_required_digest(self) -> None:
        service, _, _ = _service({t: _event(t) for t in LegalDocumentType})
        digest = await service.satisfied_digest(user_id=uuid4(), product=VEX_CONFIG, today=TODAY)
        assert digest is not None
        assert digest == service.registry.required_digest(VEX_CONFIG, today=TODAY)

    async def test_missing_type_yields_none(self) -> None:
        latest = {t: _event(t) for t in LegalDocumentType if t != LegalDocumentType.TERMS}
        service, _, _ = _service(latest)
        assert (
            await service.satisfied_digest(user_id=uuid4(), product=VEX_CONFIG, today=TODAY) is None
        )

    async def test_withdrawn_consent_yields_none(self) -> None:
        latest = {t: _event(t) for t in LegalDocumentType}
        latest[LegalDocumentType.SENSITIVE_DATA_CONSENT] = _event(
            LegalDocumentType.SENSITIVE_DATA_CONSENT, action=LegalAction.WITHDRAW
        )
        service, _, _ = _service(latest)
        assert (
            await service.satisfied_digest(user_id=uuid4(), product=VEX_CONFIG, today=TODAY) is None
        )

    async def test_old_version_yields_none_after_flagged_update(self) -> None:
        v2 = date(2026, 9, 1)
        registry = make_legal_registry(
            *(make_legal_document(t) for t in LegalDocumentType),
            make_legal_document(LegalDocumentType.TERMS, v2),
        )
        service, _, _ = _service({t: _event(t) for t in LegalDocumentType}, registry=registry)
        assert (
            await service.satisfied_digest(user_id=uuid4(), product=VEX_CONFIG, today=TODAY) is None
        )
        status = await service.status(user_id=uuid4(), product=VEX_CONFIG, today=TODAY)
        terms = next(d for d in status.documents if d.doc_type == LegalDocumentType.TERMS)
        assert terms.required_version == v2
        assert terms.current_version == v2
        assert terms.accepted_version == DEFAULT_LEGAL_VERSION
        assert terms.satisfied is False
        assert status.all_satisfied is False

    async def test_empty_required_set_skips_db(self) -> None:
        service, repo, _ = _service()
        assert (
            await service.satisfied_digest(user_id=uuid4(), product=SYNTHARA_CONFIG, today=TODAY)
            is None
        )
        status = await service.status(user_id=uuid4(), product=SYNTHARA_CONFIG, today=TODAY)
        assert status.documents == ()
        assert status.all_satisfied is True
        repo.latest_per_type.assert_not_awaited()

    async def test_status_all_satisfied(self) -> None:
        service, _, _ = _service({t: _event(t) for t in LegalDocumentType})
        status = await service.status(user_id=uuid4(), product=VEX_CONFIG, today=TODAY)
        assert status.all_satisfied is True
        assert all(d.accepted_at is not None for d in status.documents)


def test_accepted_helper_round_trips() -> None:
    doc = make_legal_document(LegalDocumentType.TERMS, product=Product.VEX)
    assert accepted(doc).sha256 == doc.sha256


class TestRepositoryIsAppendOnly:
    """C14 (unit half) — no update/delete surface on the repository."""

    def test_public_surface(self) -> None:
        public = {n for n in dir(LegalAcceptanceRepository) if not n.startswith("_")}
        assert public == {"add_many", "is_user_active", "latest_per_type", "lock_user_ledger"}
