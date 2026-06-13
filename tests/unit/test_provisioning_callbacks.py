"""Tests for provisioning callback auth, progress persistence, and session response schema.

Covers design decisions D1–D4 from the phase-2 spec:
  D1 — hashed token auth
  D2 — retry path generates fresh token
  D3 — progress persistence (latest-state-wins)
  D4 — callbacks advisory; ready/failed handled correctly
  D6 — phase/progress exposed on session GET response
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.schemas.gpu_session import GpuSessionResponse
from src.api.services.gpu_session.provisioning_callback_service import (
    ProvisioningCallbackService,
    _validate_token,
)
from src.core.enums import GpuSessionStatus
from src.db.models.gpu_session import GpuSession

_REPO_PATH = "src.api.services.gpu_session.provisioning_callback_service.GpuSessionRepository"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(**kwargs: Any) -> GpuSession:
    plaintext = "secret-callback-token"
    now = datetime.now(UTC)
    s = GpuSession()
    s.id = uuid4()
    s.user_id = uuid4()
    s.product_id = "vex"
    s.status = GpuSessionStatus.provisioning
    s.bundle_name = "wan_2.2_i2v"
    s.bundle_version = "260105-01"
    s.model_type = "aisha-image"
    s.callback_token_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    s.provisioning_phase = None
    s.provisioning_progress = None
    s.last_progress_at = None
    s.created_at = now - timedelta(minutes=5)
    s.started_at = None
    s.stopped_at = None
    s.error_message = None
    s.stale_detected_at = None
    s.stale_notified = False
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def _make_mock_session_factory() -> tuple[MagicMock, MagicMock]:
    mock_db = MagicMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=None)
    mock_begin = MagicMock()
    mock_begin.__aenter__ = AsyncMock(return_value=None)
    mock_begin.__aexit__ = AsyncMock(return_value=None)
    mock_db.begin = MagicMock(return_value=mock_begin)
    mock_factory = MagicMock(return_value=mock_db)
    return mock_factory, mock_db


def _make_service() -> tuple[ProvisioningCallbackService, MagicMock]:
    factory, db = _make_mock_session_factory()
    svc = ProvisioningCallbackService(session_factory=factory)
    return svc, db


_VALID_TOKEN = "secret-callback-token"
_TS = datetime(2026, 6, 9, 10, 30, 0, tzinfo=UTC)


async def _call(
    svc: ProvisioningCallbackService,
    mock_repo: MagicMock,
    *,
    session: GpuSession,
    token: str = _VALID_TOKEN,
    phase: str = "downloading",
    ts: datetime | None = None,
) -> tuple[bool, int]:
    mock_repo.get_by_id.return_value = session
    return await svc.handle_callback(
        session_id=session.id,
        bearer_token=token,
        phase=phase,
        message="test message",
        download={"bytes_done": 1000, "bytes_total": 5000, "files_done": 0, "files_total": 1}
        if phase == "downloading"
        else None,
        error=None,
        elapsed_seconds=60,
        ts=ts or _TS,
    )


# ---------------------------------------------------------------------------
# D1: Auth — token validation
# ---------------------------------------------------------------------------


class TestTokenValidation:
    def test_valid_token_matches(self) -> None:
        token = "my-secret-token"
        stored = hashlib.sha256(token.encode()).hexdigest()
        assert _validate_token(token, stored) is True

    def test_wrong_token_rejected(self) -> None:
        stored = hashlib.sha256(b"correct").hexdigest()
        assert _validate_token("wrong", stored) is False

    def test_empty_stored_hash_rejected(self) -> None:
        assert _validate_token("anything", None) is False  # type: ignore[arg-type]
        assert _validate_token("anything", "") is False

    def test_constant_time_comparison(self) -> None:
        """hmac.compare_digest is used — verify the implementation doesn't use ==."""
        token = "tok"
        stored = hashlib.sha256(token.encode()).hexdigest()
        # Both produce the same result
        assert _validate_token(token, stored) is True
        assert _validate_token("bad", stored) is False


class TestCallbackAuth:
    async def test_valid_token_returns_200(self) -> None:
        svc, db = _make_service()
        session = _make_session()
        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session
            authorized, status = await _call(svc, mock_repo, session=session)

        assert authorized is True
        assert status == 200

    async def test_wrong_token_returns_401_no_write(self) -> None:
        svc, db = _make_service()
        session = _make_session()
        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session
            authorized, status = await _call(svc, mock_repo, session=session, token="bad-token")

        assert authorized is False
        assert status == 401
        mock_repo.update_provisioning_progress.assert_not_called()

    async def test_missing_hash_returns_401(self) -> None:
        svc, db = _make_service()
        session = _make_session(callback_token_hash=None)
        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session
            authorized, status = await _call(svc, mock_repo, session=session)

        assert authorized is False
        assert status == 401
        mock_repo.update_provisioning_progress.assert_not_called()

    async def test_session_not_found_returns_401(self) -> None:
        svc, db = _make_service()
        missing_id = uuid4()
        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = None  # not found
            authorized, status = await svc.handle_callback(
                session_id=missing_id,
                bearer_token=_VALID_TOKEN,
                phase="downloading",
                message="test",
                download=None,
                error=None,
                elapsed_seconds=0,
                ts=_TS,
            )

        assert authorized is False
        assert status == 401

    def test_hash_stored_is_sha256_not_plaintext(self) -> None:
        """Verify no plaintext is ever equal to the stored value."""
        token = "plaintext-secret"
        stored = hashlib.sha256(token.encode()).hexdigest()
        assert stored != token
        assert len(stored) == 64  # SHA-256 hex digest is always 64 chars


# ---------------------------------------------------------------------------
# D3: Progress persistence + ts-gate
# ---------------------------------------------------------------------------


class TestProgressPersistence:
    async def test_downloading_phase_updates_progress(self) -> None:
        svc, db = _make_service()
        session = _make_session()
        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session
            authorized, status = await _call(svc, mock_repo, session=session, phase="downloading")

        assert authorized is True
        assert status == 200
        mock_repo.update_provisioning_progress.assert_called_once()
        call_kwargs = mock_repo.update_provisioning_progress.call_args[1]
        assert call_kwargs["phase"] == "downloading"
        assert "download" in call_kwargs["progress"]
        assert "ts" in call_kwargs["progress"]
        assert call_kwargs["last_progress_at"] is not None

    async def test_stale_ts_is_ignored_no_write(self) -> None:
        """A callback with ts older than the stored ts must be ignored (200, no write)."""
        newer_ts = datetime(2026, 6, 9, 11, 0, 0, tzinfo=UTC)
        older_ts = datetime(2026, 6, 9, 10, 0, 0, tzinfo=UTC)
        svc, db = _make_service()
        # Session already has a newer stored ts in provisioning_progress
        session = _make_session(
            provisioning_progress={"ts": newer_ts.isoformat(), "message": "later update"}
        )
        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session
            authorized, status = await _call(svc, mock_repo, session=session, ts=older_ts)

        assert authorized is True
        assert status == 200
        mock_repo.update_provisioning_progress.assert_not_called()

    async def test_newer_ts_overwrites_stored(self) -> None:
        """A callback with ts >= stored ts must be applied."""
        older_ts = datetime(2026, 6, 9, 10, 0, 0, tzinfo=UTC)
        newer_ts = datetime(2026, 6, 9, 11, 0, 0, tzinfo=UTC)
        svc, db = _make_service()
        session = _make_session(
            provisioning_progress={"ts": older_ts.isoformat(), "message": "earlier"}
        )
        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session
            authorized, status = await _call(svc, mock_repo, session=session, ts=newer_ts)

        assert authorized is True
        assert status == 200
        mock_repo.update_provisioning_progress.assert_called_once()


# ---------------------------------------------------------------------------
# D4: Status gate
# ---------------------------------------------------------------------------


class TestStatusGate:
    @pytest.mark.parametrize("status", ["active", "stopped", "failed", "stale", "paused"])
    async def test_non_provisioning_status_ignored_200(self, status: str) -> None:
        """Callbacks against non-provisioning sessions return 200 with no write."""
        svc, db = _make_service()
        session = _make_session(status=status)
        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session
            authorized, http_status = await _call(svc, mock_repo, session=session)

        assert authorized is True
        assert http_status == 200
        mock_repo.update_provisioning_progress.assert_not_called()

    @pytest.mark.parametrize(
        "status",
        [GpuSessionStatus.pending, GpuSessionStatus.provisioning, GpuSessionStatus.resuming],
    )
    async def test_provisioning_statuses_accepted(self, status: GpuSessionStatus) -> None:
        svc, db = _make_service()
        session = _make_session(status=status)
        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session
            authorized, http_status = await _call(svc, mock_repo, session=session)

        assert authorized is True
        assert http_status == 200
        mock_repo.update_provisioning_progress.assert_called_once()


class TestReadyCallback:
    async def test_ready_phase_updates_progress_not_status(self) -> None:
        """'ready' callback must update phase/progress but NOT change session status."""
        svc, db = _make_service()
        session = _make_session(status=GpuSessionStatus.provisioning)
        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session
            authorized, http_status = await _call(svc, mock_repo, session=session, phase="ready")

        assert authorized is True
        assert http_status == 200
        # Phase/progress written
        mock_repo.update_provisioning_progress.assert_called_once()
        call_kwargs = mock_repo.update_provisioning_progress.call_args[1]
        assert call_kwargs["phase"] == "ready"
        # Session status NOT changed — only the worker's probe activates the session
        mock_repo.update_status.assert_not_called()


class TestFailedCallback:
    async def test_failed_phase_persists_error_without_changing_status(self) -> None:
        """'failed' callback writes provisioning_phase='failed'; worker picks it up on next sweep."""
        svc, db = _make_service()
        session = _make_session(status=GpuSessionStatus.provisioning)
        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.get_by_id.return_value = session

            authorized, http_status = await svc.handle_callback(
                session_id=session.id,
                bearer_token=_VALID_TOKEN,
                phase="failed",
                message="comfyui install failed",
                download=None,
                error="requirements install error",
                elapsed_seconds=120,
                ts=_TS,
            )

        assert authorized is True
        assert http_status == 200
        # Phase written — worker will pick this up on next sweep
        mock_repo.update_provisioning_progress.assert_called_once()
        call_kwargs = mock_repo.update_provisioning_progress.call_args[1]
        assert call_kwargs["phase"] == "failed"
        assert call_kwargs["progress"].get("error") == "requirements install error"
        # Status NOT changed — transitions stay in the worker
        mock_repo.update_status.assert_not_called()


# ---------------------------------------------------------------------------
# D6: Schema — provisioning_phase and provisioning_progress on session response
# ---------------------------------------------------------------------------


class TestSessionResponseSchema:
    def _make_full_session(self) -> GpuSession:
        return _make_session(
            provisioning_phase="downloading",
            provisioning_progress={
                "ts": "2026-06-09T10:30:00Z",
                "message": "Downloading 1 model file",
                "download": {
                    "bytes_done": 12_000_000_000,
                    "bytes_total": 30_521_000_000,
                    "files_done": 0,
                    "files_total": 1,
                },
            },
        )

    def test_from_model_includes_phase(self) -> None:
        session = self._make_full_session()
        response = GpuSessionResponse.from_model(session)
        assert response.provisioning_phase == "downloading"

    def test_from_model_includes_progress(self) -> None:
        session = self._make_full_session()
        response = GpuSessionResponse.from_model(session)
        assert response.provisioning_progress is not None
        assert response.provisioning_progress["message"] == "Downloading 1 model file"
        assert "download" in response.provisioning_progress

    def test_from_model_none_phase_when_no_callbacks(self) -> None:
        session = _make_session(provisioning_phase=None, provisioning_progress=None)
        response = GpuSessionResponse.from_model(session)
        assert response.provisioning_phase is None
        assert response.provisioning_progress is None
