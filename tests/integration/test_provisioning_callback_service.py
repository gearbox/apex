"""Integration tests for ProvisioningCallbackService with a real AsyncSession.

These tests catch the SQLAlchemy autobegin conflict that mock-based unit tests
cannot reproduce: get_by_id() autobegins a transaction, and a subsequent
db.begin() on the same session raises InvalidRequestError.

The fix (async with self._session_factory() as db, db.begin():) is verified here
by running handle_callback against a real AsyncSession sharing the test connection.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.api.schemas.gpu_session import DownloadProgressBody
from src.api.services.gpu_session.provisioning_callback_service import (
    ProvisioningCallbackService,
)
from src.core.enums import GpuSessionStatus
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User

# ---------------------------------------------------------------------------
# Fixture: shared connection + session factory on the same connection
# ---------------------------------------------------------------------------


SessionFactory = Callable[[], AsyncSession]


@pytest_asyncio.fixture
async def provisioning_harness(
    db_engine: AsyncEngine,
) -> AsyncGenerator[tuple[AsyncSession, SessionFactory]]:
    """Yield (setup_session, session_factory) sharing the same connection.

    setup_session is used to INSERT test rows (flushed but not committed).
    session_factory creates new AsyncSessions on the same connection so the
    service can see the flushed rows and its own begin()/flush() are isolated
    within the outer SAVEPOINT — all rolled back on teardown.
    """
    async with db_engine.connect() as conn:
        tx = await conn.begin()
        async with AsyncSession(bind=conn, expire_on_commit=False) as setup_session:
            sp = await conn.begin_nested()
            try:

                def session_factory() -> AsyncSession:
                    return AsyncSession(bind=conn, expire_on_commit=False)

                yield setup_session, session_factory
            finally:
                await setup_session.rollback()
                if sp.is_active:
                    await sp.rollback()
        await tx.rollback()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _token_hash(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()


async def _make_user(session: AsyncSession, *, tag: str = "") -> User:
    user = User(
        id=uuid4(),
        email=f"cb-integ-{tag or uuid4().hex[:8]}@example.com",
        password_hash="hashed",
        product_id="vex",
    )
    session.add(user)
    await session.flush()
    return user


async def _make_gpu_session(
    session: AsyncSession,
    user_id: UUID,
    *,
    token_plaintext: str,
    status: str = GpuSessionStatus.provisioning.value,
    provisioning_progress: dict[str, object] | None = None,
) -> GpuSession:
    gpu_session = GpuSession(
        id=uuid4(),
        user_id=user_id,
        product_id="vex",
        bundle_name="wan_2.2_i2v",
        model_type="aisha-image",
        status=status,
        callback_token_hash=_token_hash(token_plaintext),
        provisioning_progress=provisioning_progress,
    )
    session.add(gpu_session)
    await session.flush()
    return gpu_session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_handle_callback_no_autobegin_conflict(
    provisioning_harness: tuple[AsyncSession, SessionFactory],
) -> None:
    """Regression: handle_callback must not raise InvalidRequestError.

    Before the fix, get_by_id() autobegun a transaction on the session, and
    the subsequent db.begin() conflicted. This test runs against a real
    AsyncSession so SQLAlchemy's autobegin behaviour is exercised.
    """
    setup_session, session_factory = provisioning_harness

    token = "integration-test-callback-token"
    user = await _make_user(setup_session, tag="noconflict")
    gpu = await _make_gpu_session(setup_session, user.id, token_plaintext=token)

    svc = ProvisioningCallbackService(session_factory=session_factory)  # type: ignore[arg-type]
    ts = datetime(2026, 6, 16, 10, 0, 0, tzinfo=UTC)

    authorized, status_code = await svc.handle_callback(
        session_id=gpu.id,
        bearer_token=token,
        phase="downloading",
        message="fetching models",
        download=DownloadProgressBody(
            bytes_done=1_000, bytes_total=10_000, files_done=0, files_total=1
        ),
        error=None,
        elapsed_seconds=60,
        ts=ts,
    )

    assert authorized is True
    assert status_code == 200

    # Verify the row was updated by the service
    await setup_session.refresh(gpu)
    assert gpu.provisioning_phase == "downloading"
    assert gpu.last_progress_at is not None
    assert gpu.provisioning_progress is not None
    assert gpu.provisioning_progress["ts"] == ts.isoformat()
    assert gpu.provisioning_progress["message"] == "fetching models"
    download_blob = gpu.provisioning_progress.get("download")
    assert download_blob is not None
    assert download_blob["bytes_done"] == 1_000  # type: ignore[index]


async def test_handle_callback_stale_ts_returns_200_no_write(
    provisioning_harness: tuple[AsyncSession, SessionFactory],
) -> None:
    """Early return inside db.begin() block (stale-ts gate) leaves row unchanged.

    This exercises an early return that happens AFTER the transaction opens but
    BEFORE the write — confirming that early returns in the begin-first pattern
    are harmless (commit a read-only transaction).
    """
    setup_session, session_factory = provisioning_harness

    newer_ts = datetime(2026, 6, 16, 12, 0, 0, tzinfo=UTC)
    stale_ts = datetime(2026, 6, 16, 10, 0, 0, tzinfo=UTC)

    token = "integration-test-stale-token"
    user = await _make_user(setup_session, tag="stale")
    gpu = await _make_gpu_session(
        setup_session,
        user.id,
        token_plaintext=token,
        provisioning_progress={"ts": newer_ts.isoformat(), "message": "newer update"},
    )

    svc = ProvisioningCallbackService(session_factory=session_factory)  # type: ignore[arg-type]

    authorized, status_code = await svc.handle_callback(
        session_id=gpu.id,
        bearer_token=token,
        phase="downloading",
        message="stale callback",
        download=None,
        error=None,
        elapsed_seconds=30,
        ts=stale_ts,
    )

    assert authorized is True
    assert status_code == 200

    # Row must be unchanged — stale-ts gate fired the early return
    await setup_session.refresh(gpu)
    assert gpu.provisioning_progress is not None
    assert gpu.provisioning_progress["ts"] == newer_ts.isoformat()
    assert gpu.provisioning_progress["message"] == "newer update"
