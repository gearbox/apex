"""Thin receiver service for GPU node provisioning callbacks.

The receiver is a pure writer: it validates auth, applies status/ts gates,
and persists the latest progress. All session state-machine transitions
remain in GpuProvisioningWorker — the worker picks up node-reported failures
and stalls on its next sweep.

SECURITY: never log tokens.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from src.api.schemas.gpu_session import DownloadProgressBody
from src.db.repositories.gpu_session import PROVISIONING_STATUSES, GpuSessionRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.db.models.gpu_session import GpuSession

logger = structlog.get_logger(__name__)

# Phases that carry download progress; others only update phase + message.
_DOWNLOAD_PHASES = frozenset({"downloading"})


def _validate_token(presented: str, stored_hash: str | None) -> bool:
    """Constant-time comparison of presented bearer token against stored SHA-256 hash."""
    if not stored_hash:
        return False
    presented_hash = hashlib.sha256(presented.encode()).hexdigest()
    return hmac.compare_digest(presented_hash, stored_hash)


class ProvisioningCallbackService:
    """Validates and persists provisioning callbacks from GPU nodes."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def handle_callback(
        self,
        *,
        session_id: UUID,
        bearer_token: str,
        phase: str,
        message: str,
        download: DownloadProgressBody | None,
        error: str | None,
        elapsed_seconds: int,
        ts: datetime,
    ) -> tuple[bool, int]:
        """Validate and persist one provisioning callback.

        Returns:
            (authorized, http_status) — authorized=False means 401; True always
            returns 200 regardless of whether the write was applied (status-gated
            and stale-ts callbacks return 200 with no write per the design spec).
        """
        async with self._session_factory() as db, db.begin():
            repo = GpuSessionRepository(db)
            session = await repo.get_by_id(session_id)

            if session is None:
                logger.warning(
                    "gpu_session.callback.rejected",
                    session_id=str(session_id),
                    reason="session_not_found",
                )
                return False, 401

            if not _validate_token(bearer_token, session.callback_token_hash):
                logger.warning(
                    "gpu_session.callback.rejected",
                    session_id=str(session_id),
                    reason="invalid_token",
                )
                return False, 401

            if session.status not in PROVISIONING_STATUSES:
                logger.info(
                    "gpu_session.callback.ignored_status",
                    session_id=str(session_id),
                    session_status=session.status,
                    phase=phase,
                )
                return True, 200

            # TS gate: ignore stale callbacks (out-of-order delivery from the node).
            stored_ts = _extract_stored_ts(session)
            if stored_ts is not None and ts < stored_ts:
                logger.debug(
                    "gpu_session.callback.stale_ts",
                    session_id=str(session_id),
                    callback_ts=ts.isoformat(),
                    stored_ts=stored_ts.isoformat(),
                )
                return True, 200

            # Build the progress blob stored in JSONB.
            progress: dict[str, object] = {
                "ts": ts.isoformat(),
                "message": message,
                "elapsed_seconds": elapsed_seconds,
            }
            if download is not None:
                if phase in _DOWNLOAD_PHASES:
                    progress["download"] = {
                        "bytes_done": download.bytes_done,
                        "bytes_total": download.bytes_total,
                        "files_done": download.files_done,
                        "files_total": download.files_total,
                    }
                else:
                    logger.debug(
                        "gpu_session.callback.unexpected_download",
                        session_id=str(session_id),
                        phase=phase,
                    )
            if error is not None:
                progress["error"] = error

            now = datetime.now(UTC)
            await repo.update_provisioning_progress(
                session_id,
                phase=phase,
                progress=progress,
                last_progress_at=now,
            )

        logger.info(
            "gpu_session.callback.received",
            session_id=str(session_id),
            phase=phase,
            elapsed_seconds=elapsed_seconds,
        )
        return True, 200


def _extract_stored_ts(session: GpuSession) -> datetime | None:
    """Pull the stored ts from provisioning_progress JSONB, or None if absent."""
    if not session.provisioning_progress:
        return None
    raw = session.provisioning_progress.get("ts")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None
