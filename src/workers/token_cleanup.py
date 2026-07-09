"""Periodic background task that deletes expired auth tokens.

Interval and batch behaviour configured via Settings.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

from src.db.repositories.auth_tokens import AuthTokenRepository
from src.db.repositories.user import UserRepository
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from src.db.session import DatabaseManager

logger = structlog.get_logger(__name__)


class TokenCleanupWorker(PeriodicWorker):
    """Worker that periodically cleans up expired tokens."""

    def __init__(
        self, db_manager: DatabaseManager, interval: int, *, redis_enabled: bool = False
    ) -> None:
        """Initialize the worker.

        Args:
            db_manager: Database manager for creating sessions.
            interval: Seconds between cleanup runs.
            redis_enabled: Whether Redis is configured (enables the leader lease).
        """
        super().__init__(
            name="token_cleanup",
            interval_seconds=interval,
            jitter_seconds=5.0,
            redis_enabled=redis_enabled,
        )
        self._db_manager = db_manager

    async def run_once(self) -> None:
        """Execute a single token cleanup run."""
        start_time = time.perf_counter()
        refresh_count = 0
        email_count = 0
        password_count = 0

        try:
            async with self._db_manager.session() as session:
                user_repo = UserRepository(session)
                token_repo = AuthTokenRepository(session)

                # All cleanups happen within the single transaction provided by the session context manager
                refresh_count = await user_repo.cleanup_expired_tokens()
                email_count = await token_repo.cleanup_expired_email_verification_tokens()
                password_count = await token_repo.cleanup_expired_password_reset_tokens()

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                "token_cleanup",
                refresh=refresh_count,
                email_verification=email_count,
                password_reset=password_count,
                duration_ms=duration_ms,
            )
        except Exception as e:
            logger.exception("token_cleanup_worker.error", error=str(e))
