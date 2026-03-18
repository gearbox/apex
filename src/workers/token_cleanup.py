"""Periodic background task that deletes expired auth tokens.

Runs as a simple asyncio loop; no external scheduler dependency.
Interval and batch behaviour configured via Settings.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from src.db.repositories.auth_tokens import AuthTokenRepository
from src.db.repositories.user import UserRepository
from src.db.session import DatabaseManager

logger = structlog.get_logger(__name__)


class TokenCleanupWorker:
    """Worker that periodically cleans up expired tokens."""

    def __init__(self, db_manager: DatabaseManager, interval: int) -> None:
        """Initialize the worker.

        Args:
            db_manager: Database manager for creating sessions.
            interval: Seconds between cleanup runs.
        """
        self._db_manager = db_manager
        self._interval = interval
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the worker loop in the background."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("token_cleanup_worker.started", interval_s=self._interval)

    async def stop(self) -> None:
        """Stop the worker loop."""
        if not self._running:
            return

        from contextlib import suppress

        self._running = False
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        logger.info("token_cleanup_worker.stopped")

    async def _run_loop(self) -> None:
        """Main periodic execution loop."""
        while self._running:
            await self._run_once()
            if self._running:
                await asyncio.sleep(self._interval)

    async def _run_once(self) -> None:
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
