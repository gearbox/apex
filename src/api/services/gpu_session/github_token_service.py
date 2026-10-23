"""Issue the GitHub credential to a GPU node, authenticated by its session callback token.

Nodes PULL a token whenever they need one — at bootstrap, on resume, and before each
registry sync — instead of keeping the one in their Vast env (fixed at creation, so it
would be dead after an hour of App-token lifetime). The token is the process-wide
``GitHubTokenProvider``'s: the shared App installation token (or the static PAT).

Read-only: no DB writes and no row lock. The same token is handed to every node of this
process; it is deliberately not revoked when a session stops (that would break the
others — the 1 h expiry bounds exposure).

SECURITY: never log the token (only ``mode``, ``expires_at`` and ids).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import structlog

from src.api.security.callback_token import validate_callback_token
from src.api.services.github.token_provider import GitHubTokenUnavailableError
from src.core.enums import GpuSessionStatus
from src.db.repositories.gpu_session import GpuSessionRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.github.token_provider import GitHubToken, GitHubTokenProvider

logger = structlog.get_logger(__name__)

# A node needs the token from the moment its instance exists until teardown begins.
# Paused/stopping/stopped/failed sessions have no node that should still be cloning.
GITHUB_TOKEN_ELIGIBLE_STATUSES: Final = frozenset(
    {
        GpuSessionStatus.pending,
        GpuSessionStatus.provisioning,
        GpuSessionStatus.active,
        GpuSessionStatus.stale,
        GpuSessionStatus.resuming,
    }
)


class GitHubTokenIssueOutcome(StrEnum):
    issued = "issued"
    unauthorized = "unauthorized"
    not_eligible = "not_eligible"
    unavailable = "unavailable"


@dataclass(frozen=True, slots=True)
class GitHubTokenIssueResult:
    """Outcome for the route to map to HTTP; ``token`` is set only for ``issued``."""

    outcome: GitHubTokenIssueOutcome
    token: GitHubToken | None = None


class GpuSessionGitHubTokenService:
    """Validates a node's callback token, then serves the shared GitHub credential."""

    def __init__(self, *, provider: GitHubTokenProvider | None) -> None:
        # None = no GitHub credential configured (auth mode `none`) -> always `unavailable`.
        self._provider = provider

    async def issue(
        self, *, session_id: UUID, bearer_token: str, db: AsyncSession
    ) -> GitHubTokenIssueResult:
        session = await GpuSessionRepository(db).get_by_id(session_id)
        if session is None:
            # Indistinguishable from a wrong token, so the endpoint never reveals which
            # session ids exist (same as the provisioning webhook).
            logger.warning(
                "gpu_session.github_token.rejected",
                session_id=str(session_id),
                reason="invalid_token",
            )
            return GitHubTokenIssueResult(GitHubTokenIssueOutcome.unauthorized)
        if not validate_callback_token(bearer_token, session.callback_token_hash):
            logger.warning(
                "gpu_session.github_token.rejected",
                session_id=str(session_id),
                reason="invalid_token",
            )
            return GitHubTokenIssueResult(GitHubTokenIssueOutcome.unauthorized)

        if session.status not in GITHUB_TOKEN_ELIGIBLE_STATUSES:
            logger.info(
                "gpu_session.github_token.not_eligible",
                session_id=str(session_id),
                status=str(session.status),
            )
            return GitHubTokenIssueResult(GitHubTokenIssueOutcome.not_eligible)

        if self._provider is None:
            logger.error(
                "gpu_session.github_token.unavailable",
                session_id=str(session_id),
                reason="no_provider",
            )
            return GitHubTokenIssueResult(GitHubTokenIssueOutcome.unavailable)
        try:
            token = await self._provider.get_token()
        except GitHubTokenUnavailableError as exc:
            logger.exception(
                "gpu_session.github_token.unavailable",
                session_id=str(session_id),
                mode=self._provider.mode.value,
                reason=str(exc),
            )
            return GitHubTokenIssueResult(GitHubTokenIssueOutcome.unavailable)

        logger.info(
            "gpu_session.github_token.issued",
            session_id=str(session_id),
            mode=self._provider.mode.value,
            expires_at=token.expires_at.isoformat() if token.expires_at else None,
        )
        return GitHubTokenIssueResult(GitHubTokenIssueOutcome.issued, token)
