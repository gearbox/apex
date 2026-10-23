"""GitHub credential for the private ``aisha`` / ``ai-bundles`` repos.

One ``GitHubTokenProvider`` per process, chosen from ``Settings.github_auth_mode``:

- ``GitHubAppTokenProvider`` (mode ``app``) — mints a read-only installation token limited
  to the two repos, caches it in memory and re-mints it single-flight when under
  ``refresh_margin`` remains. Several apex processes each mint their own token; that is
  expected (a token per process per hour is trivial for GitHub).
- ``StaticGitHubTokenProvider`` (mode ``pat``) — the long-lived ``AI_BUNDLES_GITHUB_TOKEN``.

``build_github_token_provider`` picks between them (``none`` → ``None``).

Both serve apex's own fetches (bundle-index tarball, bootstrap script) and the node token
endpoint, so a node and apex always hold the same kind of credential.

SECURITY: never log, raise, or ``repr`` the App JWT, an installation token, or the private
key. ``GitHubToken.value`` is excluded from ``repr``; every error message here is a fixed
string plus a status code or an exception class name — never a response body.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

import httpx
import jwt
import structlog

from src.core.enums import GitHubAuthMode

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from src.core.config import Settings

logger = structlog.get_logger(__name__)

_GITHUB_API_BASE = "https://api.github.com"
_MINT_TIMEOUT_SECONDS = 10.0
# GitHub rejects an App JWT whose exp is more than 10 minutes after iat. Backdating iat by
# 60 s absorbs clock skew; exp = now + 540 s keeps the whole window at 10 minutes.
_JWT_IAT_BACKDATE_SECONDS = 60
_JWT_EXP_SECONDS = 540


@dataclass(frozen=True, slots=True)
class GitHubToken:
    """A GitHub credential. ``expires_at`` is ``None`` for a static PAT."""

    value: str = field(repr=False)
    expires_at: datetime | None


class GitHubTokenUnavailableError(Exception):
    """Minting failed (GitHub unreachable, non-201, malformed body). Message has no secrets."""


class GitHubTokenProvider(Protocol):
    """Source of the credential for the private repos."""

    @property
    def mode(self) -> GitHubAuthMode: ...

    async def get_token(self) -> GitHubToken:
        """Return a currently-valid token.

        Raises:
            GitHubTokenUnavailableError: a fresh token could not be obtained.
        """
        ...


class StaticGitHubTokenProvider:
    """Serves one fixed PAT (``pat`` mode)."""

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("token must be non-empty")
        self._token = GitHubToken(value=token, expires_at=None)

    @property
    def mode(self) -> GitHubAuthMode:
        return GitHubAuthMode.pat

    async def get_token(self) -> GitHubToken:
        return self._token


class GitHubAppTokenProvider:
    """Mints and caches a GitHub App installation token (``app`` mode)."""

    def __init__(
        self,
        *,
        app_id: int,
        installation_id: int,
        private_key_pem: bytes,
        repositories: Sequence[str],
        http_client: httpx.AsyncClient,
        refresh_margin: timedelta,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._app_id = app_id
        self._installation_id = installation_id
        self._private_key_pem = private_key_pem
        self._repositories = tuple(repositories)
        self._http = http_client
        self._refresh_margin = refresh_margin
        self._clock = clock
        self._cached: GitHubToken | None = None
        self._mint_lock = asyncio.Lock()

    @property
    def mode(self) -> GitHubAuthMode:
        return GitHubAuthMode.app

    async def get_token(self) -> GitHubToken:
        cached = self._cached
        if cached is not None and self._is_fresh(cached):
            return cached
        async with self._mint_lock:
            # Single-flight: callers queued on the lock find the winner's token here.
            cached = self._cached
            if cached is not None and self._is_fresh(cached):
                return cached
            token = await self._mint()
            self._cached = token
            return token

    def _is_fresh(self, token: GitHubToken) -> bool:
        return (
            token.expires_at is not None and token.expires_at - self._clock() > self._refresh_margin
        )

    def _app_jwt(self) -> str:
        now = int(self._clock().timestamp())
        claims: dict[str, Any] = {
            "iat": now - _JWT_IAT_BACKDATE_SECONDS,
            "exp": now + _JWT_EXP_SECONDS,
            "iss": str(self._app_id),
        }
        try:
            return jwt.encode(claims, self._private_key_pem, algorithm="RS256")
        except (jwt.PyJWTError, ValueError, TypeError) as exc:
            raise _mint_failed(exc.__class__.__name__) from None

    async def _mint(self) -> GitHubToken:
        url = f"{_GITHUB_API_BASE}/app/installations/{self._installation_id}/access_tokens"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._app_jwt()}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        body = {"repositories": list(self._repositories), "permissions": {"contents": "read"}}
        try:
            resp = await self._http.post(
                url, headers=headers, json=body, timeout=_MINT_TIMEOUT_SECONDS
            )
        except httpx.HTTPError as exc:
            logger.warning("github.app_token.mint_failed", error_class=exc.__class__.__name__)
            raise _mint_failed(exc.__class__.__name__) from None

        if resp.status_code != httpx.codes.CREATED:
            logger.warning("github.app_token.mint_failed", status_code=resp.status_code)
            raise _mint_failed(f"status {resp.status_code}")

        token = self._parse_token(resp)
        logger.info(
            "github.app_token.minted",
            expires_at=token.expires_at.isoformat() if token.expires_at else None,
            repositories=list(self._repositories),
        )
        return token

    @staticmethod
    def _parse_token(resp: httpx.Response) -> GitHubToken:
        try:
            payload = resp.json()
            value = payload["token"]
            # GitHub sends RFC 3339 with a trailing "Z".
            expires_at = datetime.fromisoformat(payload["expires_at"])
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("github.app_token.mint_failed", error_class=exc.__class__.__name__)
            raise _mint_failed(exc.__class__.__name__) from None
        if not isinstance(value, str) or not value or expires_at.tzinfo is None:
            logger.warning("github.app_token.mint_failed", error_class="MalformedResponse")
            raise _mint_failed("MalformedResponse")
        return GitHubToken(value=value, expires_at=expires_at.astimezone(UTC))


def _mint_failed(detail: str) -> GitHubTokenUnavailableError:
    return GitHubTokenUnavailableError(f"github app token mint failed: {detail}")


def build_github_token_provider(
    settings: Settings, *, http_client: httpx.AsyncClient
) -> GitHubTokenProvider | None:
    """Pick the GitHub credential provider for ``settings.github_auth_mode``.

    ``None`` means auth mode ``none`` (no credential configured): consumers then send no
    credential, and the session start / retry paths and the node token endpoint fail
    closed. Call once per process — a per-consumer provider would defeat the token cache.

    When both the App and a PAT are configured the App wins; a ``pat_ignored`` warning flags
    the leftover PAT (no values logged).
    """
    match settings.github_auth_mode:
        case GitHubAuthMode.app:
            private_key_pem = settings.github_app_private_key_pem
            if (
                settings.github_app_id is None
                or settings.github_app_installation_id is None
                or private_key_pem is None
            ):
                # Unreachable: Settings validates the trio as all-or-none at boot.
                raise ValueError("GitHub App settings are incomplete")
            if settings.github_content_token:
                logger.warning("github_token_provider.pat_ignored", reason="github_app_configured")
            return GitHubAppTokenProvider(
                app_id=settings.github_app_id,
                installation_id=settings.github_app_installation_id,
                private_key_pem=private_key_pem,
                repositories=settings.github_repositories,
                http_client=http_client,
                refresh_margin=timedelta(seconds=settings.github_app_token_refresh_margin_seconds),
            )
        case GitHubAuthMode.pat:
            return StaticGitHubTokenProvider(settings.github_content_token)
        case GitHubAuthMode.none:
            return None
