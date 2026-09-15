"""Bootstrap-script delivery for Vast.ai GPU nodes (D1-D6).

Resolves the Aisha ComfyUI bootstrap script from the private gearbox/aisha repo, using
the server-side GitHub PAT, and caches successful fetches in Redis so the GitHub
Contents API is hit at most once per (variant, ref) per cache TTL. Failures are never
cached (D5) — a fixed ref would otherwise stay broken for the whole TTL.

SECURITY: repo/path are never taken from the request — see
src.core.constants.SCRIPT_VARIANT_SOURCES. Callers must never log the callback token
or a full script/webhook URL with its query string.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
import structlog

from src.api.security.callback_token import validate_callback_token
from src.core.constants import PROVISIONING_REF_PATTERN, SCRIPT_VARIANT_SOURCES
from src.core.enums import ScriptServeOutcome, ScriptVariant
from src.db.repositories.gpu_session import GpuSessionRepository

if TYPE_CHECKING:
    from uuid import UUID

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.core.config import Settings

logger = structlog.get_logger(__name__)

_FETCH_TIMEOUT_SECONDS = 10.0
_GITHUB_API_BASE = "https://api.github.com"


class ProvisioningScriptError(Exception):
    """Base error for bootstrap-script resolution failures."""


class ProvisioningScriptRefNotFoundError(ProvisioningScriptError):
    """The (variant, ref) pair does not exist in the source repo (upstream 404)."""


class ProvisioningScriptUnavailableError(ProvisioningScriptError):
    """Upstream GitHub error (5xx, or 403 rate-limited) — transient, retry later."""


@dataclass(frozen=True, slots=True)
class ResolvedScript:
    """A fetched bootstrap script body plus its content hash."""

    content: str
    sha256: str
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class ScriptServeResult:
    """Outcome of a full request-level resolve, for the route to map to HTTP."""

    outcome: ScriptServeOutcome
    script: ResolvedScript | None = None


def _cache_key(repo: str, ref: str, path: str) -> str:
    path_hash = hashlib.sha256(path.encode()).hexdigest()
    return f"provisioning:script:{repo}:{ref}:{path_hash}"


class ProvisioningScriptService:
    """Resolves the Vast.ai bootstrap script from a private GitHub repo, with caching."""

    def __init__(
        self,
        *,
        http: httpx.AsyncClient,
        redis: Redis | None,
        settings: Settings,
    ) -> None:
        self._http = http
        self._redis = redis
        self._settings = settings

    async def resolve(self, variant: ScriptVariant, ref: str) -> ResolvedScript:
        """Return the script body + sha256, from cache when available.

        Raises:
            ProvisioningScriptRefNotFoundError: upstream 404, or `variant` has no
                configured source (the `base` reserved slot today).
            ProvisioningScriptUnavailableError: upstream 5xx or rate-limited.
        """
        immutable_ref = bool(PROVISIONING_REF_PATTERN.fullmatch(ref))
        if not self._ref_allowed(ref):
            raise ProvisioningScriptRefNotFoundError(f"Ref {ref!r} is not allowed")

        source = SCRIPT_VARIANT_SOURCES.get(variant)
        if source is None:
            raise ProvisioningScriptRefNotFoundError(
                f"No source configured for variant {variant!r}"
            )
        repo, path = source

        cache_key = _cache_key(repo, ref, path)
        if self._redis is not None:
            cached = await self._read_cache(cache_key)
            if cached is not None:
                return cached

        resolved = await self._fetch_from_github(repo=repo, path=path, ref=ref)

        if self._redis is not None:
            await self._write_cache(cache_key, resolved, ttl=self._cache_ttl_for(immutable_ref))

        return resolved

    async def serve_for_session(
        self,
        *,
        db: AsyncSession,
        session_id: UUID | None,
        token: str | None,
        variant: str,
        ref: str,
    ) -> ScriptServeResult:
        """Full request-level resolve: validates variant/ref/session/token, then resolve().

        Kept on the service (rather than in the route) so the route stays a thin
        HTTP-mapping layer and this logic is unit-testable without a live DB/HTTP
        stack — mirrors OperationEventService.handle_event's shape.
        """
        if variant not in {member.value for member in ScriptVariant}:
            return ScriptServeResult(outcome=ScriptServeOutcome.bad_request)
        variant_enum = ScriptVariant(variant)

        if not self._ref_allowed(ref):
            return ScriptServeResult(outcome=ScriptServeOutcome.bad_request)

        if session_id is None or not token:
            logger.warning("provisioning.script.rejected", reason="missing_session_or_token")
            return ScriptServeResult(outcome=ScriptServeOutcome.unauthorized)

        session_row = await GpuSessionRepository(db).get_by_id(session_id)
        if session_row is None or not validate_callback_token(
            token, session_row.callback_token_hash
        ):
            logger.warning(
                "provisioning.script.rejected", session_id=str(session_id), reason="invalid_token"
            )
            return ScriptServeResult(outcome=ScriptServeOutcome.unauthorized)

        # S8: the request-scoped DB session isn't needed past the token check above —
        # release its pooled connection before resolve()'s outbound GitHub fetch, which
        # takes up to _FETCH_TIMEOUT_SECONDS (10s) on a cache miss. Committing (rather
        # than closing) keeps `db` itself usable for the caller's later teardown without
        # relying on close-then-reuse semantics; there's nothing to write here, so the
        # commit is a no-op beyond returning the connection to the pool. Under a
        # simultaneous multi-node boot this avoids tying up pool connections for the
        # full fetch duration for no reason.
        await db.commit()

        try:
            resolved = await self.resolve(variant_enum, ref)
        except ProvisioningScriptRefNotFoundError:
            return ScriptServeResult(outcome=ScriptServeOutcome.not_found)
        except ProvisioningScriptUnavailableError:
            return ScriptServeResult(outcome=ScriptServeOutcome.unavailable)

        logger.info(
            "provisioning.script.served",
            variant=variant_enum.value,
            ref=ref,
            sha256_prefix=resolved.sha256[:12],
            cache_hit=resolved.cache_hit,
            session_id=str(session_id),
        )
        return ScriptServeResult(outcome=ScriptServeOutcome.ok, script=resolved)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ref_allowed(self, ref: str) -> bool:
        if PROVISIONING_REF_PATTERN.fullmatch(ref):
            return True
        dev_ref = self._settings.provisioning_script_dev_ref
        return bool(dev_ref) and ref == dev_ref and self._settings.environment != "production"

    def _cache_ttl_for(self, immutable_ref: bool) -> int:
        if immutable_ref:
            return self._settings.provisioning_script_cache_ttl_seconds
        return self._settings.provisioning_script_dev_cache_ttl_seconds

    async def _read_cache(self, cache_key: str) -> ResolvedScript | None:
        # Fail-open: a Redis outage must fall through to a live GitHub fetch,
        # never fail the request outright.
        try:
            cached = await self._redis.hgetall(cache_key)  # type: ignore[union-attr]
        except Exception:
            logger.warning("provisioning.script.cache_read_failed", exc_info=True)
            return None
        if not cached:
            return None
        content = cached.get("content")
        sha256 = cached.get("sha256")
        if content is None or sha256 is None:
            return None
        return ResolvedScript(content=str(content), sha256=str(sha256), cache_hit=True)

    async def _write_cache(self, cache_key: str, resolved: ResolvedScript, *, ttl: int) -> None:
        try:
            pipe = self._redis.pipeline()  # type: ignore[union-attr]
            pipe.hset(cache_key, mapping={"content": resolved.content, "sha256": resolved.sha256})
            pipe.expire(cache_key, ttl)
            await pipe.execute()
        except Exception:
            # Best-effort: a cache-write failure must not fail an otherwise-successful fetch.
            logger.warning("provisioning.script.cache_write_failed", exc_info=True)

    async def _fetch_from_github(self, *, repo: str, path: str, ref: str) -> ResolvedScript:
        url = f"{_GITHUB_API_BASE}/repos/{repo}/contents/{path}"
        headers = {
            "Accept": "application/vnd.github.raw",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._settings.github_content_token:
            headers["Authorization"] = f"Bearer {self._settings.github_content_token}"

        try:
            resp = await self._http.get(
                url, headers=headers, params={"ref": ref}, timeout=_FETCH_TIMEOUT_SECONDS
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "provisioning.script.fetch_error",
                repo=repo,
                ref=ref,
                error_class=exc.__class__.__name__,
            )
            raise ProvisioningScriptUnavailableError(f"GitHub fetch failed: {exc}") from exc

        if resp.status_code == 404:
            logger.info("provisioning.script.ref_not_found", repo=repo, ref=ref)
            raise ProvisioningScriptRefNotFoundError(f"{repo}@{ref}:{path} not found")

        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            logger.warning("provisioning.script.rate_limited", repo=repo, ref=ref)
            raise ProvisioningScriptUnavailableError("GitHub API rate limit exhausted")

        if resp.status_code != 200:
            logger.warning(
                "provisioning.script.fetch_failed",
                repo=repo,
                ref=ref,
                status_code=resp.status_code,
            )
            raise ProvisioningScriptUnavailableError(
                f"GitHub fetch returned unexpected status {resp.status_code}"
            )

        content = resp.text
        sha256 = hashlib.sha256(resp.content).hexdigest()
        return ResolvedScript(content=content, sha256=sha256, cache_hit=False)
