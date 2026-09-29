"""Content proxy — auth-gated streaming access to R2 objects.

Provides stable, non-expiring URLs for user content. The server
resolves ownership, then streams bytes from R2 with Cache-Control
headers. Presigned URLs are never exposed to the client.

Endpoints:
  GET /v1/content/outputs/{output_id}      — stream a generated output
  GET /v1/content/uploads/{image_id}       — stream an uploaded image
  GET /v1/content/feedback/{report_id}     — ADMIN/SUPERADMIN: stream the asset a feedback
                                             report points at (audit-logged, no-store)

All support HTTP Range (single range only — see src.api.utils.http_range)
and If-None-Match conditional requests.

Cache-Control stays `private, max-age=<content_url_ttl>, immutable` —
deliberately NOT `no-store`. A library grid renders on the order of thirty
thumbnails; `no-store` would re-fetch all of them on every render and page
switch, reproducing the parallel-request saturation behind a prior mobile
bug ("images disappear in the grid, showing blank black squares" after
several page switches), and would nullify the video prewarm design (its
`cacheMode: 'default'` warm exists specifically so `<video>` range requests
reuse HTTP-cached bytes). The residue concern `no-store` was proposed to fix
— private images sitting in a shared device's HTTP cache after an account
switch — is instead addressed by `Clear-Site-Data: "cache", "storage"` on
every session-ending endpoint (see CLEAR_SITE_DATA_HEADER,
src/api/security/response_headers.py) plus client-side session isolation
(the frontend never requests a previous account's content URLs). If further
defence in depth is wanted, shorten `content_url_ttl` instead of reaching
for `no-store` — that shrinks the residue window while preserving the
within-session cache benefit.

The feedback route is the exception: one-off admin triage views of other users'
media send `private, no-store` so an admin device never keeps them in HTTP cache.
"""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from litestar import Controller, Request, Response, get
from litestar.di import Provide
from litestar.response import Stream
from litestar.status_codes import (
    HTTP_206_PARTIAL_CONTENT,
    HTTP_304_NOT_MODIFIED,
    HTTP_404_NOT_FOUND,
    HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
    HTTP_502_BAD_GATEWAY,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_admin_user, get_current_user_id
from src.api.schemas.errors import ErrorEnvelope
from src.api.security import content_auth_guard
from src.api.services.content_proxy import ContentNotFoundError, ContentProxyService
from src.api.services.feedback import (
    FeedbackAssetNotFoundError,
    FeedbackNotFoundError,
    FeedbackService,
)
from src.api.services.storage.exceptions import StorageError, StorageRangeNotSatisfiableError
from src.api.services.storage.r2 import (
    ALLOWED_CONTENT_TYPES as _STORED_CONTENT_TYPES,
)
from src.api.services.storage.r2 import (
    R2StorageService,
)
from src.api.utils.http_range import (
    FullBody,
    ParsedRange,
    ServedRange,
    Unsatisfiable,
    parse_range,
)
from src.core.library_ref import LibraryAssetSource
from src.db.models import User

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

logger = structlog.get_logger(__name__)

# Types the proxy serves inline. Derived, not mirrored: exactly
# r2.ALLOWED_CONTENT_TYPES (services/storage/schemas.py::ALLOWED_UPLOAD_CONTENT_TYPES) —
# normalized image formats (upload normalization guarantees png/jpeg/webp — see
# services/image_normalization.py) plus the raw video formats accepted for
# user-uploaded and generated video content.
# NOTE: routes/storage.py::ALLOWED_CONTENT_TYPES is a DIFFERENT, wider set
# (services/storage/schemas.py::ALLOWED_CLIENT_UPLOAD_CONTENT_TYPES — the
# client-declared upload gate, incl. heic/heif/avif that are normalized to PNG
# before storage). Do NOT unify with that one.
_INLINE_SAFE_CONTENT_TYPES: frozenset[str] = frozenset(_STORED_CONTENT_TYPES)

_NO_STORE = "private, no-store"

_FEEDBACK_NOT_FOUND = ErrorEnvelope(
    error="feedback_not_found",
    message="Feedback report not found",
    status_code=HTTP_404_NOT_FOUND,
)
_ASSET_NOT_FOUND = ErrorEnvelope(
    error="asset_not_found",
    message="Asset not found",
    status_code=HTTP_404_NOT_FOUND,
)


def _if_none_match_satisfied(if_none_match: str, quoted_etag: str) -> bool:
    """Check a raw If-None-Match header value against our quoted ETag.

    Handles the wildcard form and comma-separated multi-value lists per
    RFC 7232 §3.2. Weak-validator prefixes aren't stripped — this proxy
    never emits weak ETags, so a strict match is correct.
    """
    if if_none_match.strip() == "*":
        return True
    candidates = (c.strip() for c in if_none_match.split(","))
    return quoted_etag in candidates


def _is_view_start(parsed: ParsedRange) -> bool:
    """True for the request that begins a view: a full body or a range from byte 0.

    Later range chunks of the same ``<video>`` playback are continuations and
    must not each write an audit row.
    """
    return isinstance(parsed, FullBody) or (isinstance(parsed, ServedRange) and parsed.start == 0)


class ContentProxyController(Controller):
    """Auth-gated streaming proxy for R2 content."""

    path = "/v1/content"
    tags: Sequence[str] | None = ("Content",)
    dependencies = {"current_user_id": Provide(get_current_user_id)}  # noqa: RUF012

    @get("/outputs/{output_id:uuid}", guards=[content_auth_guard])
    async def proxy_output(
        self,
        request: Request[Any, Any, Any],
        current_user_id: UUID,
        product_id: str,
        output_id: UUID,
        session: AsyncSession,
        content_proxy: ContentProxyService,
        r2_storage: R2StorageService,
    ) -> Stream | Response[ErrorEnvelope]:
        """Stream a generated output."""
        try:
            storage_key, etag, size_bytes = await content_proxy.resolve_output(
                output_id,
                user_id=current_user_id,
                product_id=product_id,
                session=session,
            )
        except ContentNotFoundError:
            return Response(
                content=ErrorEnvelope(
                    error="not_found",
                    message="Output not found",
                    status_code=HTTP_404_NOT_FOUND,
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

        return await self._stream_from_r2(
            r2_storage,
            storage_key,
            etag,
            size_bytes,
            content_proxy.ttl,
            range_header=request.headers.get("range"),
            if_none_match=request.headers.get("if-none-match"),
        )

    @get("/uploads/{image_id:uuid}", guards=[content_auth_guard])
    async def proxy_upload(
        self,
        request: Request[Any, Any, Any],
        current_user_id: UUID,
        product_id: str,
        image_id: UUID,
        session: AsyncSession,
        content_proxy: ContentProxyService,
        r2_storage: R2StorageService,
    ) -> Stream | Response[ErrorEnvelope]:
        """Stream an uploaded image."""
        try:
            storage_key, etag, size_bytes = await content_proxy.resolve_upload(
                image_id,
                user_id=current_user_id,
                product_id=product_id,
                session=session,
            )
        except ContentNotFoundError:
            return Response(
                content=ErrorEnvelope(
                    error="not_found",
                    message="Upload not found",
                    status_code=HTTP_404_NOT_FOUND,
                ),
                status_code=HTTP_404_NOT_FOUND,
            )

        return await self._stream_from_r2(
            r2_storage,
            storage_key,
            etag,
            size_bytes,
            content_proxy.ttl,
            range_header=request.headers.get("range"),
            if_none_match=request.headers.get("if-none-match"),
        )

    @get(
        "/feedback/{report_id:uuid}",
        guards=[content_auth_guard],
        dependencies={"admin": Provide(get_current_admin_user)},
    )
    async def proxy_feedback_asset(
        self,
        request: Request[Any, Any, Any],
        admin: User,
        product_id: str,
        report_id: UUID,
        session: AsyncSession,
        content_proxy: ContentProxyService,
        r2_storage: R2StorageService,
        feedback_service: FeedbackService,
    ) -> Stream | Response[ErrorEnvelope]:
        """Stream the asset a feedback report points at — ADMIN/SUPERADMIN, report-scoped.

        Order is security-relevant: admin role (dependency) → report, scoped to
        the request's product → *owner-scoped* resolve as the reporter → open
        R2 stream → audit (commit) → stream body. Nothing touches R2 before
        every check has passed, and the audit row is committed before the
        response starts streaming (the session must not be relied on once the
        body is being yielded).
        """
        try:
            target = await feedback_service.get_asset_target_for_admin(
                report_id, product_id=product_id
            )
        except FeedbackNotFoundError:
            return Response(content=_FEEDBACK_NOT_FOUND, status_code=HTTP_404_NOT_FOUND)
        except FeedbackAssetNotFoundError:
            return Response(content=_ASSET_NOT_FOUND, status_code=HTTP_404_NOT_FOUND)

        resolver = (
            content_proxy.resolve_output
            if target.source is LibraryAssetSource.OUTPUT
            else content_proxy.resolve_upload
        )
        try:
            storage_key, etag, size_bytes = await resolver(
                target.asset_id,
                user_id=target.owner_id,
                product_id=product_id,
                session=session,
            )
        except ContentNotFoundError:
            return Response(content=_ASSET_NOT_FOUND, status_code=HTTP_404_NOT_FOUND)

        range_header = request.headers.get("range")
        on_stream_opened: Callable[[], Awaitable[None]] | None = None
        if _is_view_start(parse_range(range_header, size_bytes)):

            async def record_view_on_stream_opened() -> None:
                await feedback_service.record_asset_view(
                    target, admin_id=admin.id, product_id=product_id
                )
                await session.commit()
                logger.info(
                    "content.feedback_asset.viewed",
                    report_id=str(target.report_id),
                    asset_ref=target.asset_ref,
                    admin_id=str(admin.id),
                    owner_id=str(target.owner_id),
                    product_id=product_id,
                )

            on_stream_opened = record_view_on_stream_opened

        return await self._stream_from_r2(
            r2_storage,
            storage_key,
            etag,
            size_bytes,
            content_proxy.ttl,
            range_header=range_header,
            if_none_match=request.headers.get("if-none-match"),
            cache_control=_NO_STORE,
            on_stream_opened=on_stream_opened,
        )

    @staticmethod
    async def _stream_from_r2(
        r2: R2StorageService,
        storage_key: str,
        etag: str,
        size_bytes: int,
        cache_ttl: int,
        *,
        range_header: str | None,
        if_none_match: str | None,
        cache_control: str | None = None,
        on_stream_opened: Callable[[], Awaitable[None]] | None = None,
    ) -> Stream | Response[ErrorEnvelope]:
        """Resolve ownership → conditional GET → single R2 GET → callback → response.

        Ordering is deliberate and security-relevant: this runs strictly
        after the caller has already resolved ownership (storage_key/etag
        came from an owner-scoped DB lookup) — nothing here streams a byte
        before that check has passed. The conditional-GET short-circuit
        (D3) comes next since it can skip R2 entirely; the DB-recorded
        size_bytes (immutable once written) lets an unsatisfiable Range be
        rejected with zero R2 traffic too (D1). Only a genuinely servable
        request reaches R2, and it does so with exactly one GetObject call
        (D2) — ranged or full, decided before the call is made. Once R2 has
        produced a stream, ``on_stream_opened`` runs (for example, to commit
        an audit row) before the response is returned, so no body byte is
        yielded before it completes.
        """
        quoted_etag = f'"{etag}"'
        if cache_control is None:
            cache_control = f"private, max-age={cache_ttl}, immutable"

        if if_none_match is not None and _if_none_match_satisfied(if_none_match, quoted_etag):
            return Response(
                content=None,  # type: ignore[arg-type]
                status_code=HTTP_304_NOT_MODIFIED,
                headers={"ETag": quoted_etag, "Cache-Control": cache_control},
            )

        parsed_range = parse_range(range_header, size_bytes)

        if isinstance(parsed_range, Unsatisfiable):
            return Response(
                content=ErrorEnvelope(
                    error="range_not_satisfiable",
                    message="The requested range is not satisfiable",
                    status_code=HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                ),
                status_code=HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                headers={"Content-Range": f"bytes */{size_bytes}"},
            )

        r2_range_header = (
            f"bytes={parsed_range.start}-{parsed_range.end}"
            if isinstance(parsed_range, ServedRange)
            else None
        )

        try:
            stream_ctx = r2.stream_object(storage_key, range_header=r2_range_header)
            obj = await stream_ctx.__aenter__()
        except StorageRangeNotSatisfiableError:
            # R2 itself rejected the range — only reachable if the
            # DB-recorded size_bytes used above was stale.
            logger.warning("content_proxy.r2_range_rejected", storage_key=storage_key)
            return Response(
                content=ErrorEnvelope(
                    error="range_not_satisfiable",
                    message="The requested range is not satisfiable",
                    status_code=HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                ),
                status_code=HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                headers={"Content-Range": f"bytes */{size_bytes}"},
            )
        except StorageError:
            logger.warning("content_proxy.r2_fetch_failed", storage_key=storage_key)
            return Response(
                content=ErrorEnvelope(
                    error="upstream_error",
                    message="Failed to fetch content",
                    status_code=HTTP_502_BAD_GATEWAY,
                ),
                status_code=HTTP_502_BAD_GATEWAY,
            )

        if on_stream_opened is not None:
            try:
                await on_stream_opened()
            except BaseException:
                # The R2 context was successfully opened, so close it with
                # the real callback exception before propagating the failure.
                await stream_ctx.__aexit__(*sys.exc_info())
                raise

        async def _streaming_body() -> AsyncIterator[bytes]:
            try:
                async for chunk in obj.chunks:
                    yield chunk
            except BaseException:
                # Forward the real exception (including a client-initiated
                # GeneratorExit on early stream close) so the R2 client
                # context tears down the same way it would inside a normal
                # `async with` block, instead of masking it as a clean exit.
                await stream_ctx.__aexit__(*sys.exc_info())
                raise
            else:
                await stream_ctx.__aexit__(None, None, None)

        is_inline_safe = obj.content_type in _INLINE_SAFE_CONTENT_TYPES
        media_type = obj.content_type if is_inline_safe else "application/octet-stream"
        disposition = "inline" if is_inline_safe else "attachment"

        headers = {
            "Cache-Control": cache_control,
            "ETag": quoted_etag,
            "X-Content-Id": etag,
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": disposition,
            "Accept-Ranges": "bytes",
            "Content-Length": str(obj.content_length),
        }

        if obj.content_range is not None:
            headers["Content-Range"] = obj.content_range
            status_code = HTTP_206_PARTIAL_CONTENT
        else:
            status_code = None  # Stream defaults to 200

        return Stream(
            _streaming_body(),
            media_type=media_type,
            headers=headers,
            status_code=status_code,
        )
