"""R2-cached currency logo download/store.

Fetched only at sync time (D8 — worker tick or admin refresh), never in a
request path. Logos are served from the R2 public assets domain, a distinct
origin from the API/FE, so a third-party SVG embedding a script is inert to
anyone but a direct navigator of the assets domain (D9).
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

import httpx
import structlog

from src.api.services.billing_errors import LogoCacheError, LogoStorageError
from src.api.services.storage.exceptions import StorageError

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.api.services.storage import R2StorageService

logger = structlog.get_logger(__name__)

# D11: hard caps on what a logo download may look like before it's trusted
# enough to store and serve from our own assets domain.
MAX_LOGO_BYTES = 512 * 1024
FETCH_TIMEOUT_SECONDS = 10.0
CONTENT_TYPE_EXTENSIONS: dict[str, str] = {
    "image/svg+xml": "svg",
    "image/png": "png",
    "image/webp": "webp",
    "image/jpeg": "jpg",
}
LOGO_KEY_PREFIX = "payment-currency-logos"
# Content-addressed keys never change contents, so the public assets domain
# can cache them forever.
LOGO_CACHE_CONTROL = "public, max-age=31536000, immutable"


class LogoCacheService:
    """Downloads and R2-caches provider currency logos, content-addressed (D7)."""

    def __init__(
        self,
        *,
        r2_client: R2StorageService,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        concurrency: int = 5,
    ) -> None:
        self._r2 = r2_client
        self._client_factory = http_client_factory or httpx.AsyncClient
        self._semaphore = asyncio.Semaphore(concurrency)

    async def ensure_cached(self, logo_url: str) -> str:
        """Download, content-hash, and R2-store a logo; return its object key.

        Content-addressed key (D7): re-downloading the same bytes for a
        different ticker/product is harmless — the exists-check makes the
        upload idempotent. Download and upload happen inside the same
        semaphore-bounded slot so buffered bytes are always bounded by
        ``concurrency * MAX_LOGO_BYTES``.

        Raises:
            LogoCacheError: On any download/validation/upload failure. The
                caller (PaymentCurrencySyncService) treats this as a
                per-logo failure (D10) — the entry's logo fields stay unset
                and the sync continues. Storage-backend failures (auth,
                timeout, 5xx) raise the `LogoStorageError` subclass so the
                caller can short-circuit the rest of the run (P1-2) instead
                of retrying a dead bucket once per ticker.
        """
        async with self._semaphore:
            content, content_type = await self._download(logo_url)
            ext = CONTENT_TYPE_EXTENSIONS[content_type]
            digest = hashlib.sha256(content).hexdigest()[:16]
            key = f"{LOGO_KEY_PREFIX}/{digest}.{ext}"

            try:
                if not await self._r2.exists(key):
                    await self._r2.put_raw(
                        key, content, content_type=content_type, cache_control=LOGO_CACHE_CONTROL
                    )
            except StorageError as exc:
                raise LogoStorageError(f"R2 storage failure for {key}: {exc}") from exc
            return key

    async def _download(self, logo_url: str) -> tuple[bytes, str]:
        try:
            async with (
                self._client_factory() as client,
                client.stream("GET", logo_url, timeout=FETCH_TIMEOUT_SECONDS) as response,
            ):
                response.raise_for_status()
                content_type = _validate_content_type(response, logo_url)
                content = await _read_bounded(response, logo_url)
                return content, content_type
        except LogoCacheError:
            raise
        except httpx.HTTPError as exc:
            raise LogoCacheError(f"Failed to download logo {logo_url}: {exc}") from exc


def _validate_content_type(response: httpx.Response, logo_url: str) -> str:
    content_type: str = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in CONTENT_TYPE_EXTENSIONS:
        raise LogoCacheError(f"Disallowed logo content-type '{content_type}' for {logo_url}")
    return content_type


async def _read_bounded(response: httpx.Response, logo_url: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > MAX_LOGO_BYTES:
            raise LogoCacheError(f"Logo exceeds {MAX_LOGO_BYTES} bytes: {logo_url}")
        chunks.append(chunk)
    return b"".join(chunks)
