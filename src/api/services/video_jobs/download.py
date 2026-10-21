"""Credential-free HTTP client for fetching finished provider videos."""

from __future__ import annotations

import httpx


class VideoDownloadClient:
    """Owns the ``httpx`` client used to download a result URL.

    Deliberately separate from any provider API client: result URLs are
    presigned object-store links and must never be fetched with the provider's
    bearer credentials attached.
    """

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Open the client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0),
                follow_redirects=True,
                transport=self._transport,
            )

    async def close(self) -> None:
        """Close the client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """The open client; raises before ``connect()``."""
        if self._client is None:
            raise RuntimeError("VideoDownloadClient is not connected. Call connect() first.")
        return self._client
