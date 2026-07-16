"""Unit tests for LogoCacheService: download hardening + content-addressed caching."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.services.billing_errors import LogoCacheError, LogoStorageError
from src.api.services.payment_currency_logos import (
    LOGO_CACHE_CONTROL,
    LOGO_KEY_PREFIX,
    MAX_LOGO_BYTES,
    LogoCacheService,
)
from src.api.services.storage.exceptions import StorageConnectionError

pytestmark = pytest.mark.unit


class _FakeStreamResponse:
    def __init__(self, *, content_type: str, chunks: list[bytes], status_ok: bool = True) -> None:
        self.headers = {"content-type": content_type}
        self._chunks = chunks
        self._status_ok = status_ok

    def raise_for_status(self) -> None:
        if not self._status_ok:
            import httpx

            raise httpx.HTTPStatusError("boom", request=MagicMock(), response=MagicMock())

    async def aiter_bytes(self) -> Any:
        for chunk in self._chunks:
            yield chunk


class _FakeStreamContextManager:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *args: object) -> bool:
        return False


def _client_for(response: _FakeStreamResponse) -> AsyncMock:
    client = AsyncMock()
    client.stream = MagicMock(return_value=_FakeStreamContextManager(response))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _r2_client(*, exists: bool = False) -> AsyncMock:
    r2 = AsyncMock()
    r2.exists = AsyncMock(return_value=exists)
    r2.put_raw = AsyncMock(return_value=None)
    return r2


async def test_ensure_cached_uploads_and_returns_content_addressed_key() -> None:
    content = b"<svg>coin</svg>"
    response = _FakeStreamResponse(content_type="image/svg+xml", chunks=[content])
    r2 = _r2_client(exists=False)
    service = LogoCacheService(r2_client=r2, http_client_factory=lambda: _client_for(response))

    key = await service.ensure_cached("https://nowpayments.io/images/coins/btc.svg")

    digest = hashlib.sha256(content).hexdigest()[:16]
    assert key == f"{LOGO_KEY_PREFIX}/{digest}.svg"
    r2.put_raw.assert_awaited_once_with(
        key, content, content_type="image/svg+xml", cache_control=LOGO_CACHE_CONTROL
    )


async def test_ensure_cached_skips_upload_when_key_already_exists() -> None:
    content = b"\x89PNG..."
    response = _FakeStreamResponse(content_type="image/png", chunks=[content])
    r2 = _r2_client(exists=True)
    service = LogoCacheService(r2_client=r2, http_client_factory=lambda: _client_for(response))

    await service.ensure_cached("https://nowpayments.io/images/coins/eth.png")

    r2.put_raw.assert_not_awaited()


async def test_identical_bytes_produce_stable_key_regardless_of_source_url() -> None:
    content = b"same-bytes"
    r2 = _r2_client(exists=False)

    response_a = _FakeStreamResponse(content_type="image/webp", chunks=[content])
    service_a = LogoCacheService(r2_client=r2, http_client_factory=lambda: _client_for(response_a))
    key_a = await service_a.ensure_cached("https://nowpayments.io/a.webp")

    response_b = _FakeStreamResponse(content_type="image/webp", chunks=[content])
    service_b = LogoCacheService(r2_client=r2, http_client_factory=lambda: _client_for(response_b))
    key_b = await service_b.ensure_cached("https://nowpayments.io/b.webp")

    assert key_a == key_b


async def test_disallowed_content_type_raises() -> None:
    response = _FakeStreamResponse(content_type="text/html", chunks=[b"<html></html>"])
    r2 = _r2_client()
    service = LogoCacheService(r2_client=r2, http_client_factory=lambda: _client_for(response))

    with pytest.raises(LogoCacheError, match="content-type"):
        await service.ensure_cached("https://nowpayments.io/x.svg")


async def test_oversized_download_raises() -> None:
    chunk = b"a" * (MAX_LOGO_BYTES // 2 + 1)
    response = _FakeStreamResponse(content_type="image/png", chunks=[chunk, chunk])
    r2 = _r2_client()
    service = LogoCacheService(r2_client=r2, http_client_factory=lambda: _client_for(response))

    with pytest.raises(LogoCacheError, match="exceeds"):
        await service.ensure_cached("https://nowpayments.io/big.png")


async def test_http_error_wrapped_as_logo_cache_error() -> None:
    response = _FakeStreamResponse(content_type="image/png", chunks=[b"x"], status_ok=False)
    r2 = _r2_client()
    service = LogoCacheService(r2_client=r2, http_client_factory=lambda: _client_for(response))

    with pytest.raises(LogoCacheError, match="Failed to download"):
        await service.ensure_cached("https://nowpayments.io/missing.png")


async def test_exists_storage_error_raises_logo_storage_error() -> None:
    content = b"<svg>coin</svg>"
    response = _FakeStreamResponse(content_type="image/svg+xml", chunks=[content])
    r2 = _r2_client()
    r2.exists = AsyncMock(side_effect=StorageConnectionError("403 Forbidden on HeadObject"))
    service = LogoCacheService(r2_client=r2, http_client_factory=lambda: _client_for(response))

    with pytest.raises(LogoStorageError, match="R2 storage failure"):
        await service.ensure_cached("https://nowpayments.io/btc.svg")

    # LogoStorageError is a LogoCacheError subclass, so existing callers that
    # only catch the base still see it.
    assert issubclass(LogoStorageError, LogoCacheError)


async def test_put_raw_storage_error_raises_logo_storage_error() -> None:
    content = b"<svg>coin</svg>"
    response = _FakeStreamResponse(content_type="image/svg+xml", chunks=[content])
    r2 = _r2_client(exists=False)
    r2.put_raw = AsyncMock(side_effect=StorageConnectionError("403 Forbidden on PutObject"))
    service = LogoCacheService(r2_client=r2, http_client_factory=lambda: _client_for(response))

    with pytest.raises(LogoStorageError, match="R2 storage failure"):
        await service.ensure_cached("https://nowpayments.io/btc.svg")


async def test_jpeg_extension_mapped_to_jpg() -> None:
    content = b"\xff\xd8\xff"
    response = _FakeStreamResponse(content_type="image/jpeg", chunks=[content])
    r2 = _r2_client(exists=False)
    service = LogoCacheService(r2_client=r2, http_client_factory=lambda: _client_for(response))

    key = await service.ensure_cached("https://nowpayments.io/x")

    assert key.endswith(".jpg")
