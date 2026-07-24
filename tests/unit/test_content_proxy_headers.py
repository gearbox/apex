"""Content-proxy response header hardening (M2) + D1/D2/D3 streaming behavior.

Covers ``ContentProxyController._stream_from_r2``: inline-safe content types
are served inline, everything else is forced to download as
application/octet-stream, and X-Content-Type-Options/Content-Disposition are
always present. Since D2, metadata comes from a single ``stream_object``
GetObject call (no more ``head_object``) — the stub below mirrors that
contract by yielding an ``ObjectStream``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from litestar.response import Stream

from src.api.routes.content import ContentProxyController
from src.api.services.storage.r2 import ObjectStream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.unit

CACHE_TTL = 3600
SIZE_BYTES = 1234
ETAG = "etag123"


def _make_r2_mock(
    content_type: str | None,
    *,
    content_length: int = SIZE_BYTES,
    content_range: str | None = None,
    chunks: tuple[bytes, ...] = (b"chunk",),
) -> MagicMock:
    """Stub R2StorageService.stream_object as an async context manager.

    Mirrors the real ``stream_object`` contract post-D2: a single call
    yields an ``ObjectStream`` carrying content_type/content_length/
    content_range straight from the (stubbed) GetObject response.
    """
    resolved_content_type = content_type or "application/octet-stream"

    @asynccontextmanager
    async def _stream_object(
        _storage_key: str, *, range_header: str | None = None
    ) -> AsyncIterator[ObjectStream]:
        del range_header

        async def _chunks() -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

        yield ObjectStream(
            chunks=_chunks(),
            content_type=resolved_content_type,
            content_length=content_length,
            content_range=content_range,
        )

    r2_mock = MagicMock()
    r2_mock.stream_object = _stream_object
    return r2_mock


class TestInlineSafeContentTypes:
    async def test_inline_safe_image_serves_inline(self) -> None:
        r2_mock = _make_r2_mock("image/png")

        result = await ContentProxyController._stream_from_r2(
            r2_mock,
            "users/abc/outputs/img.png",
            ETAG,
            SIZE_BYTES,
            CACHE_TTL,
            range_header=None,
            if_none_match=None,
        )

        assert isinstance(result, Stream)
        assert result.media_type == "image/png"
        assert result.headers["X-Content-Type-Options"] == "nosniff"
        assert result.headers["Content-Disposition"] == "inline"

    async def test_inline_safe_video_serves_inline(self) -> None:
        r2_mock = _make_r2_mock("video/mp4")

        result = await ContentProxyController._stream_from_r2(
            r2_mock,
            "users/abc/outputs/vid.mp4",
            ETAG,
            SIZE_BYTES,
            CACHE_TTL,
            range_header=None,
            if_none_match=None,
        )

        assert isinstance(result, Stream)
        assert result.media_type == "video/mp4"
        assert result.headers["Content-Disposition"] == "inline"


class TestHostileContentTypesForceDownload:
    @pytest.mark.parametrize("stored_content_type", ["text/html", "image/svg+xml"])
    async def test_hostile_content_type_forces_download(self, stored_content_type: str) -> None:
        r2_mock = _make_r2_mock(stored_content_type)

        result = await ContentProxyController._stream_from_r2(
            r2_mock,
            "users/abc/outputs/file",
            ETAG,
            SIZE_BYTES,
            CACHE_TTL,
            range_header=None,
            if_none_match=None,
        )

        assert isinstance(result, Stream)
        assert result.media_type == "application/octet-stream"
        assert result.headers["X-Content-Type-Options"] == "nosniff"
        assert result.headers["Content-Disposition"] == "attachment"

    async def test_unknown_content_type_forces_download(self) -> None:
        r2_mock = _make_r2_mock("application/octet-stream")

        result = await ContentProxyController._stream_from_r2(
            r2_mock,
            "users/abc/outputs/file",
            ETAG,
            SIZE_BYTES,
            CACHE_TTL,
            range_header=None,
            if_none_match=None,
        )

        assert isinstance(result, Stream)
        assert result.media_type == "application/octet-stream"
        assert result.headers["Content-Disposition"] == "attachment"


class TestNosniffAlwaysPresent:
    async def test_nosniff_present_regardless_of_content_type(self) -> None:
        for content_type in ("image/jpeg", "image/webp", "text/html"):
            r2_mock = _make_r2_mock(content_type)
            result = await ContentProxyController._stream_from_r2(
                r2_mock,
                "users/abc/outputs/file",
                ETAG,
                SIZE_BYTES,
                CACHE_TTL,
                range_header=None,
                if_none_match=None,
            )
            assert isinstance(result, Stream)
            assert result.headers["X-Content-Type-Options"] == "nosniff"


class TestMissingContentTypeFallsBackToDownload:
    async def test_missing_content_type_defaults_to_octet_stream_attachment(self) -> None:
        r2_mock = _make_r2_mock(content_type=None)

        result = await ContentProxyController._stream_from_r2(
            r2_mock,
            "users/abc/outputs/file",
            ETAG,
            SIZE_BYTES,
            CACHE_TTL,
            range_header=None,
            if_none_match=None,
        )

        assert isinstance(result, Stream)
        assert result.media_type == "application/octet-stream"
        assert result.headers["X-Content-Type-Options"] == "nosniff"
        assert result.headers["Content-Disposition"] == "attachment"


class TestAcceptRangesAlwaysPresent:
    async def test_full_200_response_advertises_accept_ranges(self) -> None:
        r2_mock = _make_r2_mock("image/png")

        result = await ContentProxyController._stream_from_r2(
            r2_mock,
            "users/abc/outputs/img.png",
            ETAG,
            SIZE_BYTES,
            CACHE_TTL,
            range_header=None,
            if_none_match=None,
        )

        assert isinstance(result, Stream)
        assert result.headers["Accept-Ranges"] == "bytes"
        assert result.headers["Content-Length"] == str(SIZE_BYTES)


class TestInlineSafeSetDerivation:
    def test_inline_safe_is_stored_images_plus_mp4(self) -> None:
        from src.api.routes.content import _INLINE_SAFE_CONTENT_TYPES
        from src.api.services.storage.r2 import ALLOWED_CONTENT_TYPES

        assert frozenset(ALLOWED_CONTENT_TYPES) | {"video/mp4"} == _INLINE_SAFE_CONTENT_TYPES
