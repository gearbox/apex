"""Content-proxy response header hardening (M2).

Covers ``ContentProxyController._stream_from_r2``: inline-safe content types
are served inline, everything else is forced to download as
application/octet-stream, and X-Content-Type-Options/Content-Disposition are
always present. Reuses the R2 mocking pattern from
``tests/unit/test_route_handlers.py::TestContentStreamFromR2`` — a MagicMock
R2 client whose ``_get_client()`` is an async context manager yielding a mock
with ``head_object``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from litestar.response import Stream

from src.api.routes.content import ContentProxyController

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.unit


def _make_r2_mock(
    content_type: str | None,
    content_length: int = 1234,
) -> MagicMock:
    head: dict[str, object] = {"ContentLength": content_length}
    if content_type is not None:
        head["ContentType"] = content_type
    client_mock = AsyncMock()
    client_mock.head_object = AsyncMock(return_value=head)

    r2_mock = MagicMock()
    r2_mock._settings.bucket_name = "test-bucket"

    @asynccontextmanager
    async def _fake_get_client() -> AsyncIterator[AsyncMock]:
        yield client_mock

    r2_mock._get_client = _fake_get_client
    return r2_mock


class TestInlineSafeContentTypes:
    async def test_inline_safe_image_serves_inline(self) -> None:
        r2_mock = _make_r2_mock("image/png")

        result = await ContentProxyController._stream_from_r2(
            r2_mock, "users/abc/outputs/img.png", "etag123", 3600
        )

        assert isinstance(result, Stream)
        assert result.media_type == "image/png"
        assert result.headers["X-Content-Type-Options"] == "nosniff"
        assert result.headers["Content-Disposition"] == "inline"

    async def test_inline_safe_video_serves_inline(self) -> None:
        r2_mock = _make_r2_mock("video/mp4")

        result = await ContentProxyController._stream_from_r2(
            r2_mock, "users/abc/outputs/vid.mp4", "etag123", 3600
        )

        assert isinstance(result, Stream)
        assert result.media_type == "video/mp4"
        assert result.headers["Content-Disposition"] == "inline"


class TestHostileContentTypesForceDownload:
    @pytest.mark.parametrize("stored_content_type", ["text/html", "image/svg+xml"])
    async def test_hostile_content_type_forces_download(self, stored_content_type: str) -> None:
        r2_mock = _make_r2_mock(stored_content_type)

        result = await ContentProxyController._stream_from_r2(
            r2_mock, "users/abc/outputs/file", "etag123", 3600
        )

        assert isinstance(result, Stream)
        assert result.media_type == "application/octet-stream"
        assert result.headers["X-Content-Type-Options"] == "nosniff"
        assert result.headers["Content-Disposition"] == "attachment"

    async def test_unknown_content_type_forces_download(self) -> None:
        r2_mock = _make_r2_mock("application/octet-stream")

        result = await ContentProxyController._stream_from_r2(
            r2_mock, "users/abc/outputs/file", "etag123", 3600
        )

        assert isinstance(result, Stream)
        assert result.media_type == "application/octet-stream"
        assert result.headers["Content-Disposition"] == "attachment"


class TestNosniffAlwaysPresent:
    async def test_nosniff_present_regardless_of_content_type(self) -> None:
        for content_type in ("image/jpeg", "image/webp", "text/html"):
            r2_mock = _make_r2_mock(content_type)
            result = await ContentProxyController._stream_from_r2(
                r2_mock, "users/abc/outputs/file", "etag123", 3600
            )
            assert isinstance(result, Stream)
            assert result.headers["X-Content-Type-Options"] == "nosniff"


class TestMissingContentTypeFallsBackToDownload:
    async def test_head_without_content_type_defaults_to_octet_stream_attachment(self) -> None:
        r2_mock = _make_r2_mock(content_type=None)

        result = await ContentProxyController._stream_from_r2(
            r2_mock, "users/abc/outputs/file", "etag123", 3600
        )

        assert isinstance(result, Stream)
        assert result.media_type == "application/octet-stream"
        assert result.headers["X-Content-Type-Options"] == "nosniff"
        assert result.headers["Content-Disposition"] == "attachment"


class TestInlineSafeSetDerivation:
    def test_inline_safe_is_stored_images_plus_mp4(self) -> None:
        from src.api.routes.content import _INLINE_SAFE_CONTENT_TYPES
        from src.api.services.storage.r2 import ALLOWED_CONTENT_TYPES

        assert frozenset(ALLOWED_CONTENT_TYPES) | {"video/mp4"} == _INLINE_SAFE_CONTENT_TYPES
