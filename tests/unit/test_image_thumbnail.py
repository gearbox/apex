"""Unit tests for server-side image thumbnail generation (Pillow)."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from src.api.services.image_thumbnail import (
    ImageDimensions,
    ThumbnailResult,
    make_image_thumbnail,
    read_dimensions,
)

pytestmark = pytest.mark.unit


def _make_png(width: int, height: int, mode: str = "RGB") -> bytes:
    """Create minimal PNG bytes of given size."""
    buf = io.BytesIO()
    img = Image.new(mode, (width, height), color=(128, 64, 32))
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_rgba_png(width: int, height: int) -> bytes:
    return _make_png(width, height, mode="RGBA")


class TestReadDimensions:
    async def test_returns_correct_dimensions(self) -> None:
        data = _make_png(1024, 768)
        result = await read_dimensions(data)
        assert result == ImageDimensions(width=1024, height=768)

    async def test_returns_none_for_non_image_bytes(self) -> None:
        result = await read_dimensions(b"not an image")
        assert result is None

    async def test_returns_none_for_empty_bytes(self) -> None:
        result = await read_dimensions(b"")
        assert result is None


class TestMakeImageThumbnail:
    async def test_downscales_landscape_to_max_edge(self) -> None:
        data = _make_png(2048, 1536)
        result = await make_image_thumbnail(data, max_edge=512)
        assert result is not None
        assert result.width == 512
        # aspect ratio: 2048/1536 = 4/3 → height = 384
        assert result.height == 384

    async def test_downscales_portrait_to_max_edge(self) -> None:
        data = _make_png(1024, 2048)
        result = await make_image_thumbnail(data, max_edge=512)
        assert result is not None
        assert result.height == 512
        assert result.width == 256

    async def test_result_is_webp(self) -> None:
        data = _make_png(800, 600)
        result = await make_image_thumbnail(data, max_edge=512)
        assert result is not None
        assert result.format == "webp"
        assert result.content_type == "image/webp"
        # Verify it's actually a WEBP by opening with Pillow
        img = Image.open(io.BytesIO(result.data))
        assert img.format == "WEBP"

    async def test_alpha_png_processed_successfully(self) -> None:
        """RGBA input does not cause an error; lossy WEBP may downconvert opaque alpha."""
        data = _make_rgba_png(400, 400)
        result = await make_image_thumbnail(data, max_edge=512)
        assert result is not None
        img = Image.open(io.BytesIO(result.data))
        # Lossy WEBP may strip a fully-opaque alpha channel; mode is RGB or RGBA.
        assert img.mode in ("RGB", "RGBA")

    async def test_small_image_not_upscaled(self) -> None:
        """Images smaller than max_edge should not be upscaled."""
        data = _make_png(100, 100)
        result = await make_image_thumbnail(data, max_edge=512)
        assert result is not None
        assert result.width <= 100
        assert result.height <= 100

    async def test_returns_none_for_non_image_bytes(self) -> None:
        result = await make_image_thumbnail(b"not an image")
        assert result is None

    async def test_returns_none_for_empty_bytes(self) -> None:
        result = await make_image_thumbnail(b"")
        assert result is None

    async def test_thumbnail_result_has_data(self) -> None:
        data = _make_png(512, 512)
        result = await make_image_thumbnail(data)
        assert isinstance(result, ThumbnailResult)
        assert len(result.data) > 0
