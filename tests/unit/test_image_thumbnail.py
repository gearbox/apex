"""Unit tests for server-side image thumbnail generation (Pillow)."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from src.api.services.image_thumbnail import (
    GeneratedThumbnail,
    ImageDimensions,
    ThumbnailResult,
    make_image_thumbnail,
    make_image_thumbnails,
    read_dimensions,
)
from src.core.thumbnails import THUMBNAIL_SPECS

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

    async def test_palette_image_converted_and_thumbnailed(self) -> None:
        """Palette (P mode) images trigger the non-RGB/RGBA conversion branch."""
        src = Image.new("RGB", (300, 200), "red").convert("P")
        buf = io.BytesIO()
        src.save(buf, format="PNG")
        data = buf.getvalue()

        result = await make_image_thumbnail(data, max_edge=512)

        assert result is not None
        assert result.format == "webp"
        assert max(result.width, result.height) <= 512
        img = Image.open(io.BytesIO(result.data))
        assert img.format == "WEBP"

    async def test_cmyk_image_converted_and_thumbnailed(self) -> None:
        """CMYK images trigger the non-RGB/RGBA conversion branch."""
        src = Image.new("CMYK", (300, 200))
        buf = io.BytesIO()
        src.save(buf, format="JPEG")
        data = buf.getvalue()

        result = await make_image_thumbnail(data, max_edge=512)

        assert result is not None
        assert result.format == "webp"
        img = Image.open(io.BytesIO(result.data))
        assert img.format == "WEBP"


def _make_palette_png(width: int, height: int, fill_rgb: tuple[int, int, int]) -> bytes:
    src = Image.new("RGB", (width, height), fill_rgb)
    palette_img = src.quantize(colors=16)
    buf = io.BytesIO()
    palette_img.save(buf, format="PNG")
    return buf.getvalue()


class TestMakeImageThumbnails:
    async def test_returns_one_thumbnail_per_spec(self) -> None:
        data = _make_png(800, 600)
        results = await make_image_thumbnails(data, THUMBNAIL_SPECS)
        assert len(results) == len(THUMBNAIL_SPECS)
        assert all(isinstance(r, GeneratedThumbnail) for r in results)

    async def test_each_result_is_webp(self) -> None:
        data = _make_png(800, 600)
        results = await make_image_thumbnails(data, THUMBNAIL_SPECS)
        for r in results:
            assert r.result.format == "webp"
            assert r.result.content_type == "image/webp"

    async def test_longest_edge_matches_spec(self) -> None:
        data = _make_png(1024, 768)
        results = await make_image_thumbnails(data, THUMBNAIL_SPECS)
        for r in results:
            longest = max(r.result.width, r.result.height)
            assert longest <= r.spec.max_edge

    async def test_portrait_dims_truthful(self) -> None:
        """Portrait input preserves width < height in generated thumbnails."""
        data = _make_png(400, 800)
        results = await make_image_thumbnails(data, THUMBNAIL_SPECS)
        assert len(results) > 0
        for r in results:
            assert r.result.width < r.result.height

    async def test_returns_empty_list_for_invalid_input(self) -> None:
        results = await make_image_thumbnails(b"not an image")
        assert results == []

    async def test_palette_png_color_preserved(self) -> None:
        """Palette images must not render as solid black (F1 regression guard)."""
        # Pure red palette PNG — frombytes() strips palette, giving black before fix
        data = _make_palette_png(300, 200, fill_rgb=(200, 10, 10))
        results = await make_image_thumbnails(data, THUMBNAIL_SPECS)
        assert len(results) > 0
        for r in results:
            img = Image.open(io.BytesIO(r.result.data)).convert("RGB")
            # At least one pixel must have dominant red — proves palette was applied
            pixels = list(img.getdata())  # type: ignore[arg-type]
            assert any(p[0] > 150 and p[1] < 80 and p[2] < 80 for p in pixels), (
                f"All pixels appear black/wrong for spec {r.spec.label}: sample={pixels[:4]}"
            )

    async def test_cmyk_color_preserved(self) -> None:
        """CMYK images must not lose color across all size variants."""
        src = Image.new("CMYK", (300, 200), (0, 200, 200, 0))
        buf = io.BytesIO()
        src.save(buf, format="JPEG")
        data = buf.getvalue()
        results = await make_image_thumbnails(data, THUMBNAIL_SPECS)
        assert len(results) > 0
        for r in results:
            img = Image.open(io.BytesIO(r.result.data)).convert("RGB")
            pixels = list(img.getdata())  # type: ignore[arg-type]
            assert any(p != (0, 0, 0) for p in pixels)
