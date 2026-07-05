"""Unit tests for byte-level image format sniffing and normalization."""

from __future__ import annotations

import io

import pytest
from PIL import Image, features

from src.api.services.image_normalization import (
    ImageNormalizationError,
    NormalizedImage,
    SniffedFormat,
    _convert_to_png_sync,
    ensure_comfyui_input,
    normalize_image,
    sniff_format,
)
from src.core.enums import MediaFormat

pytestmark = pytest.mark.unit


def _png(mode: str = "RGB", size: tuple[int, int] = (16, 12)) -> bytes:
    color: tuple[int, ...] = (255, 0, 0, 128) if mode == "RGBA" else (255, 0, 0)
    im = Image.new(mode, size, color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg(size: tuple[int, int] = (16, 12)) -> bytes:
    im = Image.new("RGB", size, (0, 255, 0))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()


def _webp(mode: str = "RGB", size: tuple[int, int] = (16, 12)) -> bytes:
    color: tuple[int, ...] = (10, 20, 30, 128) if mode == "RGBA" else (10, 20, 30)
    im = Image.new(mode, size, color)
    buf = io.BytesIO()
    im.save(buf, format="WEBP")
    return buf.getvalue()


def _webp_animated(size: tuple[int, int] = (16, 12)) -> bytes:
    frames = [Image.new("RGB", size, "red"), Image.new("RGB", size, "blue")]
    buf = io.BytesIO()
    frames[0].save(buf, format="WEBP", save_all=True, append_images=frames[1:])
    return buf.getvalue()


def _heic(mode: str = "RGB", size: tuple[int, int] = (16, 12)) -> bytes:
    color: tuple[int, ...] = (200, 100, 50, 128) if mode == "RGBA" else (200, 100, 50)
    im = Image.new(mode, size, color)
    buf = io.BytesIO()
    im.save(buf, format="HEIF")
    return buf.getvalue()


def _avif(size: tuple[int, int] = (16, 12)) -> bytes:
    im = Image.new("RGB", size, (5, 6, 7))
    buf = io.BytesIO()
    im.save(buf, format="AVIF")
    return buf.getvalue()


def _tiff_with_orientation(orientation: int, size: tuple[int, int] = (20, 10)) -> bytes:
    """Build a TIFF (sniffs as UNKNOWN) carrying an EXIF Orientation tag.

    Routes through the decode-and-convert path in normalize_image, letting
    us exercise exif_transpose without relying on JPEG/PNG passthrough.
    """
    im = Image.new("RGB", size, "red")
    exif = im.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    im.save(buf, format="TIFF", exif=exif)
    return buf.getvalue()


_GARBAGE = b"this is definitely not an image, just plain text bytes"


class TestSniffFormat:
    def test_sniff_png(self) -> None:
        assert sniff_format(_png()) == SniffedFormat.PNG

    def test_sniff_jpeg(self) -> None:
        assert sniff_format(_jpeg()) == SniffedFormat.JPEG

    def test_sniff_webp_static(self) -> None:
        assert sniff_format(_webp()) == SniffedFormat.WEBP

    def test_sniff_webp_animated(self) -> None:
        data = _webp_animated()
        assert data[12:16] == b"VP8X"
        assert data[20] & 0x02, "test fixture must actually set the animation flag"
        assert sniff_format(data) == SniffedFormat.WEBP_ANIMATED

    def test_sniff_heic(self) -> None:
        assert sniff_format(_heic()) == SniffedFormat.HEIF

    @pytest.mark.skipif(not features.check("avif"), reason="Pillow built without AVIF support")
    def test_sniff_avif(self) -> None:
        assert sniff_format(_avif()) == SniffedFormat.AVIF

    def test_sniff_unknown_garbage(self) -> None:
        assert sniff_format(_GARBAGE) == SniffedFormat.UNKNOWN

    def test_sniff_never_raises_on_short_buffers(self) -> None:
        for n in range(24):
            assert sniff_format(b"\x00" * n) == SniffedFormat.UNKNOWN


class TestNormalizeImage:
    async def test_normalize_png_passthrough(self) -> None:
        data = _png()
        result = await normalize_image(data)

        assert isinstance(result, NormalizedImage)
        assert result.data == data
        assert result.format is MediaFormat.PNG
        assert result.content_type == "image/png"
        assert result.converted is False
        assert result.sniffed == SniffedFormat.PNG

    async def test_normalize_jpeg_passthrough(self) -> None:
        data = _jpeg()
        result = await normalize_image(data)

        assert result.data == data
        assert result.format is MediaFormat.JPEG
        assert result.converted is False

    async def test_normalize_static_webp_passthrough(self) -> None:
        data = _webp()
        result = await normalize_image(data)

        assert result.data == data
        assert result.format is MediaFormat.WEBP
        assert result.converted is False

    async def test_normalize_heic_converts_to_png(self) -> None:
        result = await normalize_image(_heic())

        assert result.converted is True
        assert result.format is MediaFormat.PNG
        assert result.sniffed == SniffedFormat.HEIF
        assert sniff_format(result.data) == SniffedFormat.PNG

    async def test_normalize_animated_webp_first_frame(self) -> None:
        size = (16, 12)
        result = await normalize_image(_webp_animated(size))

        assert result.converted is True
        assert result.format is MediaFormat.PNG
        with Image.open(io.BytesIO(result.data)) as im:
            assert im.size == size

    async def test_normalize_preserves_alpha(self) -> None:
        result = await normalize_image(_heic(mode="RGBA"))

        assert result.converted is True
        with Image.open(io.BytesIO(result.data)) as im:
            assert im.mode == "RGBA"
            pixel = im.getpixel((0, 0))
            assert isinstance(pixel, tuple)
            assert pixel[3] == 128  # alpha channel intact

    async def test_normalize_applies_exif_transpose(self) -> None:
        original_size = (20, 10)
        data = _tiff_with_orientation(6, size=original_size)

        result = await normalize_image(data)

        assert result.converted is True
        with Image.open(io.BytesIO(result.data)) as im:
            # Orientation 6 is a 90-degree rotation: width/height swap.
            assert im.size == (original_size[1], original_size[0])

    async def test_normalize_garbage_raises(self) -> None:
        with pytest.raises(ImageNormalizationError):
            await normalize_image(_GARBAGE)

    @pytest.mark.skipif(not features.check("avif"), reason="Pillow built without AVIF support")
    async def test_normalize_avif_converts_to_png(self) -> None:
        result = await normalize_image(_avif())
        assert result.format is MediaFormat.PNG
        assert result.converted is True
        assert result.sniffed is SniffedFormat.AVIF
        assert sniff_format(result.data) is SniffedFormat.PNG

    def test_normalize_palette_transparency_preserved(self) -> None:
        rgba = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
        rgba.putpixel((0, 0), (0, 255, 0, 255))
        pal = rgba.convert("P", palette=Image.Palette.ADAPTIVE)
        buf = io.BytesIO()
        pal.save(buf, format="PNG")

        result_data = _convert_to_png_sync(buf.getvalue())

        with Image.open(io.BytesIO(result_data)) as im:
            assert im.mode == "RGBA"
            pixel = im.getpixel((1, 1))
            assert isinstance(pixel, tuple)
            assert pixel[3] == 0

    def test_normalize_opaque_grayscale_converts_to_rgb_not_rgba(self) -> None:
        im = Image.new("L", (4, 4), 128)
        buf = io.BytesIO()
        im.save(buf, format="PNG")

        result_data = _convert_to_png_sync(buf.getvalue())

        with Image.open(io.BytesIO(result_data)) as reopened:
            assert reopened.mode == "RGB"


class TestEnsureComfyUIInput:
    async def test_ensure_comfyui_input_converts_static_webp(self) -> None:
        """The one behavioral delta vs normalize_image: static WebP -> PNG."""
        data = _webp()

        result = await ensure_comfyui_input(data)

        assert result.converted is True
        assert result.format is MediaFormat.PNG
        assert result.sniffed == SniffedFormat.WEBP

    async def test_ensure_comfyui_input_png_passthrough(self) -> None:
        data = _png()
        result = await ensure_comfyui_input(data)

        assert result.data == data
        assert result.converted is False

    async def test_ensure_comfyui_input_garbage_raises(self) -> None:
        with pytest.raises(ImageNormalizationError):
            await ensure_comfyui_input(_GARBAGE)
