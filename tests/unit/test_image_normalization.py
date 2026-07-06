"""Unit tests for byte-level image format sniffing and normalization."""

from __future__ import annotations

import io
import struct
import zlib

import pytest
from PIL import Image, features

from src.api.services.image_normalization import (
    ImageNormalizationError,
    ImageTooLargeError,
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


def _fake_large_png(width: int, height: int) -> bytes:
    """Build a PNG with a legitimate IHDR declaring huge dimensions, but no
    real pixel data in IDAT.

    ``Image.open()`` parses only IHDR to populate ``.size`` — it never calls
    ``.load()`` (which would decode IDAT). This lets us prove the pixel cap
    is enforced from the header alone: if the cap check were accidentally
    moved after a decode attempt, this file would instead fail with a
    generic ``ImageNormalizationError`` (truncated/invalid IDAT), not
    ``ImageTooLargeError``.
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return sig + chunk(b"IHDR", ihdr_data) + chunk(b"IDAT", b"") + chunk(b"IEND", b"")


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
        result = await normalize_image(data, max_megapixels=100.0)

        assert isinstance(result, NormalizedImage)
        assert result.data == data
        assert result.format is MediaFormat.PNG
        assert result.content_type == "image/png"
        assert result.converted is False
        assert result.sniffed == SniffedFormat.PNG

    async def test_normalize_jpeg_passthrough(self) -> None:
        data = _jpeg()
        result = await normalize_image(data, max_megapixels=100.0)

        assert result.data == data
        assert result.format is MediaFormat.JPEG
        assert result.converted is False

    async def test_normalize_static_webp_passthrough(self) -> None:
        data = _webp()
        result = await normalize_image(data, max_megapixels=100.0)

        assert result.data == data
        assert result.format is MediaFormat.WEBP
        assert result.converted is False

    async def test_normalize_heic_converts_to_png(self) -> None:
        result = await normalize_image(_heic(), max_megapixels=100.0)

        assert result.converted is True
        assert result.format is MediaFormat.PNG
        assert result.sniffed == SniffedFormat.HEIF
        assert sniff_format(result.data) == SniffedFormat.PNG

    async def test_normalize_animated_webp_first_frame(self) -> None:
        size = (16, 12)
        result = await normalize_image(_webp_animated(size), max_megapixels=100.0)

        assert result.converted is True
        assert result.format is MediaFormat.PNG
        with Image.open(io.BytesIO(result.data)) as im:
            assert im.size == size

    async def test_normalize_preserves_alpha(self) -> None:
        result = await normalize_image(_heic(mode="RGBA"), max_megapixels=100.0)

        assert result.converted is True
        with Image.open(io.BytesIO(result.data)) as im:
            assert im.mode == "RGBA"
            pixel = im.getpixel((0, 0))
            assert isinstance(pixel, tuple)
            assert pixel[3] == 128  # alpha channel intact

    async def test_normalize_applies_exif_transpose(self) -> None:
        original_size = (20, 10)
        data = _tiff_with_orientation(6, size=original_size)

        result = await normalize_image(data, max_megapixels=100.0)

        assert result.converted is True
        with Image.open(io.BytesIO(result.data)) as im:
            # Orientation 6 is a 90-degree rotation: width/height swap.
            assert im.size == (original_size[1], original_size[0])

    async def test_normalize_garbage_raises(self) -> None:
        with pytest.raises(ImageNormalizationError):
            await normalize_image(_GARBAGE, max_megapixels=100.0)

    @pytest.mark.skipif(not features.check("avif"), reason="Pillow built without AVIF support")
    async def test_normalize_avif_converts_to_png(self) -> None:
        result = await normalize_image(_avif(), max_megapixels=100.0)
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

        result_data = _convert_to_png_sync(buf.getvalue(), max_megapixels=100.0)

        with Image.open(io.BytesIO(result_data)) as im:
            assert im.mode == "RGBA"
            pixel = im.getpixel((1, 1))
            assert isinstance(pixel, tuple)
            assert pixel[3] == 0

    def test_normalize_opaque_grayscale_converts_to_rgb_not_rgba(self) -> None:
        im = Image.new("L", (4, 4), 128)
        buf = io.BytesIO()
        im.save(buf, format="PNG")

        result_data = _convert_to_png_sync(buf.getvalue(), max_megapixels=100.0)

        with Image.open(io.BytesIO(result_data)) as reopened:
            assert reopened.mode == "RGB"


class TestEnsureComfyUIInput:
    async def test_ensure_comfyui_input_converts_static_webp(self) -> None:
        """The one behavioral delta vs normalize_image: static WebP -> PNG."""
        data = _webp()

        result = await ensure_comfyui_input(data, max_megapixels=100.0)

        assert result.converted is True
        assert result.format is MediaFormat.PNG
        assert result.sniffed == SniffedFormat.WEBP

    async def test_ensure_comfyui_input_png_passthrough(self) -> None:
        data = _png()
        result = await ensure_comfyui_input(data, max_megapixels=100.0)

        assert result.data == data
        assert result.converted is False

    async def test_ensure_comfyui_input_garbage_raises(self) -> None:
        with pytest.raises(ImageNormalizationError):
            await ensure_comfyui_input(_GARBAGE, max_megapixels=100.0)


# ---------------------------------------------------------------------------
# D4 — decompression-bomb pixel cap (F4)
# ---------------------------------------------------------------------------


class TestPixelCap:
    async def test_pixel_cap_rejects_oversized_png_before_decode(self) -> None:
        """A PNG whose IHDR declares dimensions over the cap is rejected from
        the header alone — proven by an IDAT that would fail a real decode:
        if the cap check ran after (or instead of) a decode attempt, this
        would raise a generic ImageNormalizationError, not ImageTooLargeError.
        """
        # 12000x12000 = 144 MP: over our 100 MP cap, but still under Pillow's
        # own DecompressionBombError hard limit (~178 MP) so Image.open()
        # itself doesn't block first — only warns — leaving our cap to fire.
        data = _fake_large_png(12_000, 12_000)

        with pytest.raises(ImageTooLargeError) as exc_info:
            await normalize_image(data, max_megapixels=100.0)

        assert exc_info.value.megapixels == pytest.approx(144.0)
        assert exc_info.value.limit == 100.0

    async def test_pixel_cap_applies_to_conversion_path(self) -> None:
        """force_png_for_webp routes static WebP through the conversion path
        (ensure_comfyui_input) — the cap must still apply there, not just to
        the default passthrough path."""
        data = _fake_large_png(12_000, 12_000)

        with pytest.raises(ImageTooLargeError):
            await ensure_comfyui_input(data, max_megapixels=100.0)

    @pytest.mark.parametrize(
        "builder,expected_sniffed",
        [
            (_png, SniffedFormat.PNG),
            (_jpeg, SniffedFormat.JPEG),
            (_webp, SniffedFormat.WEBP),
        ],
    )
    async def test_pixel_cap_applies_to_passthrough_formats(
        self, builder, expected_sniffed: SniffedFormat
    ) -> None:
        """The pixel cap gates passthrough formats too — a huge PNG/JPEG/
        static WebP would never otherwise be opened at all (previously
        passthrough bytes were trusted and returned unchecked), yet would
        still bomb the thumbnailer downstream.

        Uses a real, tiny image against an artificially tiny cap rather than
        a huge image, so the test stays fast — the size comparison logic is
        identical regardless of the absolute pixel counts involved.
        """
        data = builder()
        assert sniff_format(data) == expected_sniffed

        with pytest.raises(ImageTooLargeError):
            await normalize_image(data, max_megapixels=0.0001)  # 100 px cap; fixture is 16x12=192px

    async def test_image_too_large_error_distinct_from_decode_error(self) -> None:
        """ImageTooLargeError and a plain decode failure are never conflated —
        callers that need to distinguish "too big" from "not an image" (e.g.
        to map to 413 vs 400) can rely on the subclass."""
        assert issubclass(ImageTooLargeError, ImageNormalizationError)

        with pytest.raises(ImageTooLargeError):
            await normalize_image(_fake_large_png(12_000, 12_000), max_megapixels=100.0)

        with pytest.raises(ImageNormalizationError) as exc_info:
            await normalize_image(_GARBAGE, max_megapixels=100.0)
        assert not isinstance(exc_info.value, ImageTooLargeError)
