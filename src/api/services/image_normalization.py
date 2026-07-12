"""Byte-level image format sniffing and normalization (Pillow).

The sniffed bytes are the single source of truth for image format.
Declared content types and filename extensions are provenance metadata only.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from enum import StrEnum

import structlog
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from src.core.enums import MediaFormat

register_heif_opener()

logger = structlog.get_logger(__name__)


class ImageNormalizationError(Exception):
    """Raised when bytes cannot be decoded as an image."""


class ImageTooLargeError(ImageNormalizationError):
    """Raised when an image's pixel count exceeds the configured cap.

    Distinct from a generic decode failure: the bytes were a perfectly
    decodable image, they're just too large to safely process. Raised only
    after a successful ``Image.open(...).size`` header read — never for
    truncated/hostile bytes that fail to open at all (those are a plain
    ``ImageNormalizationError``).
    """

    def __init__(self, *, megapixels: float, limit: float) -> None:
        self.megapixels = megapixels
        self.limit = limit
        super().__init__(
            f"Input image exceeds maximum pixel count: {megapixels:.1f}MP > {limit:.1f}MP limit"
        )


class SniffedFormat(StrEnum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"  # static
    WEBP_ANIMATED = "webp_animated"
    HEIF = "heif"  # heic/heif brands
    AVIF = "avif"
    UNKNOWN = "unknown"


_HEIF_BRANDS = {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1"}
_AVIF_BRANDS = {b"avif", b"avis"}


def sniff_format(data: bytes) -> SniffedFormat:
    """Inspect header bytes to determine image format without decoding."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return SniffedFormat.PNG
    if data[:3] == b"\xff\xd8\xff":
        return SniffedFormat.JPEG
    if len(data) >= 16 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk_fourcc = data[12:16]
        if chunk_fourcc in (b"VP8 ", b"VP8L"):
            return SniffedFormat.WEBP
        if chunk_fourcc == b"VP8X":
            if len(data) >= 21 and data[20] & 0x02:
                return SniffedFormat.WEBP_ANIMATED
            return SniffedFormat.WEBP
        return SniffedFormat.UNKNOWN
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in _AVIF_BRANDS:
            return SniffedFormat.AVIF
        return SniffedFormat.HEIF if brand in _HEIF_BRANDS else SniffedFormat.UNKNOWN
    return SniffedFormat.UNKNOWN


@dataclass(frozen=True, slots=True)
class NormalizedImage:
    data: bytes
    format: MediaFormat  # PNG | JPEG | WEBP only
    content_type: str  # derived: format.content_type
    converted: bool  # True when bytes were re-encoded
    sniffed: SniffedFormat  # what the input actually was


_PASSTHROUGH_FORMATS = {
    SniffedFormat.PNG: MediaFormat.PNG,
    SniffedFormat.JPEG: MediaFormat.JPEG,
    SniffedFormat.WEBP: MediaFormat.WEBP,
}


def _check_pixel_cap(img: Image.Image, limit_mp: float) -> None:
    """Raise ``ImageTooLargeError`` if ``img``'s pixel count exceeds ``limit_mp``.

    Reads only ``img.size`` — ``Image.open(...)`` decodes the header lazily,
    so this runs before any full pixel decode (``.load()``, ``exif_transpose``,
    ``.convert()``), bounding per-request decode memory.
    """
    width, height = img.size
    megapixels = (width * height) / 1_000_000
    if megapixels > limit_mp:
        raise ImageTooLargeError(megapixels=megapixels, limit=limit_mp)


def _enforce_pixel_cap(img: Image.Image, limit_mp: float, *, sniffed: SniffedFormat) -> None:
    """``_check_pixel_cap`` plus the structured warning log on rejection."""
    try:
        _check_pixel_cap(img, limit_mp)
    except ImageTooLargeError as e:
        logger.warning(
            "image.pixel_cap_exceeded",
            megapixels=e.megapixels,
            limit=e.limit,
            sniffed=sniffed.value,
        )
        raise


def coerce_mode_for_encode(img: Image.Image) -> Image.Image:
    """Coerce to RGB/RGBA for lossless re-encode.

    Transparency-aware: palette images with tRNS and any mode carrying an
    alpha band go to RGBA; opaque modes (L, CMYK, ...) go to RGB. Blanket
    RGBA would silently inflate opaque grayscale/CMYK with a useless alpha
    channel; blanket RGB (previous behavior) silently dropped palette
    transparency.
    """
    if img.mode in ("RGB", "RGBA"):
        return img
    has_alpha = "A" in img.mode or "transparency" in img.info
    return img.convert("RGBA" if has_alpha else "RGB")


def _convert_to_png_sync(
    data: bytes,
    *,
    max_megapixels: float,
    sniffed: SniffedFormat = SniffedFormat.UNKNOWN,
) -> bytes:
    try:
        with Image.open(io.BytesIO(data)) as source:
            # Header-only size read — checked before exif_transpose/convert/.load()
            # decode the full pixel buffer.
            _enforce_pixel_cap(source, max_megapixels, sniffed=sniffed)
            source.seek(0)
            img = ImageOps.exif_transpose(source)
            img = coerce_mode_for_encode(img)
            buf = io.BytesIO()
            img.save(buf, format="PNG", icc_profile=img.info.get("icc_profile"))
            return buf.getvalue()
    except ImageTooLargeError:
        raise
    except Exception as e:
        raise ImageNormalizationError("File is not a decodable image") from e


def _check_passthrough_pixel_cap(
    data: bytes, max_megapixels: float, *, sniffed: SniffedFormat
) -> None:
    """Header-only size gate for passthrough formats (PNG/JPEG/static WebP).

    Passthrough bytes are never otherwise decoded, but a huge PNG/JPEG would
    still bomb the thumbnailer downstream — reject it here, at the same gate
    as the conversion path. ``Image.open(...).size`` does not decode pixel
    data; ``.load()`` is never called.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            _enforce_pixel_cap(img, max_megapixels, sniffed=sniffed)
    except ImageTooLargeError:
        raise
    except Exception as e:
        raise ImageNormalizationError("File is not a decodable image") from e


def _normalize_sync(
    data: bytes, *, force_png_for_webp: bool = False, max_megapixels: float
) -> NormalizedImage:
    sniffed = sniff_format(data)
    passthrough_format = _PASSTHROUGH_FORMATS.get(sniffed)
    if passthrough_format is not None and not (
        force_png_for_webp and sniffed is SniffedFormat.WEBP
    ):
        _check_passthrough_pixel_cap(data, max_megapixels, sniffed=sniffed)
        return NormalizedImage(
            data=data,
            format=passthrough_format,
            content_type=passthrough_format.content_type,
            converted=False,
            sniffed=sniffed,
        )
    converted_data = _convert_to_png_sync(data, max_megapixels=max_megapixels, sniffed=sniffed)
    return NormalizedImage(
        data=converted_data,
        format=MediaFormat.PNG,
        content_type=MediaFormat.PNG.content_type,
        converted=True,
        sniffed=sniffed,
    )


def _read_image_dimensions_sync(data: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as img:
            # Header-only size read — same as _check_pixel_cap, no full decode.
            return img.size
    except Exception as e:
        raise ImageNormalizationError("File is not a decodable image") from e


async def read_image_dimensions(data: bytes) -> tuple[int, int]:
    """Read (width, height) from image bytes without a full pixel decode.

    Used to derive an i2i output canvas from the source image's aspect when
    the caller omits an explicit aspect_ratio. Raises
    ``ImageNormalizationError`` if the bytes cannot be decoded by Pillow.
    """
    return await asyncio.to_thread(_read_image_dimensions_sync, data)


async def normalize_image(data: bytes, *, max_megapixels: float) -> NormalizedImage:
    """Normalize arbitrary image bytes to a storable format.

    PNG, JPEG, and static WebP pass through unchanged. Everything else
    (animated WebP, HEIF, AVIF, unrecognized headers) is decoded and
    re-encoded as PNG. Raises ``ImageNormalizationError`` if the bytes
    cannot be decoded by Pillow, or ``ImageTooLargeError`` (a subclass) if
    the pixel count exceeds ``max_megapixels``.

    Args:
        data: Raw image bytes.
        max_megapixels: Required cap — callers must pass
            ``Settings.image_max_input_megapixels`` explicitly; this
            function does not own a config default.
    """
    return await asyncio.to_thread(_normalize_sync, data, max_megapixels=max_megapixels)


async def ensure_comfyui_input(data: bytes, *, max_megapixels: float) -> NormalizedImage:
    """Like ``normalize_image``, but also converts static WebP to PNG.

    ComfyUI-side WebP handling is unreliable; this bridge guarantees
    ComfyUI only ever receives PNG or JPEG bytes.

    Args:
        data: Raw image bytes.
        max_megapixels: Required cap — callers must pass
            ``Settings.image_max_input_megapixels`` explicitly; this
            function does not own a config default.
    """
    return await asyncio.to_thread(
        _normalize_sync, data, force_png_for_webp=True, max_megapixels=max_megapixels
    )
